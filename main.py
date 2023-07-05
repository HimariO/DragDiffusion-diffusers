import fire
import torch
import matplotlib.pyplot as plt
from functools import partial

from PIL import Image, ImageDraw
from diffusers.schedulers import DDIMInverseScheduler

from drag_inpaint_pipeline import DragDiffusionInpaintPipeline, prepare_mask_and_masked_image
from drag_pipeline import DragDiffusionPipeline, load_img
from motion_sup import MotionSup, unet_feat_hook


DEFAULT_IMAGE = "asset/lora/train/overture-creations-5sI6fQgYIuo.png"
DEFAULT_MASK = "asset/overture-creations-5sI6fQgYIuo_mask.png"
DEFAULT_PROMPT = "A dog sitting on the bench in the park"


def test_motion_super():
    handle_points = [[1,1]]
    target_points = [[4,4]]
    handle_points_pt = torch.tensor([[x / 8, y / 8] for y, x in handle_points])
    target_points_pt = torch.tensor([[x / 8, y / 8] for y, x in target_points])
    handle_points_pt = (handle_points_pt - 0.5) * 2
    target_points_pt = (target_points_pt - 0.5) * 2
    
    dummy = torch.arange(0, 64).view([1, 8, 8]).float() / 64
    mask = torch.ones_like(dummy)
    ms = MotionSup(handle_points_pt, target_points_pt, dummy, dummy, mask)

    for i in range(1000):
        ms.step(ms.ref_latent)
        ms.search_handle(dummy, ms.ref_latent.data.clone().detach())
    
    print(ms.init_latent)
    print('-' * 20)
    print(ms.ref_latent[0].detach().cpu().numpy())
    print('-' * 20)
    print((ms.ref_latent[0] - ms.init_latent).detach().cpu().numpy())
    
    plt.subplot(1, 2, 1)
    plt.imshow(ms.ref_latent[0].detach().cpu().numpy())
    plt.subplot(1, 2, 2)
    plt.imshow((ms.ref_latent[0] - ms.init_latent).detach().cpu().numpy()[0])
    plt.show()


def run_drag_inpaint_diffusion():
    lora_model_path = "./sd-model-lora"

    pipeline = DragDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    # pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.load_attn_procs(lora_model_path)
    pipeline = pipeline.to("cuda")

    init_image = Image.open(DEFAULT_IMAGE).resize((512, 512))
    mask_image = Image.open(DEFAULT_MASK).resize((512, 512))

    handle_points = [[150, 245], [150, 270], [170, 255]]
    target_points = [[200, 240], [200, 265], [220, 250]]
    handle_points_pt = torch.tensor([[x / 512, y / 512] for y, x in handle_points])
    target_points_pt = torch.tensor([[x / 512, y / 512] for y, x in target_points])
    handle_points_pt = (handle_points_pt - 0.5) * 2
    target_points_pt = (target_points_pt - 0.5) * 2

    prompt = DEFAULT_PROMPT

    image = pipeline(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        handle_points=handle_points_pt,
        target_points=target_points_pt,
        num_inference_steps=80,
        strength=0.7,
    ).images[0]

    # inv_latent, image = pipeline.invert(prompt, image=init_image)
    # image = pipeline.invert(prompt, image=init_image, guidance_scale=.99, num_inference_steps=10)[1][0]
    draw = ImageDraw.Draw(init_image, )
    for y, x in handle_points:
        draw.ellipse((x-5, y-5, x+5, y+5), fill = 'blue', width=10)
    for y, x in target_points:
        draw.ellipse((x-5, y-5, x+5, y+5), fill = 'red', width=10)
    # init_image.show()

    plt.subplot(1, 2, 1)
    plt.imshow(init_image)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()


def run_drag_diffusion(
    lora_model_path="./sd-model-lora",
    image_file=DEFAULT_IMAGE,
    mask_file=DEFAULT_MASK,
    prompt=DEFAULT_PROMPT,
    handle_points=None,
    target_points=None,
):

    pipeline = DragDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    pipeline.unet.load_attn_procs(lora_model_path)
    pipeline = pipeline.to("cuda")

    """
    Prepare inputs
    """
    init_image = Image.open(image_file).resize((512, 512))
    mask_image = Image.open(mask_file).resize((512, 512))

    if handle_points is None:
        handle_points = [[150, 245], [150, 270], [170, 255]]
    if target_points is None:
        target_points = [[200, 240], [200, 265], [220, 250]]
    
    handle_points_pt = torch.tensor([[x / 512, y / 512] for y, x in handle_points])
    target_points_pt = torch.tensor([[x / 512, y / 512] for y, x in target_points])
    handle_points_pt = (handle_points_pt - 0.5) * 2
    target_points_pt = (target_points_pt - 0.5) * 2

    src_mask, src_masked, src_image = prepare_mask_and_masked_image(init_image, mask_image, 512, 512, return_image=True)
    src_mask = torch.nn.functional.interpolate(src_mask, size=(64, 64))
    
    image_latents = pipeline.get_image_latents(
        src_image.to("cuda").half())
    text_embeddings = pipeline.get_text_embedding(prompt)

    """
    Inverse input image latent -> Drag handle point -> Denoise latent
    """
    
    reversed_latents = pipeline.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=50,
        early_stop_step=40,
    )

    drag_reverse_latents = pipeline.tune_latent(
        reversed_latents.clone(),
        (1 - src_mask).to('cuda'), 
        pipeline.timesteps_tensor[-1].clone(),
        text_embeddings.clone(),
        {},
        handle_points_pt.to('cuda'),
        target_points_pt.to('cuda'),
        steps=120,
    )

    print("Draged latent delta: ", torch.abs(drag_reverse_latents - reversed_latents).mean())
    
    reconstructed_latents = pipeline.backward_diffusion(
        latents=drag_reverse_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=50,
    )

    """
    Visualize result
    """

    # helper function to show images
    def latents_to_imgs(latents):
        x = pipeline.decode_image(latents)
        x = pipeline.torch_to_numpy(x)
        x = pipeline.numpy_to_pil(x)
        return x
    
    image = latents_to_imgs(reconstructed_latents)[0]

    draw = ImageDraw.Draw(init_image, )
    for y, x in handle_points:
        draw.ellipse((x-5, y-5, x+5, y+5), fill = 'blue', width=10)
    for y, x in target_points:
        draw.ellipse((x-5, y-5, x+5, y+5), fill = 'red', width=10)

    plt.subplot(1, 2, 1)
    plt.imshow(init_image)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    fire.Fire({
        "drag_diffusion": run_drag_diffusion,
        "drag_inpaint_diffusion": run_drag_inpaint_diffusion,
        "test_motion_super": test_motion_super,
    })