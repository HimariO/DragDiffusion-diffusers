import torch
import requests
import matplotlib.pyplot as plt
from functools import partial

from PIL import Image, ImageDraw
from io import BytesIO
from diffusers.schedulers import DDIMInverseScheduler

from drag_inpaint_pipeline import DragDiffusionInpaintPipeline, prepare_mask_and_masked_image
from drag_pipeline import DragDiffusionPipeline, load_img
from motion_sup import MotionSup, unet_feat_hook



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

    init_image = Image.open("asset/lora/train/overture-creations-5sI6fQgYIuo.png").resize((512, 512))
    mask_image = Image.open("asset/overture-creations-5sI6fQgYIuo_mask.png").resize((512, 512))

    handle_points = [[150, 245], [150, 270], [170, 255]]
    target_points = [[200, 240], [200, 265], [220, 250]]
    handle_points_pt = torch.tensor([[x / 512, y / 512] for y, x in handle_points])
    target_points_pt = torch.tensor([[x / 512, y / 512] for y, x in target_points])
    handle_points_pt = (handle_points_pt - 0.5) * 2
    target_points_pt = (target_points_pt - 0.5) * 2

    # inv_pipeline = DragDiffusionPipeline(
    #     vae=pipeline.vae,
    #     text_encoder=pipeline.text_encoder,
    #     tokenizer=pipeline.tokenizer,
    #     unet=pipeline.unet,
    #     scheduler=pipeline.scheduler,
    # )

    prompt = "A dog sitting on the bench in the park"
    # src_image = load_img("asset/lora/train/overture-creations-5sI6fQgYIuo.png", target_size=512)
    # image_latents = inv_pipeline.get_image_latents(
    #     src_image.unsqueeze(0).to("cuda").half())
    # text_embeddings = inv_pipeline.get_text_embedding(prompt)


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


def run_drag_diffusion():
    lora_model_path = "./sd-model-lora"

    pipeline = DragDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    # pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.load_attn_procs(lora_model_path)
    pipeline = pipeline.to("cuda")

    init_image = Image.open("asset/lora/train/overture-creations-5sI6fQgYIuo.png").resize((512, 512))
    mask_image = Image.open("asset/overture-creations-5sI6fQgYIuo_mask.png").resize((512, 512))

    handle_points = [[150, 245], [150, 270], [170, 255]]
    target_points = [[200, 240], [200, 265], [220, 250]]
    handle_points_pt = torch.tensor([[x / 512, y / 512] for y, x in handle_points])
    target_points_pt = torch.tensor([[x / 512, y / 512] for y, x in target_points])
    handle_points_pt = (handle_points_pt - 0.5) * 2
    target_points_pt = (target_points_pt - 0.5) * 2

    inv_pipeline = DragDiffusionPipeline(
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
    )

    prompt = "A dog sitting on the bench in the park"
    # src_image = load_img("asset/lora/train/overture-creations-5sI6fQgYIuo.png", target_size=512)
    src_mask, src_masked, src_image = prepare_mask_and_masked_image(init_image, mask_image, 512, 512, return_image=True)
    src_mask = torch.nn.functional.interpolate(src_mask, size=(64, 64))
    
    image_latents = inv_pipeline.get_image_latents(
        src_image.to("cuda").half())
    text_embeddings = inv_pipeline.get_text_embedding(prompt)
    
    reversed_latents = inv_pipeline.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=50,
        early_stop_step=40,
    )

    drag_reverse_latents = inv_pipeline.tune_latent(
        reversed_latents.clone(),
        (1 - src_mask).to('cuda'), 
        inv_pipeline.timesteps_tensor[-1].clone(),
        text_embeddings.clone(),
        {},
        handle_points_pt.to('cuda'),
        target_points_pt.to('cuda'),
        steps=120,
    )

    print("Drag Diff: ", torch.abs(drag_reverse_latents - reversed_latents).mean())
    
    reconstructed_latents = inv_pipeline.backward_diffusion(
        latents=drag_reverse_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=50,
    )
    

    # helper function to show images
    def latents_to_imgs(latents):
        x = inv_pipeline.decode_image(latents)
        x = inv_pipeline.torch_to_numpy(x)
        x = inv_pipeline.numpy_to_pil(x)
        return x
    
    image = latents_to_imgs(reconstructed_latents)[0]

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


if __name__ == '__main__':
    # test_motion_super()
    run_drag_diffusion()