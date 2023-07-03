import torch
import requests
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from io import BytesIO
from diffusers.schedulers import DDIMInverseScheduler
from drag_pipeline import DragDiffusionPipeline, MotionSup
from inv_pipeline import StableDiffusionInversePipeline



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


def run_drag_diffusion():
    lora_model_path = "./sd-model-lora"

    pipeline = DragDiffusionPipeline.from_pretrained(
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


    prompt = "A dog sitting on the bench in the park"
    image = pipeline(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        handle_points=handle_points_pt,
        target_points=target_points_pt,
        num_inference_steps=80,
        strength=0.7,
    ).images[0]

    # image = pipeline(
    #     prompt=prompt,
    #     image=init_image,
    #     mask_image=mask_image,
    #     handle_points=handle_points_pt,
    #     target_points=target_points_pt,
    #     num_inference_steps=50
    # )[0]

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


if __name__ == '__main__':
    # test_motion_super()
    run_drag_diffusion()