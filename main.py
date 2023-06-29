import torch
import requests
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from io import BytesIO
from diffusers.schedulers import DDIMInverseScheduler
from drag_pipeline import DragDiffusionPipeline
from inv_pipeline import StableDiffusionInversePipeline


pipeline = DragDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to("cuda")


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = Image.open("asset/overture-creations-5sI6fQgYIuo.png").resize((512, 512))
mask_image = Image.open("asset/overture-creations-5sI6fQgYIuo_mask.png").resize((512, 512))

handle_points = [[160, 240], [160, 265], [180, 250]]
target_points = [[200, 240], [200, 265], [220, 250]]
handle_points_pt = torch.tensor([[x / 512, y / 512] for y, x in handle_points])
target_points_pt = torch.tensor([[x / 512, y / 512] for y, x in target_points])
handle_points_pt = (handle_points_pt - 0.5) * 2
target_points_pt = (target_points_pt - 0.5) * 2


prompt = "Face of a yellow dog, high resolution, sitting on a park bench"
image = pipeline(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    handle_points=handle_points_pt,
    target_points=target_points_pt,
    num_inference_steps=40
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