import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_base = "runwayml/stable-diffusion-v1-5"
lora_model_path = "./sd-model-lora"

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(lora_model_path)
pipe.to("cuda")

image = pipe(
    "A dog sitting on the bench in the park", 
    num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}
).images[0]

image.save("blue_pokemon.png")