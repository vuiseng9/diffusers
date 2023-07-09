


import torch
import intel_extension_for_pytorch as ipex
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchinfo.benchutils import report_timeit_stats

model_id = "runwayml/stable-diffusion-v1-5"
dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=dpm, torch_dtype=torch.bfloat16)

pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)

for i in range(3):
    prompt = "sailing ship in storm by Rembrandt"
    image = pipe(prompt, num_inference_steps=20).images[0]  

report_timeit_stats()
print("End of script.")