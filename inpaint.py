from PIL import Image
import torch
from huggingface_hub import hf_hub_download
import huggingface_hub
huggingface_hub.cached_download = hf_hub_download
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

def pad_to_square(image: Image.Image, fill=(0, 0, 0)):
    width, height = image.size
    max_side = max(width, height)
    pad_left = (max_side - width) // 2
    pad_top = (max_side - height) // 2

    padded = Image.new("RGB", (max_side, max_side), fill)
    padded.paste(image, (pad_left, pad_top))
    return padded, pad_left, pad_top

def inpaint_image(cropped_image: Image.Image, prompt: str) -> Image.Image:
    cropped_image = cropped_image.convert("RGB")
    original_size = cropped_image.size

    padded_img, pad_left, pad_top = pad_to_square(cropped_image)

    padded_img_resized = padded_img.resize((512, 512))
    
    mask = Image.new("L", (512, 512), color=255)

    result = pipe(prompt=prompt, image=padded_img_resized, mask_image=mask).images[0]

    result = result.resize(padded_img.size)

    final = result.crop((pad_left, pad_top, pad_left + original_size[0], pad_top + original_size[1]))
    return final