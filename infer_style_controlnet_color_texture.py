import torch
from diffusers import ControlNetModel

from pipeline_controlnet_sd_xl import  StableDiffusionXLControlNetPipeline
from ip_adapter import IPAdapterPlusXL
import cv2
from PIL import Image
import os
from ip_adapter import IPAdapterXL

import cv2
import numpy as np
import random

device = "cuda:3"

seed = 1589485

# some examples
# content_path = 'assets/content/cnt2.jpg'
# color_image_path = 'assets/color/2.png'
# texture_image_path = 'assets/texture/7.png'

content_path = 'assets/content/cnt1.png'
color_image_path = 'assets/color/4.png'
texture_image_path = 'assets/texture/11.png'


out_dir_root = r'/disk1/users/jqin/SADis/results/ctrlnettest/'
os.makedirs(out_dir_root, exist_ok=True)



prompt = ''


# f = color_scale*f_clr - color_sub_scale*f_grayclr + texture_scale*f_txtr
color_scale = 1
color_sub_scale =  1  # 0 denotes no gray substraction
texture_scale = 1.2
controlnet_conditioning_scale = 0.4 # 0.6


wct_guidance = 0
wct_starts_step = 0.2 
wct_ends_step = 0.3
wctnoise_add_scale = 0.008


punish_weight = 0.008   # 0.005 0.005, 0.008
punish_type = 'soft-weight'   # highest:0.4,  None,  soft-weight:0.005

# set mask for content preservation
ca_mask=False

ip_scale = 1.0
num_samples = 1
steps = 30


neg_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"



base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
target_blocks=["up_blocks.0.attentions.1"]  #  for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
# ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=target_blocks)

ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16, target_blocks=target_blocks, ca_mask=ca_mask)

# control image
input_image = cv2.imread(content_path)
input_image = cv2.resize(input_image, (1024,1024))
detected_map = cv2.Canny(input_image, 50, 100)  # fox 50 120
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))




color_gt = color_image_path
cnt_name = os.path.basename(content_path).split('.')[0]

clr_name = os.path.basename(color_image_path).split('.')[0]
sty_name2 = os.path.basename(texture_image_path).split('.')[0]
out_dir = os.path.join(out_dir_root, f"cnt[{cnt_name}]")
# out_dir = os.path.join(out_dir_root, f"{prompt}_C[{clr_name}]T[{sty_name2}]_seed{seed}_step{steps}")


os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, f'canny.png'), detected_map)


color_image = Image.open(color_image_path).convert("RGB")
color_image.save(os.path.join(out_dir, os.path.basename(color_image_path)))

cnt_image = Image.open(content_path)
cnt_image.save(os.path.join(out_dir, os.path.basename(content_path)))

style_image2 = Image.open(texture_image_path).convert("RGB")


color_image_gray = color_image.convert("L")
texture_image_gray = style_image2.convert("L")
svname = f"C[{clr_name}]T[{sty_name2}]_c{color_scale}s{color_sub_scale}t{texture_scale}_pg[{punish_type}{punish_weight}]_WCT{wct_guidance}[{wct_starts_step}-{wct_ends_step}T_noise{wctnoise_add_scale}]]_sd[{seed}]_mask{ca_mask}"
save_path = os.path.join(out_dir, svname+".png")


if len(target_blocks)==0:
    save_path = out_dir+f"{prompt}_seed{seed}_sdxlplus.png"


# generate image
images = ip_model.generate(
                        clr_ref_img=color_image,
                        clr_texture_ref_img=color_image_gray,
                        texture_ref_img=texture_image_gray,
                        prompt=prompt,   # , masterpiece, best quality, high quality
                        negative_prompt=neg_prompt,
                        scale=ip_scale,
                        guidance_scale=5,   # 5
                        num_samples=1,
                        num_inference_steps=steps, 
                        seed=seed,
                        color_scale=color_scale,
                        substract_scale=color_sub_scale,
                        texture_scale=texture_scale,
                        wct_guidance=wct_guidance,
                        wct_starts_step=wct_starts_step*steps,
                        wct_ends_step=wct_ends_step*steps,
                        clr_ref_img_dir=color_image_path,
                        sty_ref_img_dir=texture_image_path,
                        wctnoise_add_scale=wctnoise_add_scale,
                        punish_weight=punish_weight,    # 0.005
                        punish_type=punish_type,   #
                        image=canny_map,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        )

images[0].save(save_path)
