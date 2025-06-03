'''
Description: 
Date: 2024-06-25 21:46:54
LastEditTime: 2025-06-03 18:12:33
FilePath: \SADis\infer_style_plus_color_texture.py
'''
import torch
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from PIL import Image
import os
import numpy as np
from ip_adapter import IPAdapterPlusXL
import cv2

#################################################################
# input two images, one for color transfer, the other for texture transfer
# disentangle color：color embedding - gray embedding
#################################################################

color_dir = r'assets/color/'
texture_dir = r'assets/texture/'
save_dir = r'results/'



prompt_list = [
            #    "A photo of a bird",
                'A lighthouse by the sea',
                'A traditional Chinese building'
               ]


steps = 50  # 50 300
seed = 42  # 11  22 44 42  66
device = "cuda:0"
num_samples = 1

####################### trade-off for color and texture ##################
# f = color_scale*f_clr - color_sub_scale*f_grayclr + texture_scale*f_txtr
color_scale = 1.1  
color_sub_scale =  1.1  # 0 denotes no gray substraction
texture_scale = 1.  # greater weight to get better textures
########################################################################


################### regWCT for zt #######################################
wct_guidance = 0.5   # 0.5 by default, other options:0, zt = (1-colorfix_guidances)*zt + colorfix_guidances*regWCT(zt)
# [[1/5, 2/5]] by default, but we found [0.2, 0.3] is better then.
wct_starts_step = 0.2
wct_ends_step = 0.3
wctnoise_add_scale = 0.01   # 0.01 default,  other options:0.005, 0.008  weight of noise addition for regWCT
########################################################################

################# SVD for suppress gray####################################
punish_weight = 0.003   # 0.003 default  other options: 0
punish_type = 'soft-weight'   # there are three options: highest:0.4,  None,  soft-weight:0.003     The number here is its corresponding punish_weights.
########################################################################

######### add mask for foreground #######################################
ca_mask = False    # for material transfer: set True, otherwise False 
ca_mask_thres = 0.3   # top k preserve
########################################################################



# configuration for ip-adapter
# load ip-adapter
# target_blocks=[] # for originalk sdxl
# target_blocks=["block"] #for original IP-Adapter
target_blocks=["up_blocks.0.attentions.1"]  #for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks



color_paths = [os.path.join(color_dir, i) for i in os.listdir(color_dir) if i.endswith('jpg') or i.endswith('png') or i.endswith('jpeg')]
texture_paths = [os.path.join(texture_dir, i) for i in os.listdir(texture_dir) if i.endswith('jpg') or i.endswith('png') or i.endswith('jpeg')]

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
                                base_model_path,
                                torch_dtype=torch.float16,
                                add_watermarker=False,
                                )
pipe.enable_vae_tiling()
ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16, target_blocks=target_blocks, ca_mask=ca_mask)

os.makedirs(save_dir, exist_ok=True)



for prompt in prompt_list:
    for color_image_path in color_paths:
        for sam_i in range(num_samples):
            color_gt = color_image_path
            for style_image_path in texture_paths:
                color_image = Image.open(color_image_path).convert("RGB")
                color_image_gray = color_image.convert("L")
                style_image = Image.open(style_image_path).convert("RGB")
                texture_image2_gray = style_image.convert("L")

                color_name_ext = os.path.basename(color_image_path)
                style_name_ext = os.path.basename(style_image_path)
                svname = f"{prompt}_{color_name_ext}_{style_name_ext}_sd{seed}.jpg"
                save_path = os.path.join(save_dir, svname)
                    

                # set ca mask
                save_atten_dir = os.path.join(os.path.dirname(save_path), f'{prompt}_{color_name_ext}_{style_name_ext}_attenmap')
                if ca_mask:
                    os.makedirs(save_atten_dir, exist_ok=True)
                ip_model.set_ip_variables(save_atten_dir, ca_mask_thres)

                # generate image
                images = ip_model.generate(clr_ref_img=color_image,
                                        prompt=prompt,   # , masterpiece, best quality, high quality
                                        negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, gray color",
                                        scale=1.0,
                                        guidance_scale=5,   # 5
                                        num_samples=1,
                                        num_inference_steps=steps, 
                                        seed=seed,
                                        clr_texture_ref_img = color_image_gray,
                                        substract_scale = color_sub_scale,
                                        color_scale = color_scale,
                                        texture_scale = texture_scale,
                                        texture_ref_img=texture_image2_gray,
                                        wct_guidance = wct_guidance,
                                        wct_starts_step = wct_starts_step*steps,
                                        wct_ends_step = wct_ends_step*steps,
                                        clr_ref_img_dir=color_image_path,
                                        sty_ref_img_dir=style_image_path,
                                        wctnoise_add_scale=wctnoise_add_scale,
                                        punish_weight = punish_weight,    # 0.005
                                        punish_type = punish_type,   #
                                        #neg_content_prompt="a rabbit",
                                        #neg_content_scale=0.5,
                                        )

                images[0].save(save_path)
