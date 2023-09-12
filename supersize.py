"""
Supersizes an image 4x by decomposing the image into a grid, then upscaling the slice
  with Stable Diffusion upscale and img2img pipelines.

Usage:
supersized_image = supersize5(image, prompt, n_prompt, seed=seed, crispy=False, run=1)

Parameters:
image: PIL Image
prompt: String of prompt(s)
n_prompt: String of negative prompts
seed: The seed for the generator to use
crispy: A special mode that overprocesses the input
run: Runmodes 1 and 2 correspond mostly to iteration and strength values
output: PIL Image

"""

from diffusers import DiffusionPipeline, StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionDepth2ImgPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionSAGPipeline, VersatileDiffusionDualGuidedPipeline, StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline
from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers.models import AutoencoderKL
from transformers import pipeline
import numpy as np
from IPython.display import clear_output
from time import sleep
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
import os
from tqdm import tqdm
from exif import Image as Exif_Image
import color_transfer
import colorsys
import skimage
from RealESRGAN import RealESRGAN


def sc(self, clip_input, images) :
    return images, [False for i in images]

# edit StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
safety_checker.StableDiffusionSafetyChecker.forward = sc


def make_gradient2(left=False, right=False, top=False, bottom=False, width=512, height=512):
    """ 
    Use Numpy to make a gradient in 1d, then tile it to 2d
    Fade from 255 to 0, 0 on the side of the True value
    The gradient should only be 64 pixels wide for left and right, 
    64 pixels height for top and bottom, applied to the side of the True value
    
    """
    
    gradient = Image.new('L', (width, height))
    
    if left:
        alpha_gr_left = np.tile(np.linspace(255, 0, 64), (height, 1))
    else:
        alpha_gr_left = np.zeros((height, 64))
    if right:
        alpha_gr_right = np.tile(np.linspace(0, 255, 64), (height, 1))
    else:
        alpha_gr_right = np.zeros((height, 64))
    if top:
        # Same as left, but transpose the array
        alpha_gr_top = np.tile(np.linspace(255, 0, 64), (width, 1)).T
    else:
        alpha_gr_top = np.zeros((64, width))
    if bottom:
        # Same as right, but transpose the array
        alpha_gr_bottom = np.tile(np.linspace(0, 255, 64), (width, 1)).T
    else:
        alpha_gr_bottom = np.zeros((64, width))
    
    alpha_gr_512 = np.zeros((height, width))
    
    # Combine the gradients onto alpha_gr_512, the alpha needs to be combined over the whole image
    
    alpha_gr_512[:, :64] = alpha_gr_left + alpha_gr_512[:, :64]
    alpha_gr_512[:, -64:] = alpha_gr_right + alpha_gr_512[:, -64:]
    alpha_gr_512[:64, :] = alpha_gr_top + alpha_gr_512[:64, :]
    alpha_gr_512[-64:, :] = alpha_gr_bottom + alpha_gr_512[-64:, :]
    
    # Values in alpha_gr_512 should be between 0 and 255, adjust if necessary
    for x in range(width):
        for y in range(height):
            if alpha_gr_512[y, x] > 255:
                alpha_gr_512[y, x] = 255
            elif alpha_gr_512[y, x] < 0:
                alpha_gr_512[y, x] = 0
    
    # invert the alpha so that the gradient is transparent on the side of the True value
    alpha_gr_512 = 255 - alpha_gr_512
    
    alpha_gr_512_img = Image.fromarray(alpha_gr_512.astype(np.uint8))
    
    gradient.putalpha(alpha_gr_512_img)
    
    return gradient


# Slice 'image' into an 8x8 grid slightly overlapping slices
# Upscale each slice to 256x256
# Stitch slices back together

def esrgan_upscale(image, upscale_factor=4):
    """
    Upscale an image using RealESRGAN
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "weights/RealESRGAN_x4plus.pth"
    # model_path = "weights/RealESRGAN_x4plus_anime_6B.pth"
    model = RealESRGAN(device, scale=4)
    model.load_weights(model_path, download=True)
    upscaled_image = model.predict(image)
    return upscaled_image


def supersize6(image, prompt, n_prompt, seed, crispy=False, run=0):
    """
    Upscales a PIL iamge using a diffusion model
    
    Image is upscaled by a factor of original size * factor
    Upscale is achieved by slicing source into a grid of squares, each square needs a 32px overlap with its neighbors
    Overlapping images need to be blended together, radial gradient to transparent in the overlapping region

    Args:
        image (Image): _description_
        prompt (string): _description_
        seed (int): _description_
        factor (int): _description_
    """
    original_width, original_height = image.size
    
    # How many slices total? Each image needs to be 128x128, plus a 32 pixel overlap on each side
    slices_per_row = image.width // 96
    rows_per_column = image.height // 96
    total_slices = slices_per_row * rows_per_column
    total_overlap_pixels_row = 32 * slices_per_row
    total_overlap_pixels_column = 32 * rows_per_column
    save_every = 10
        
    # Calculate the new dimensions, based on the factor, taking into account the overlap
    new_width, new_height = (original_width * 4), (original_height * 4)
    
    print('Input image is {}x{}, new image will be {}x{}'.format(original_width, original_height, new_width, new_height))
    
    # Create the new image with the new dimensions
    supersized_image = Image.new('RGBA', (new_width, new_height))
    
    def make_hint(image, depth_estimator):
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        hint = detected_map.permute(2, 0, 1)
        return hint

    depth_estimator = pipeline("depth-estimation")

    pipe_prior = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
    ).to("cuda")
    pipe_prior.enable_model_cpu_offload()


    pipe = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_model_cpu_offload()


    # Iterate through each column and row, slicing the original image and upscaling it
    for column in range(slices_per_row):
        for row in range(rows_per_column):
            # tqdm progress bar of the total number of slices
            tqdm.write('Upscaling slice {} of {}'.format((column * rows_per_column) + row + 1, total_slices))
            
            # Where does this slice start? Take into account the 32 pixel overlap
            start_coords = (column * 96, row * 96)
            end_coords = (start_coords[0] + 128, start_coords[1] + 128)
            
            # Crop the image to the slice
            image_slice = image.crop(start_coords + end_coords)
            
            # image_slice_hist_img = image_slice.convert("RGBA")
            
            # Save the slice to disk
            # image_slice.save("slice_{}_{}.png".format(column, row))
            
            # Crispy mode makes the images a little more... crazy. We do this by increasing the guidance scale and strength
            if crispy:
                num_inference_steps = 100
                strength = 0.5
            else:
                num_inference_steps = 50
                strength = 0.025
                                    
            # Detail the image again, using the img2img pipeline
            # detailed_slice = pipe(prompt=prompt, negative_prompt=n_prompt, image=image_slice, num_inference_steps=(num_inference_steps * 4), guidance_scale=guidance_scale, generator=generator, output_type="pil", strength=strength).images[0]
            generator = torch.Generator(device="cuda").manual_seed(seed)
            # image_slice = image_slice.resize((1024, 1024), Image.LANCZOS).convert("RGB")
            
            image_slice = esrgan_upscale(image_slice, upscale_factor=4)
            image_slice = esrgan_upscale(image_slice, upscale_factor=4)
            image_slice = image_slice.resize((1024, 1024), Image.LANCZOS).convert("RGB")
            
            # Unsharp mask the detailed slice to get rid of some of the noise
            # image_slice = image_slice.filter(ImageFilter.UnsharpMask(radius=3, percent=30, threshold=3))
            
            hint = make_hint(image_slice, depth_estimator).unsqueeze(0).half().to("cuda")
            img_emb = pipe_prior(prompt=prompt, image=image_slice, strength=0.15, generator=generator)
            negative_emb = pipe_prior(prompt=n_prompt, image=image_slice, strength=1, generator=generator)

            # run controlnet img2img pipeline
            detailed_slice = pipe(
                image=image_slice,
                strength=strength,
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=negative_emb.image_embeds,
                hint=hint,
                num_inference_steps=num_inference_steps,
                generator=generator,
                width=1024,
                height=1024
            ).images[0]
            
            """
            detailed_slice = detailed_slice.resize((512, 512), Image.BICUBIC).convert("RGB")
            hint = make_hint(detailed_slice, depth_estimator).unsqueeze(0).half().to("cuda")
            img_emb = pipe_prior(prompt=prompt, image=detailed_slice, strength=0.25, generator=generator)
            negative_emb = pipe_prior(prompt=n_prompt, image=detailed_slice, strength=1, generator=generator)

            # run controlnet img2img pipeline
            detailed_slice = pipe(
                image=detailed_slice,
                strength=strength,
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=negative_emb.image_embeds,
                hint=hint,
                num_inference_steps=(num_inference_steps * 1),
                generator=generator,
                width=512,
                height=512
            ).images[0]

            # Another unsharp mask
            detailed_slice = detailed_slice.filter(ImageFilter.UnsharpMask(radius=3, percent=30, threshold=3))

            # Desaturate the detailed slice 10%
            detailed_slice = ImageEnhance.Color(detailed_slice).enhance(0.9)
            
            detailed_slice = detailed_slice.resize((768, 768), Image.BICUBIC).convert("RGB")
            hint = make_hint(detailed_slice, depth_estimator).unsqueeze(0).half().to("cuda")
            img_emb = pipe_prior(prompt=prompt, image=detailed_slice, strength=0.25, generator=generator)
            negative_emb = pipe_prior(prompt=n_prompt, image=detailed_slice, strength=1, generator=generator)

            # run controlnet img2img pipeline
            detailed_slice = pipe(
                image=detailed_slice,
                strength=(strength * 2),
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=negative_emb.image_embeds,
                hint=hint,
                num_inference_steps=(num_inference_steps * 2),
                generator=generator,
                width=768,
                height=768
            ).images[0]
            """
            
            detailed_slice = detailed_slice.resize((512, 512), Image.LANCZOS).convert("RGB")
            
            # Keep only the center of the mask, crop to 512x512. the center is 1024x1024, the image is 2048x2048
            # only mask slices that aren't on the edge of the image
            if column > 0 and column < slices_per_row - 1 and row > 0 and row < rows_per_column - 1:
                # This is a slice in the middle of the image, all edges need to be masked
                mask = make_gradient2(left=True, right=True, top=True, bottom=True)
            if column == 0 and row == 0:
                # This is the top left corner, create a gradient mask that is transparent on the right and bottom
                mask = make_gradient2(left=False, right=True, top=False, bottom=True)
            elif column == slices_per_row - 1 and row == 0:
                # This is the top right corner, create a gradient mask that is transparent on the left and bottom
                mask = make_gradient2(left=True, right=False, top=False, bottom=True)
            elif column == 0 and row == rows_per_column - 1:
                # This is the bottom left corner, create a gradient mask that is transparent on the right and top
                mask = make_gradient2(left=False, right=True, top=True, bottom=False)
            elif column == slices_per_row - 1 and row == rows_per_column - 1:
                # This is the bottom right corner, create a gradient mask that is transparent on the left and top
                mask = make_gradient2(left=True, right=False, top=True, bottom=False)
            elif column == 0:
                # This is the left edge, create a gradient mask that is transparent on the right
                mask = make_gradient2(left=False, right=True, top=True, bottom=True)
            elif column == slices_per_row - 1:
                # This is the right edge, create a gradient mask that is transparent on the left
                mask = make_gradient2(left=True, right=False, top=True, bottom=True)
            elif row == 0:
                # This is the top edge, create a gradient mask that is transparent on the bottom
                mask = make_gradient2(left=True, right=True, top=False, bottom=True)
            elif row == rows_per_column - 1:
                # This is the bottom edge, create a gradient mask that is transparent on the top
                mask = make_gradient2(left=True, right=True, top=True, bottom=False)
            
            detailed_slice.save("temp_slice.png")
            
            # Paste the detailed slice onto the supersized image, they overlap by 32 pixels originally, now they overlap by 128 pixels
            destination_coords = (column * 4 * (128 - 32), row * 4 * (128 -32))
            supersized_image.paste(detailed_slice, destination_coords, mask=mask)
            
            # supersized_image.save("temp_supersized.png")
            little_supersized = supersized_image.resize((1024, 1024), Image.LANCZOS)
            little_supersized.save("temp_supersized.png")
            clear_output(wait=True)
            
    return(supersized_image)


def supersize7(image, prompt, n_prompt, seed, crispy=False, run=0):
    """
    Upscales a PIL iamge using a diffusion model
    
    Image is upscaled by a factor of original size * factor
    Upscale is achieved by slicing source into a grid of squares, each square needs a 32px overlap with its neighbors
    Overlapping images need to be blended together, radial gradient to transparent in the overlapping region

    Args:
        image (Image): _description_
        prompt (string): _description_
        seed (int): _description_
        factor (int): _description_
    """
    
    # Adjust prompt and n_prompt to indicate they are zoomed in like a macro lens or telephoto lens, and the image is not blurry
    prompt = prompt + ", zoomed in, macro lens, not blurry"
    n_prompt = n_prompt + ", blurry, missing detail"
    
    original_width, original_height = image.size
    
    # How many slices total? Each image needs to be 128x128, plus a 32 pixel overlap on each side
    slices_per_row = image.width // 96
    rows_per_column = image.height // 96
    total_slices = slices_per_row * rows_per_column
    total_overlap_pixels_row = 32 * slices_per_row
    total_overlap_pixels_column = 32 * rows_per_column
    save_every = 10
        
    # Calculate the new dimensions, based on the factor, taking into account the overlap
    new_width, new_height = (original_width * 4), (original_height * 4)
    
    print('Input image is {}x{}, new image will be {}x{}'.format(original_width, original_height, new_width, new_height))
    
    # Create the new image with the new dimensions
    supersized_image = Image.new('RGBA', (new_width, new_height))
    
    diffuser_list = [
        'dreamlike-art/dreamlike-photoreal-2.0',
        'prompthero/openjourney-v4',
        'hakurei/waifu-diffusion',
        'WarriorMama777/AbyssOrangeMix',
        'stabilityai/stable-diffusion-2-1',
        'runwayml/stable-diffusion-v1-5'
        ]
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(diffuser_list[4], torch_dtype=torch.float16, vae=vae).to("cuda")


    # Iterate through each column and row, slicing the original image and upscaling it
    for column in range(slices_per_row):
        for row in range(rows_per_column):
            # tqdm progress bar of the total number of slices
            tqdm.write('Upscaling slice {} of {}'.format((column * rows_per_column) + row + 1, total_slices))
            
            # Where does this slice start? Take into account the 32 pixel overlap
            start_coords = (column * 96, row * 96)
            end_coords = (start_coords[0] + 128, start_coords[1] + 128)
            
            # Crop the image to the slice
            image_slice = image.crop(start_coords + end_coords)
            
            # image_slice_hist_img = image_slice.convert("RGBA")
            
            # Save the slice to disk
            # image_slice.save("slice_{}_{}.png".format(column, row))
            
            # Crispy mode makes the images a little more... crazy. We do this by increasing the guidance scale and strength
            if crispy:
                guidance_scale = 3.0
                num_inference_steps = 400
                strength = 0.45
            else:
                guidance_scale = 0.2
                num_inference_steps = 100
                strength = 0.1
            
            if run and run == 2:
                guidance_scale = guidance_scale * 2
                num_inference_steps = 200
                strength = strength * 1.15
            
            if run and run == 3:
                guidance_scale = guidance_scale * 4
                num_inference_steps = 500
                strength = strength * 1.5
                                    
            # Detail the image again, using the img2img pipeline
            # detailed_slice = pipe(prompt=prompt, negative_prompt=n_prompt, image=image_slice, num_inference_steps=(num_inference_steps * 4), guidance_scale=guidance_scale, generator=generator, output_type="pil", strength=strength).images[0]
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            image_slice = esrgan_upscale(image_slice, upscale_factor=4)
            
            image_slice = image_slice.resize((768, 768), Image.BICUBIC).convert("RGB")
            detailed_slice = pipe(
                prompt=prompt, 
                negative_prompt=n_prompt, 
                image=image_slice, 
                num_inference_steps=(num_inference_steps), 
                guidance_scale=guidance_scale, 
                generator=generator, 
                output_type="pil", 
                strength=(strength * 2)
                ).images[0]
            
            detailed_slice = image_slice.resize((512, 512), Image.BICUBIC).convert("RGB")
            
            # Decrease saturation 10%
            # detailed_slice = ImageEnhance.Color(detailed_slice).enhance(0.9)
            
            # Sharpen the detailed slice
            # detailed_slice = detailed_slice.filter(ImageFilter.UnsharpMask(radius=3, percent=30, threshold=3))
            
            detailed_slice = pipe(
                prompt=prompt, 
                negative_prompt=n_prompt, 
                image=image_slice, 
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale, 
                generator=generator, 
                output_type="pil", 
                strength=strength,
                ).images[0]
            
            detailed_slice = esrgan_upscale(detailed_slice, upscale_factor=4)
    
            detailed_slice = detailed_slice.resize((512, 512), Image.BICUBIC)
            
            detailed_slice.convert("RGBA")
                        
            # Keep only the center of the mask, crop to 512x512. the center is 1024x1024, the image is 2048x2048
            # only mask slices that aren't on the edge of the image
            if column > 0 and column < slices_per_row - 1 and row > 0 and row < rows_per_column - 1:
                # This is a slice in the middle of the image, all edges need to be masked
                mask = make_gradient2(left=True, right=True, top=True, bottom=True)
            if column == 0 and row == 0:
                # This is the top left corner, create a gradient mask that is transparent on the right and bottom
                mask = make_gradient2(left=False, right=True, top=False, bottom=True)
            elif column == slices_per_row - 1 and row == 0:
                # This is the top right corner, create a gradient mask that is transparent on the left and bottom
                mask = make_gradient2(left=True, right=False, top=False, bottom=True)
            elif column == 0 and row == rows_per_column - 1:
                # This is the bottom left corner, create a gradient mask that is transparent on the right and top
                mask = make_gradient2(left=False, right=True, top=True, bottom=False)
            elif column == slices_per_row - 1 and row == rows_per_column - 1:
                # This is the bottom right corner, create a gradient mask that is transparent on the left and top
                mask = make_gradient2(left=True, right=False, top=True, bottom=False)
            elif column == 0:
                # This is the left edge, create a gradient mask that is transparent on the right
                mask = make_gradient2(left=False, right=True, top=True, bottom=True)
            elif column == slices_per_row - 1:
                # This is the right edge, create a gradient mask that is transparent on the left
                mask = make_gradient2(left=True, right=False, top=True, bottom=True)
            elif row == 0:
                # This is the top edge, create a gradient mask that is transparent on the bottom
                mask = make_gradient2(left=True, right=True, top=False, bottom=True)
            elif row == rows_per_column - 1:
                # This is the bottom edge, create a gradient mask that is transparent on the top
                mask = make_gradient2(left=True, right=True, top=True, bottom=False)
            detailed_slice.save("temp_slice.png")
            
            # Paste the detailed slice onto the supersized image, they overlap by 32 pixels originally, now they overlap by 128 pixels
            destination_coords = (column * 4 * (128 - 32), row * 4 * (128 -32))
            supersized_image.paste(detailed_slice, destination_coords, mask=mask)
            
            # supersized_image.save("temp_supersized.png")
            little_supersized = supersized_image.resize((1024, 1024), Image.LANCZOS)
            little_supersized.save("temp_supersized.png")
            clear_output(wait=True)
            
    return(supersized_image)
