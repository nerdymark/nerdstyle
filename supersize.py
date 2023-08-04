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
run: Runmodes 1 and 2 correspond to rgb color and sharpness techniques,
     can be useful if you are supersizing a supersized image.
     Denoising techniques used by Stable Diffusion tend to lose color 
     fidelity after multiple passes.
output: PIL Image

"""
from diffusers import DiffusionPipeline, StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionDepth2ImgPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionSAGPipeline, VersatileDiffusionDualGuidedPipeline, StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers.models import AutoencoderKL
import numpy as np
from IPython.display import clear_output
from time import sleep
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
import os
from tqdm import tqdm
from exif import Image as Exif_Image


def sc(self, clip_input, images) :
    return images, [False for i in images]

# edit StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
safety_checker.StableDiffusionSafetyChecker.forward = sc


def make_gradient(left=False, right=False, top=False, bottom=False, width=512, height=512):
    """ 
    Use Numpy to make a gradient in 1d, then tile it to 2d
    Fade from 255 to 0, 0 on the side of the True value
    The gradient should only be 32 pixels wide for left and right, 
    32 pixels height for top and bottom, applied to the side of the True value
    
    """
    
    gradient = Image.new('L', (width, height))
    
    if left:
        alpha_gr_left = np.tile(np.linspace(255, 0, 32), (height, 1))
    else:
        alpha_gr_left = np.zeros((height, 32))
    if right:
        alpha_gr_right = np.tile(np.linspace(0, 255, 32), (height, 1))
    else:
        alpha_gr_right = np.zeros((height, 32))
    if top:
        # Same as left, but transpose the array
        alpha_gr_top = np.tile(np.linspace(255, 0, 32), (width, 1)).T
    else:
        alpha_gr_top = np.zeros((32, width))
    if bottom:
        # Same as right, but transpose the array
        alpha_gr_bottom = np.tile(np.linspace(0, 255, 32), (width, 1)).T
    else:
        alpha_gr_bottom = np.zeros((32, width))
    
    alpha_gr_512 = np.zeros((height, width))
    
    # Combine the gradients onto alpha_gr_512, the alpha needs to be combined over the whole image
    
    alpha_gr_512[:, :32] = alpha_gr_left + alpha_gr_512[:, :32]
    alpha_gr_512[:, -32:] = alpha_gr_right + alpha_gr_512[:, -32:]
    alpha_gr_512[:32, :] = alpha_gr_top + alpha_gr_512[:32, :]
    alpha_gr_512[-32:, :] = alpha_gr_bottom + alpha_gr_512[-32:, :]
    
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


def supersize5(image, prompt, n_prompt, seed, crispy=False, run=0):
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
        
    # Create the upscaler
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16).to("cuda")
    # if one wants to set `leave=False`
    upscaler.set_progress_bar_config(leave=False)

    # if one wants to disable `tqdm`
    upscaler.set_progress_bar_config(disable=True)
    
    # Create the img2img pipeline
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained('prompthero/openjourney-v4', torch_dtype=torch.float16).to("cuda")
    # 
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    
    diffuser_list = [
        'dreamlike-art/dreamlike-photoreal-2.0',
        'prompthero/openjourney-v4',
        'hakurei/waifu-diffusion',
        'WarriorMama777/AbyssOrangeMix',
        'stabilityai/stable-diffusion-2-1',
        'runwayml/stable-diffusion-v1-5'
        ]
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(diffuser_list[5], torch_dtype=torch.float16, vae=vae).to("cuda")
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


    # if one wants to set `leave=False`
    pipe.set_progress_bar_config(leave=False)

    # if one wants to disable `tqdm`
    pipe.set_progress_bar_config(disable=True)
    # Hide the pipeline progress bar
    # pipe.progress_bar = False
    
    generator = torch.manual_seed(seed)    

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
            
            # Save the slice to disk
            # image_slice.save("slice_{}_{}.png".format(column, row))
            
            # Crispy mode makes the images a little more... crazy. We do this by increasing the guidance scale and strength
            if crispy:
                guidance_scale = 3.0
                num_inference_steps = 100
                strength = 0.5
            else:
                guidance_scale = 0.025
                num_inference_steps = 40
                strength = 0.025
            
            # Upscale the slice, the resulting image should be 256x256
            image_slice = upscaler(prompt=prompt, negative_prompt=n_prompt, image=image_slice, num_inference_steps=num_inference_steps, guidance_scale=0.0, generator=generator, output_type="pil").images[0]
            image_slice = image_slice.resize((256, 256), Image.BICUBIC)
            
            # Detail the slice using the img2img pipeline
            image_slice = pipe(prompt=prompt, negative_prompt=n_prompt, image=image_slice, num_inference_steps=(num_inference_steps * 2), guidance_scale=guidance_scale, generator=generator, output_type="pil", strength=strength).images[0]

            # Upscale again, the slice should now be 512x512
            image_slice = upscaler(prompt=prompt, negative_prompt=n_prompt, image=image_slice, num_inference_steps=num_inference_steps, guidance_scale=0.0, generator=generator, output_type="pil").images[0]
            image_slice = image_slice.resize((512, 512), Image.BICUBIC)
                        
            # Detail the image again, using the img2img pipeline
            detailed_slice = pipe(prompt=prompt, negative_prompt=n_prompt, image=image_slice, num_inference_steps=(num_inference_steps * 2), guidance_scale=guidance_scale, generator=generator, output_type="pil", strength=strength).images[0]
            
            # Resize the original slice to 512x512
            original_slice = image_slice.resize((512, 512), Image.BICUBIC)
            original_slice = original_slice.convert("RGBA")
            # Correct the color of the detailed slice to match the original slice
            # Make the detailed slice black and white
            detailed_slice = detailed_slice.convert("L")
            detailed_slice = detailed_slice.convert("RGBA")

            # Increase the brightness and contrast of the detailed slice
            detailed_slice = ImageEnhance.Brightness(detailed_slice).enhance(5.5)
            # detailed_slice = ImageEnhance.Contrast(detailed_slice).enhance(-1.5)
            
            # Transfer the color of the original slice to the detailed slice
            detailed_slice = ImageChops.multiply(detailed_slice, original_slice)
            
            detailed_slice.convert("RGBA")
            
            # if Crispy mode is enabled, we need to do a little more work - 
            # 1. Multiply the detailed slice by the original image
            # if crispy:
                # Make the slice Black and White
                # image_filter = ImageEnhance.Color(detailed_slice)
                # detailed_slice = image_filter.enhance(0)
                    
                
                # Multiply the slice by the original image
                # detailed_slice = ImageChops.multiply(detailed_slice, image_slice)
            
            
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
            # else:
            #     # Create a blank alpha mask
            #     mask = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                
            # Paste the mask onto the detailed slice's alpha channel
            # detailed_slice.putalpha(mask)
            
            # TODO: Sample the overlapping 32 pixels from each overlapping edge, compare the rgb histograms of the overlapping pixels. If they don't average out to the same value, then we need to adjust the histogram of the current slice.
            # Does the left edge overlap with another slice?
            # Crop the left edge of the current slice, 32 pixels wide
            # if column > 0:
            #     current_left_edge = detailed_slice.crop((0, 0, 32, 512))
            #     current_left_overlap = supersized_image.crop((column * 4 * (128 - 32), row * 4 * (128 -32), column * 4 * (128 - 32) + 32, row * 4 * (128 -32) + 512))
            #     
            #     # Compare the histograms of the left edge of the current slice and the left overlap
            #     current_left_edge_histogram = current_left_edge.histogram()
            #     current_left_overlap_histogram = current_left_overlap.histogram()
            #     
            #     # If the histograms are different, adjust the current slice's histogram to match the overlap
            #     histogram_difference = np.array(current_left_edge_histogram) - np.array(current_left_overlap_histogram)
            #     if histogram_difference.any():
            #         # The histograms are different, adjust the current slice's histogram to match the overlap
            #       blah
            
            # Does the right edge overlap with another slice?
            # Does the top edge overlap with another slice?
            # Does the bottom edge overlap with another slice?
            
            # Save a copy of the detailed slice for debugging
            
            
            
            if run == 1:
                # Color correct the detailed slice to be slightly bluer and less yellow
                r, g, b, a = detailed_slice.split()
                # Swap the green and blue channels
                # g, b = b, g
            
                r = r.point(lambda i: i * 0.75)
                g = g.point(lambda i: i * 1.3)
                b = b.point(lambda i: i * 1.3)
            
                detailed_slice = Image.merge("RGBA", (r, g, b, a))
            if run == 2:
                # De-orange the detailed slice
                r, g, b, a = detailed_slice.split()
                r = r.point(lambda i: i * 0.75)
                g = g.point(lambda i: i * 1.1)
                b = b.point(lambda i: i * 1.2)
                detailed_slice = Image.merge("RGBA", (r, g, b, a))
            
            # Enhance saturation
            # detailed_slice = ImageEnhance.Color(detailed_slice).enhance(1.5)
            
            # Brighten by 25%
            # detailed_slice = ImageEnhance.Brightness(detailed_slice).enhance(1.25)
            
            if run != 2:
                # Sharpen the detailed slice
                detailed_slice = detailed_slice.filter(ImageFilter.UnsharpMask(radius=1, percent=10, threshold=3))
                # Add some uniform noise to the detailed slice
                detailed_slice = detailed_slice.filter(ImageFilter.GaussianBlur(radius=1))
            
            detailed_slice.save("temp_slice.png")
            
            # Paste the detailed slice onto the supersized image, they overlap by 32 pixels originally, now they overlap by 128 pixels
            destination_coords = (column * 4 * (128 - 32), row * 4 * (128 -32))
            supersized_image.paste(detailed_slice, destination_coords, mask=mask)
            
            # supersized_image.save("temp_supersized.png")
            little_supersized = supersized_image.resize((1024, 1024), Image.LANCZOS)
            little_supersized.save("temp_supersized.png")
            clear_output(wait=True)
        
    # Downscale to half supersized_image.shape
    # supersized_image = supersized_image.resize((new_width // 2, new_height // 2), Image.LANCZOS)
    
            
    return(supersized_image)
