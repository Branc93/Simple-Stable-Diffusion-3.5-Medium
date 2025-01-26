The application is a user interface built with Gradio that allows you to generate images using two different deep learning pipelines: Stable Diffusion and Flux.

Here's a breakdown of what the app does:

Core Functionality:

Image Generation: The primary function is to generate images from text prompts using either the Stable Diffusion or Flux pipelines.
Pipeline Selection: You can choose between Stable Diffusion and Flux pipelines using a radio button at the top of the interface.
Prompt Input: A text box is provided where you can enter a text prompt describing the image you want to generate. This prompt guides the image generation process.
Settings for Each Pipeline:
Stable Diffusion: You can adjust settings specific to Stable Diffusion, such as:
Use LoRA: Enable or disable the use of LoRA (Low-Rank Adaptation) weights, which can fine-tune the image generation.
Steps: Control the number of inference steps, affecting the quality and generation time.
Guidance Scale: Adjust the guidance scale, which influences how closely the generated image follows the prompt.
Flux: You can adjust settings for the Flux pipeline:
CFG Scale: Control the CFG (Classifier-Free Guidance) scale, similar to guidance scale in Stable Diffusion.
Steps: Control the number of inference steps for the Flux pipeline.
Resolution Control: Sliders are provided to set the desired width and height of the generated images.
Image Preview: The generated image is displayed in an "Generated Image Preview" area. For Flux pipeline, a gallery is used to display potentially multiple images generated during the process.
Output Saving: Generated images are automatically saved to the output directory. The filenames include timestamps and pipeline names (sd or flux) for easy identification.
Progress Bars: Progress bars are displayed during the image generation process for both Stable Diffusion and Flux, giving you visual feedback on the generation progress.
In essence, this application provides a user-friendly way to interact with Stable Diffusion and Flux image generation models, allowing you to create images from text prompts with customizable settings for each pipeline.

To use the app, you would:

Choose a pipeline (Stable Diffusion or Flux).
Enter a text prompt describing the image you want.
Adjust pipeline-specific settings (if desired) within the accordions.
Set the desired image resolution using the width and height sliders.
Click the "Generate Image" button.
View the generated image in the preview area (or gallery for Flux).
Find the saved image files in the output directory.


![image](https://github.com/user-attachments/assets/f5c3ab0f-f168-4f8d-bdd7-c61dbffcd3e4)

# **v0.0.2**
19/01/25

Added enhanced LoRA loading capability. No longer need to manually paste PATH, now upon running the script, all LoRA files that are in .safetensors format, that are in the same location as the main python script, will be visible in the drop-down menu.
Added inpainting tab.

# **v0.0.3**
26/01/25

Dual Pipe-line, optimizations, Stable Diffusion pipeline uses STABLE DIFFUSION 3.5-MEDIUM-TURBO from TensorArt and Flux Pipeline uses FLEX.1-ALPHA from ostris.
Links: 
https://huggingface.co/ostris/Flex.1-alpha
https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo
