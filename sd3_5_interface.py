import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3.5-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Omit T5 for speed:
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16
)
pipe.disable_attention_slicing()
pipe.to(device)

current_lora_path = None

def load_lora(lora_path, lora_scale):
    global current_lora_path

    lora_path = lora_path.strip()
    if not lora_path:
        return "No LoRA path specified."

    if lora_path == current_lora_path:
        try:
            pipe.set_lora_scale(lora_scale)
            return f"LoRA already loaded. Updated scale to {lora_scale}."
        except AttributeError:
            return ("LoRA loaded, but `set_lora_scale` not available. Scale unchanged.")
    else:
        pipe.load_lora_weights(lora_path)
        current_lora_path = lora_path
        try:
            pipe.set_lora_scale(lora_scale)
            return f"Loaded LoRA from {lora_path}. Scale set to {lora_scale}."
        except AttributeError:
            return ("Loaded LoRA, but `set_lora_scale` not available. Using default scale.")


def generate_image(prompt, height, width, steps, guidance, seed):
    if seed == -1:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    if device == "cuda":
        with torch.autocast(device):
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
    else:
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]

    return image

# Inline CSS to style only the Generate button via elem_id="generate-btn"
custom_css = """
#generate-btn {
    background-color: orange !important;
    color: black !important;
    font-weight: bold;
}
"""

with gr.Blocks(title="Simple SD3.5 Medium", css=custom_css) as demo:
    gr.Markdown("# Simple Stable Diffusion 3.5 Medium")
    gr.Markdown("Please adhere to the [Stability Community License](https://stability.ai/license).")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value="A surreal painting of a futuristic cityscape at sunrise",
                lines=3
            )
            # Generate Image button *below prompt*, uses elem_id for custom style
            generate_button = gr.Button(
                "Generate Image",
                elem_id="generate-btn"  # matches the CSS ID above
            )

            height_slider = gr.Slider(
                minimum=128, maximum=2048, step=128, value=1024,
                label="Height (px)"
            )
            width_slider = gr.Slider(
                minimum=128, maximum=2048, step=128, value=1024,
                label="Width (px)"
            )
            steps_slider = gr.Slider(
                minimum=1, maximum=50, step=1, value=30,
                label="Number of Inference Steps"
            )
            guidance_slider = gr.Slider(
                minimum=0.0, maximum=20.0, step=0.5, value=4.5,
                label="Guidance Scale"
            )
            seed_number = gr.Number(
                value=-1,
                label="Seed (use -1 for random)"
            )

            # LoRA fields
            lora_path_box = gr.Textbox(
                label="LoRA Path (optional)",
                placeholder="my_lora.safetensors"
            )
            lora_scale_slider = gr.Slider(
                minimum=0.0, maximum=2.0, step=0.1, value=1.0,
                label="LoRA Scale"
            )
            load_lora_button = gr.Button("Load LoRA")
            lora_status = gr.Textbox(
                label="LoRA Status",
                interactive=False
            )

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")

    # Link the Generate button
    generate_button.click(
        fn=generate_image,
        inputs=[prompt, height_slider, width_slider, steps_slider, guidance_slider, seed_number],
        outputs=[output_image]
    )

    # Link the LoRA load button
    load_lora_button.click(
        fn=load_lora,
        inputs=[lora_path_box, lora_scale_slider],
        outputs=[lora_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)

