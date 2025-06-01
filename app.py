import os
import spaces
import gradio as gr
from src.data_processing import pil_to_tensor, tensor_to_pil
from PIL import Image
from src.model_processing import get_model
from huggingface_hub import snapshot_download
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

MODEL_DIR = "./VTBench_models"
# if not os.path.exists(MODEL_DIR):
#     print("Downloading VTBench_models from Hugging Face...")
#     snapshot_download(
#         repo_id="huaweilin/VTBench_models",
#         local_dir=MODEL_DIR,
#         local_dir_use_symlinks=False
#     )
#     print("Download complete.")

example_image_paths = [f"assets/app_examples/{i}.png" for i in range(0, 5)]

model_name_mapping = {
    "SD3.5L": "SD3.5L",
    "chameleon": "Chameleon",
    # "flowmo_lo": "FlowMo Lo",
    # "flowmo_hi": "FlowMo Hi",
    # "gpt4o": "GPT-4o",
    "janus_pro_1b": "Janus Pro 1B/7B",
    "llamagen-ds8": "LlamaGen ds8",
    "llamagen-ds16": "LlamaGen ds16",
    "llamagen-ds16-t2i": "LlamaGen ds16 T2I",
    "maskbit_16bit": "MaskBiT 16bit",
    "maskbit_18bit": "MaskBiT 18bit",
    "open_magvit2": "OpenMagViT",
    "titok_b64": "Titok-b64",
    "titok_bl64": "Titok-bl64",
    "titok_s128": "Titok-s128",
    "titok_bl128": "Titok-bl128",
    "titok_l32": "Titok-l32",
    "titok_sl256": "Titok-sl256",
    "var_256": "VAR-256",
    "var_512": "VAR-512",
    "FLUX.1-dev": "FLUX.1-dev",
    "infinity_d32": "Infinity-d32",
    "infinity_d64": "Infinity-d64",
    "bsqvit": "BSQ-VIT",
}

display_to_internal = {v: k for k, v in model_name_mapping.items()}

def load_model(model_name):
    model, data_params = get_model(MODEL_DIR, model_name)
    model = model.to(device)
    model.eval()
    return model, data_params

# model_dict = {
#     model_name: load_model(model_name)
#     for model_name in model_name_mapping
# }

placeholder_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

@spaces.GPU
def process_selected_models(uploaded_image, selected_display_names):
    if uploaded_image is None:
        return [gr.update(value="⚠️  Please upload an image before processing.", visible=True)] + \
               [gr.update() for _ in model_name_mapping]

    if not selected_display_names:
        return [gr.update(value="⚠️  Please select at least one model.", visible=True)] + \
               [gr.update() for _ in model_name_mapping]

    selected_results = []
    placeholder_results = []

    selected_internal = [display_to_internal[d] for d in selected_display_names]

    for model_name in model_name_mapping:
        label = model_name_mapping[model_name]

        if model_name in selected_internal:
            try:
                model, data_params = load_model(model_name)
                pixel_values = pil_to_tensor(uploaded_image, **data_params).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(pixel_values)[0]
                reconstructed_image = tensor_to_pil(output[0].cpu(), **data_params)

                del model, pixel_values, output
                torch.cuda.empty_cache()

                result = gr.update(value=reconstructed_image, label=label)
            except Exception as e:
                print(f"Error in model {model_name}: {e}")
                result = gr.update(value=placeholder_image, label=f"{label} (Error)")
            selected_results.append(result)
        else:
            result = gr.update(value=placeholder_image, label=f"{label} (Not selected)")
            placeholder_results.append(result)

    return [gr.update(visible=False)] + selected_results + placeholder_results


with gr.Blocks() as demo:
    gr.Markdown("## VTBench")
    gr.Markdown("---")

    gr.Markdown("<span style='color:red; font-weight: bold;'>⚠️ ⚠️ ⚠️  If you encounter any errors, please try again — it usually works on the second attempt.</span>")
    gr.Markdown("👋 **Welcome to VTBench!** Upload an image, select models, and click 'Start Processing' to compare results side by side.")
    gr.Markdown("🔗 **Check out our GitHub repo:** [https://github.com/huawei-lin/VTBench](https://github.com/huawei-lin/VTBench)")
    with gr.Accordion("📘 Full Instructions", open=False):
        gr.Markdown("""
**VTBench User Guide**

- **Upload an image** or click one of the example images.
- **Select one or more models** from the list.
- Click **Start Processing** to run inference.
- Selected model outputs appear first, others show placeholders.

⚠️  *Each model is downloaded on first use. Please wait patiently the first time you run a model.*
""")

    image_input = gr.Image(
        type="pil",
        label="Upload an image",
        width=512,
        height=512,
    )

    gr.Markdown("### Click on an example image to use it as input:")
    example_rows = [example_image_paths[i:i+5] for i in range(0, len(example_image_paths), 5)]
    for row in example_rows:
        with gr.Row():
            for path in row:
                ex_img = gr.Image(
                    value=path,
                    show_label=False,
                    interactive=True,
                    width=256,
                    height=256,
                )

                def make_loader(p=path):
                    def load_img():
                        return Image.open(p)
                    return load_img

                ex_img.select(fn=make_loader(), outputs=image_input)

    gr.Markdown("---")
    gr.Markdown("⚠️  **The more models you select, the longer the processing time will be.**")
    gr.Markdown("*Note: Each model is downloaded on first use. Subsequent uses will load from cache and run faster.*")

    display_names = list(model_name_mapping.values())
    default_selected = ["SD3.5L", "Chameleon", "Janus Pro 1B/7B"]

    model_selector = gr.CheckboxGroup(
        choices=display_names,
        label="Select models to run",
        value=default_selected,
        interactive=True,
    )

    status_output = gr.Markdown("", visible=False)
    run_button = gr.Button("Start Processing")

    image_outputs = []
    model_names_ordered = list(model_name_mapping.keys())
    n_columns = 5
    output_rows = [model_names_ordered[i:i+n_columns] for i in range(0, len(model_names_ordered), n_columns)]

    with gr.Column():
        for row in output_rows:
            with gr.Row():
                for model_name in row:
                    display_name = model_name_mapping[model_name]
                    out_img = gr.Image(
                        label=f"{display_name} (Not run)",
                        value=placeholder_image,
                        width=512,
                        height=512,
                    )
                    image_outputs.append(out_img)


    run_button.click(
        fn=process_selected_models,
        inputs=[image_input, model_selector],
        outputs=[status_output] + image_outputs
    )

demo.launch()

