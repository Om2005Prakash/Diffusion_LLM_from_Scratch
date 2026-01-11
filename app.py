import argparse
import torch
from transformers import AutoModelForMaskedLM
from tokenizer import get_tokenizer
from safetensors.torch import load_file
from safetensors.torch import load_file
import gradio as gr
import logging
import random

# Re-using logic from inference.py
from inference import prepare_conditional_tokens_for_inference, prepare_unconditional_tokens_for_inference, clean_text

def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

logging.getLogger("transformers").setLevel(logging.ERROR)

def load_model_and_tokenizer(path_to_weights, hf_model_name, device="cuda"):
    tokenizer = get_tokenizer(hf_model_name)
    model = AutoModelForMaskedLM.from_pretrained(hf_model_name) # Load to CPU first to avoid memory spikes if reusing
    model.resize_token_embeddings(len(tokenizer))
    
    state_dict = load_file(path_to_weights)
    model.load_state_dict(state_dict, strict=False)
    model.to(device) # Move to device
    model.tie_weights()
    model.eval()
    return model, tokenizer

@torch.inference_mode()
def inference_stream(model, tokenizer, num_steps, strategy, device, prompt, seq_len, seed):
    # Set seed
    seed_everything(seed)
    
    # Prepare tokens
    if prompt:
        input_tokens, mask, attention_mask = prepare_conditional_tokens_for_inference(
            seq_len, tokenizer, prompt, device=device
        )
    else:
        input_tokens, mask, attention_mask = prepare_unconditional_tokens_for_inference(
            seq_len, tokenizer.mask_token_id, device=device
        )
        
    original_mask = mask.clone()
    times = torch.linspace(1, 0, num_steps + 1, device=device)
    
    for t, s in zip(times[:-1], times[1:]):
        # Model forward
        logits = model(input_tokens, attention_mask=attention_mask).logits
        
        if strategy == "backward":
            probs = torch.softmax(logits[mask], dim=-1)
            input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            remask_probs = torch.rand_like(mask, dtype=torch.float, device=device)
            remask_probs = (remask_probs < s/t)
            mask = mask & remask_probs
            input_tokens[mask] = tokenizer.mask_token_id
            
        elif strategy == "predictor_corrector":
            # Predictor
            probs = torch.softmax(logits[mask], dim=-1)
            input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            remask_probs = torch.rand_like(mask, dtype=torch.float, device=device)
            remask_decision = (remask_probs < s/t)
            mask = mask & remask_decision
            input_tokens[mask] = tokenizer.mask_token_id
            
            # Corrector
            n_corrector_steps = 2
            corrector_step_size = 0.5 * (t - s) / (1 - s)
            
            if n_corrector_steps > 0 and s > 0.3:
                for _ in range(n_corrector_steps):
                    known_mask = ~mask ^ ~original_mask
                    noise_rng = torch.rand_like(known_mask, dtype=torch.float, device=device)
                    to_remask = known_mask & (noise_rng < corrector_step_size)
                    
                    input_tokens[to_remask] = tokenizer.mask_token_id
                    corr_logits = model(input_tokens, attention_mask=attention_mask).logits
                    corr_probs = torch.softmax(corr_logits[to_remask], dim=-1)
                    input_tokens[to_remask] = torch.multinomial(corr_probs, num_samples=1).squeeze(-1)
        
        # Decode for streaming
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_tokens[0])
        cleaned_tokens = []
        for tok in decoded_tokens:
            if tok == tokenizer.mask_token:
                cleaned_tokens.append(tok)
            elif tok in tokenizer.all_special_tokens:
                continue
            else:
                cleaned_tokens.append(tok)
        decoded_after = tokenizer.convert_tokens_to_string(cleaned_tokens)
        
        if prompt:
             # Remove prompt for cleaner display
            assistant_text = decoded_after.replace(prompt, "").strip()
            # Clean artifacts
            assistant_text = clean_text(assistant_text)
            yield assistant_text
        else:
            yield decoded_after

# Global model cache to avoid reloading
MODEL_CACHE = {}

def run_app(safetensors_path, hf_model_name, device, prompt, seq_len, num_steps, strategy, use_manual_seed, seed):
    # Handle random seed
    if not use_manual_seed:
        seed = random.randint(0, 2**32 - 1)

    # Load model if needed
    cache_key = (safetensors_path, hf_model_name, device)
    if cache_key not in MODEL_CACHE:
        try:
            model, tokenizer = load_model_and_tokenizer(safetensors_path, hf_model_name, device)
            MODEL_CACHE[cache_key] = (model, tokenizer)
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    model, tokenizer = MODEL_CACHE[cache_key]
    
    # Run inference generator
    try:
        # Yield status info first (optional UX improvement)
        yield f"Using Seed: {seed}..."
        
        for output in inference_stream(model, tokenizer, num_steps, strategy, device, prompt, seq_len, seed):
            yield output
    except Exception as e:
        yield f"Error during inference: {str(e)}"

# Gradio UI
css = """
.gradio-container {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
"""

with gr.Blocks(title="Diffusion Quote Generator") as demo:
    gr.Markdown("# Diffusion Quote Generator")
    gr.Markdown("Generating text by iteratively removing noise (mask tokens).")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", value="Generate a quote on inspiration")
            
            gr.Markdown("### Quick Prompts")
            selected_topics = gr.State(["Inspiration"])
            topic_buttons = []
            
            # Helper logic for toggling topics
            def toggle_topic(topic, current_selected):
                if topic in current_selected:
                    current_selected.remove(topic)
                    variant = "secondary"
                else:
                    current_selected.append(topic)
                    variant = "primary"
                
                # Dynamic prompt generation
                if not current_selected:
                    new_prompt = "Generate a quote on hope"
                else:
                    topics_str = ", ".join(current_selected)
                    new_prompt = f"Generate a quote on {topics_str.lower()}"
                
                return new_prompt, current_selected, gr.Button(variant=variant)
            
            with gr.Row():
                topics = ["Life", "Friendship", "Love", "Inspiration", "Happiness", "Suffering", "Hope", "Faith"]
                for topic in topics:
                    # check if active
                    default_variant = "primary" if topic == "Inspiration" else "secondary"
                    btn = gr.Button(topic, variant=default_variant)
                    topic_buttons.append(btn)
                    
                    # Connect click event
                    # Inputs: current state. Outputs: prompt, state, button itself
                    btn.click(
                        fn=toggle_topic,
                        inputs=[gr.State(topic), selected_topics],
                        outputs=[prompt_input, selected_topics, btn]
                    )

            with gr.Accordion("Advanced Settings", open=False):
                safetensors_input = gr.Textbox(label="Safetensors Path", value="model.safetensors")
                hf_model_input = gr.Textbox(label="Base HF Model", value="answerdotai/ModernBERT-base")
                device_input = gr.Dropdown(label="Device", choices=["cuda", "cpu", "mps"], value="cpu")
                strategy_input = gr.Dropdown(label="Strategy", choices=["predictor_corrector", "backward", "random"], value="backward")
                seq_len_input = gr.Slider(label="Sequence Length", minimum=16, maximum=512, value=64, step=16)
                num_steps_input = gr.Slider(label="Num Steps", minimum=10, maximum=500, value=32, step=1)
                
                with gr.Row():
                    use_manual_seed = gr.Checkbox(label="Use Custom Seed", value=False)
                    seed_input = gr.Number(label="Seed", value=8734578, precision=0, visible=False)
                
                # Toggle visibility of seed input
                use_manual_seed.change(fn=lambda x: gr.Number(visible=x), inputs=use_manual_seed, outputs=seed_input)
            
            run_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Generated Text", interactive=False, lines=10)

    run_btn.click(
        fn=run_app,
        inputs=[safetensors_input, hf_model_input, device_input, prompt_input, seq_len_input, num_steps_input, strategy_input, use_manual_seed, seed_input],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.queue().launch(css=css)
