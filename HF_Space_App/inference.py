import logging
import argparse
from transformers import AutoModelForMaskedLM
import torch
from rich.live import Live
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from tokenizer import get_tokenizer
from safetensors.torch import load_file

logging.getLogger("transformers").setLevel(logging.ERROR)

def load_model_and_tokenizer(path_to_weights, hf_model_name, device="cuda"):
    ### Load Tokenizer ###
    tokenizer = get_tokenizer(hf_model_name)
    
    ### Load Model and Update Embedding Size ###
    model = AutoModelForMaskedLM.from_pretrained(hf_model_name, device_map=device)
    model.resize_token_embeddings(len(tokenizer))

    # Load your checkpoint
    state_dict = torch.load(path_to_weights)
    model.load_state_dict(state_dict, strict=True)
    model.tie_weights()
    model.eval()

    return model, tokenizer

def prepare_unconditional_tokens_for_inference(seq_len, mask_token_id, device="cuda"):
    input_tokens = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device) 
    return input_tokens, mask, attention_mask

def prepare_conditional_tokens_for_inference(seq_len, tokenizer, prompt, device="cuda"):

    chat_template = [
        {"role": "user", "content": prompt}
    ]

    tokenized = tokenizer.apply_chat_template(
        chat_template,
        tokenize=True,
        add_special_tokens=True,
        add_generation_prompt=True
    )

    prompt_tokens = torch.tensor(tokenized).to(device)

    input_tokens, mask, attention_mask = prepare_unconditional_tokens_for_inference(
        seq_len, tokenizer.mask_token_id, device
    )

    input_tokens[0, :len(prompt_tokens)] = prompt_tokens

    mask[0, :len(prompt_tokens)] = False

    return input_tokens, mask, attention_mask

def format_display_for_qa(user_text, assistant_text):
    output = Text()
    output.append("USER: ", style="bold green")
    output.append(user_text + "\n\n")
    output.append("ASSISTANT: ", style="bold cyan")
    output.append(assistant_text, style="white")
    return output

def format_display_for_unconditional(gen_text):
    output = Text()
    output.append("Unconditional Generation: \n\n", style="bold green")
    output.append(gen_text, style="white")
    return output

def clean_text(raw_text: str) -> str:
    return (
        raw_text.replace("user", "")
        .replace("assistant", "")
        .strip()
    )

@torch.inference_mode()
def inference(tokenizer,
              model,
              num_steps, 
              strategy="random", 
              device="cuda",
              prompt=None,
              show_mask=True):

    if prompt is None:
        input_tokens, mask, attention_mask = prepare_unconditional_tokens_for_inference(args.seq_len, 
                                                                                        mask_token_id=tokenizer.mask_token_id,
                                                                                        device=args.device)
    else:
        input_tokens, mask, attention_mask = prepare_conditional_tokens_for_inference(args.seq_len, 
                                                                                      tokenizer=tokenizer,
                                                                                      prompt=args.prompt,
                                                                                      device=args.device)
    original_mask = mask.clone()

    ### Nice Printing Stuff ##
    console = Console(highlight=False)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        
        ### What Controls our Progress Bar ###
        task = progress.add_task("Generating...", total=num_steps)

        ### Get Timesteps for Inference ###
        times = torch.linspace(1, 0, num_steps + 1, device=device)

        with Live("", refresh_per_second=5, console=console) as live:
            for t, s in zip(times[:-1], times[1:]):

                if strategy == "backward":
                    logits = model(input_tokens, attention_mask=attention_mask).logits

                    probs = torch.softmax(logits[mask], dim=-1)
                    input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    remask_probs = torch.rand_like(mask, dtype=torch.float, device=device)
                    remask_probs = (remask_probs < s/t)
                    mask = mask & remask_probs
                    input_tokens[mask] = tokenizer.mask_token_id

                if strategy == "predictor_corrector":
                    logits = model(input_tokens, attention_mask=attention_mask).logits
                    
                    probs = torch.softmax(logits[mask], dim=-1)
                    input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    remask_probs = torch.rand_like(mask, dtype=torch.float, device=device)
                    remask_decision = (remask_probs < s/t)
                    
                    mask = mask & remask_decision 
                    input_tokens[mask] = tokenizer.mask_token_id

                    n_corrector_steps = 1
                    corrector_step_size = 0.5 * (t-s)/(1-s)

                    if n_corrector_steps > 0 and s > 0.3:
                        for _ in range(n_corrector_steps):
                            known_mask = ~mask ^ ~original_mask
                            noise_rng = torch.rand_like(known_mask, dtype=torch.float, device=device)
                            
                            to_remask = known_mask & (noise_rng < corrector_step_size)
                            
                            input_tokens[to_remask] = tokenizer.mask_token_id
                            
                            corr_logits = model(input_tokens, attention_mask=attention_mask).logits
                            
                            corr_probs = torch.softmax(corr_logits[to_remask], dim=-1)
                            corr_samples = torch.multinomial(corr_probs, num_samples=1).squeeze(-1)
                            
                            input_tokens[to_remask] = corr_samples
                
                if show_mask:
                    ### Get all of the Tokens ###
                    decoded_tokens = tokenizer.convert_ids_to_tokens(input_tokens[0])

                    ### Keep [MASK] tokens, drop all other special tokens ###
                    cleaned_tokens = []
                    for tok in decoded_tokens:
                        if tok == tokenizer.mask_token:  # keep mask tokens
                            cleaned_tokens.append(tok)
                        elif tok in tokenizer.all_special_tokens:  # drop all other specials
                            continue
                        else:
                            cleaned_tokens.append(tok)

                    ### Put all the tokens back together into a string ###
                    decoded_after = tokenizer.convert_tokens_to_string(cleaned_tokens)
                
                else:
                    decoded_after = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]

                if prompt is None:
                    format_text = format_display_for_unconditional(decoded_after)
                else:
                    ### Remove Prompt Text from Assistant Text ###
                    assistant_text = decoded_after.replace(prompt, "").strip()
                    ### Remove Keywords user and assistant ###
                    assistant_text = clean_text(assistant_text)
                    format_text = format_display_for_qa(prompt, assistant_text)
                live.update(format_text)
                progress.update(task, advance=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference LDM")
    parser.add_argument("--safetensors_path", required=True, type=str)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=512)
    parser.add_argument("--strategy", type=str, default="predictor_corrector", choices=["backward", "predictor_corrector"])
    parser.add_argument("--hf_model_name", type=str, default="distilbert/distilroberta-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    
    args = parser.parse_args()

    # class args:
    #     safetensors_path = "/kaggle/working/runs/Philosopher_v0/final_model/model.safetensors"
    #     prompt = "Generate a quote on life"
    #     seq_len = 64
    #     num_steps = 128
    #     strategy = "predictor_corrector"
    #     # strategy = "backward"
    #     hf_model_name = "answerdotai/ModernBERT-base"
    #     device = "cpu"
    #     seed = 1234
    
    # args = args()
    
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
        
    seed_everything(args.seed)

    ### Load Model ###
    model, tokenizer = load_model_and_tokenizer(args.safetensors_path, 
                                                args.hf_model_name, 
                                                args.device)

    inference(tokenizer,
              model,
              args.num_steps, 
              strategy=args.strategy, 
              device=args.device,
              prompt=args.prompt)