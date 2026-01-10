import os
import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, get_scheduler
from datasets import load_from_disk
from accelerate import Accelerator
from tqdm.auto import tqdm
from tokenizer import get_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a language model.")
    
    # Experiment parameters
    parser.add_argument("--experiment_name",
                        type=str,
                        required=True,
    )
    
    parser.add_argument("--working_dir",
                        type=str,
                        required=True,
    )

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility.",
    )

    # Model parameters
    parser.add_argument("--hf_model_name",
                        type=str,
                        required=True,
    )

    # Dataset parameters
    parser.add_argument("--path_to_prepped_data",
                        type=str,
                        required=True,
                        help="Path to the preprocessed dataset stored on disk\
                            in prepare_pretrain_data.py.",
    )

    parser.add_argument("--num_workers",
                        type=int,
                        default=24,
                        help="Number of workers for data loading.",
    )

    # Training parameters
    parser.add_argument("--mixed_precision",
                        type=str,
                        default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Whether to use mixed precision. Choose between fp16 and bf16.",
    )

    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size per GPU/TPU core/CPU for training.",
    )

    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before\
                            performing a backward/update pass.",
    )

    parser.add_argument("--num_training_steps",
                        type=int,
                        default=100000,
                        help="Total number of training steps to perform.",
    )    

    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=1.0,
                        help="Maximum gradient norm for gradient clipping.",
    )

    parser.add_argument("--lr_scheduler_type",
                        type=str,
                        default="cosine",
                        help="Type of learning rate scheduler to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument("--num_warmup_steps",
                        type=int,
                        default=1000,
                        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument("--evaluation_interval",
                        type=int,
                        default=2500,
                        help="Number of steps between evaluations.",
    )

    parser.add_argument("--checkpoint_interval",
                        type=int,
                        default=2500,
                        help="Number of steps between model checkpoints.",
    )

    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5,
                        help="Max learning rate.",
    )

    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01,
                        help="Weight decay to use.",
    )

    # Logging parameters

    parser.add_argument("--log_wandb",
                        default=False,
                        help="Whether to log metrics and model checkpoints to Weights & Biases.",
                        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    return args

"""
experiment_name: str
working_dir: str
seed: int
hf_model_name: str
path_to_prepped_data: str
num_workers: int
mixed_precision: str
batch_size: int
gradient_accumulation_steps: int
num_training_steps: int
max_grad_norm: float
lr_scheduler_type: str
num_warmup_steps: int
evaluation_interval: int
learning_rate: float
weight_decay: float
log_wandb: bool
"""

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
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

args = parse_args()

# Initialize accelerator
path_to_experiment = os.path.join(args.working_dir, args.experiment_name)
os.makedirs(path_to_experiment, exist_ok=True)
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    log_with="wandb" if args.log_wandb else None,
    project_dir=path_to_experiment,
)

if args.log_wandb:
    accelerator.init_trackers(args.experiment_name, config=vars(args))

# Init tokenizer
tokenizer = get_tokenizer(args.hf_model_name)

# Init model
model = AutoModelForMaskedLM.from_pretrained(args.hf_model_name)
model.resize_token_embeddings(len(tokenizer))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print(f"Number of trainable parameters: {params}")

# Load dataset
batch_size = args.batch_size

def collate_fn(batch):
    tokens = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
    return {"input_ids": tokens}


tokenized_data = load_from_disk(args.path_to_prepped_data)
train_dataloader = DataLoader(tokenized_data["train"],
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

eval_dataloader = DataLoader(tokenized_data["test"],
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)

# Scheduler
scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
)

# Loss Function
loss_func = nn.CrossEntropyLoss(reduction="none")

# Prepare everything with accelerator
model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
)

# Training loop
train = True
global_step = 0
progress_bar = tqdm(range(args.num_training_steps), disable=not accelerator.is_local_main_process)

while train:
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"]

        # Attend to every token (EOS also)
        batch_size, seq_len = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)

        # Random sample t to mask tokens with probability t
        t = torch.rand(batch_size, 1, device=accelerator.device)
        t = t.expand(batch_size, seq_len).clamp_min(1e-5)
        mask = torch.bernoulli(t).bool()

        # Mask Data and Don't Compute Loss for Unmasked Data
        masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)

        # Compute logits
        with accelerator.accumulate(model):
            logits = model(input_ids=masked_input_ids,
                        attention_mask=attention_mask)["logits"]
            
            # Compute loss
            num_classes = logits.size(-1)
            loss = loss_func(logits.view(batch_size * seq_len, num_classes),
                            labels.flatten())

            # Scale loss by t
            loss = loss.reshape(batch_size, seq_len) / t
            loss = loss.mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

        # Logging
        if accelerator.is_local_main_process:
            progress_bar.update(1)
            global_step += 1
            if global_step % 100 == 0:
                accelerator.log({"train_loss": loss.item(),
                                "lr": scheduler.get_last_lr()[0],
                                }, step=global_step)
        
        # Evaluation
        if global_step % args.evaluation_interval == 0:
            model.eval()

            log = {"eval_loss": 0.0}
            eval_steps = 0

            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    input_ids = batch["input_ids"]

                    # Attend to every token (EOS also)
                    batch_size, seq_len = input_ids.size()
                    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)

                    # Random sample t to mask tokens with probability t
                    t = torch.rand(batch_size, 1, device=accelerator.device)
                    t = t.expand(batch_size, seq_len).clamp_min(1e-5)
                    mask = torch.bernoulli(t).bool()

                    # Mask Data and Don't Compute Loss for Unmasked Data
                    masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
                    labels = input_ids.masked_fill(~mask, -100)

                    # Compute logits
                    logits = model(input_ids=masked_input_ids,
                                attention_mask=attention_mask)["logits"]
                    
                    # Compute loss
                    num_classes = logits.size(-1)
                    loss = loss_func(logits.view(batch_size * seq_len, num_classes),
                                    labels.flatten())

                    # Scale loss by t
                    loss = loss.reshape(batch_size, seq_len) / t
                    loss = loss.mean()

                    log["eval_loss"] += loss.item()
                    eval_steps += 1
            
            log["eval_loss"] /= eval_steps
            accelerator.log(log, step=global_step)
            model.train()
        
        # Checkpointing
        if global_step % args.checkpoint_interval == 0:
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_dir = os.path.join(
                    path_to_experiment, f"checkpoint_latest"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                unwrapped_model.save_pretrained(checkpoint_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(checkpoint_dir)

        if global_step >= args.num_training_steps:
            train = False
            break

if accelerator.is_local_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    final_dir = os.path.join(path_to_experiment, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    unwrapped_model.save_pretrained(final_dir, save_function=accelerator.save)
    tokenizer.save_pretrained(final_dir)