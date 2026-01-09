%%writefile prepare_pretrain_data.py

from datasets import load_dataset, load_from_disk, concatenate_datasets
import time
import argparse

from tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description="Prepare pretraining data")

parser.add_argument(
    "--test_split_pct",
    default=0.005,
    help="Percentage of data to use for test split",
    type=float,
)

parser.add_argument(
    "--context_length",
    default=1024,
    help="Pass in argument to override the default in Config, but then make sure config\
        reflects this change when training",
    type=int,
)

parser.add_argument(
    "--path_to_data_store",
    required=True,
    help="Path to where you want to save the final tokenized dataset",
    type=str,
)

parser.add_argument(
    "--huggingface_cache_dir",
    default=None,
    help="Path to huggingface cache dir. If not set, will use default huggingface cache dir",
    type=str,
)

parser.add_argument(
    "--dataset_split_seed",
    default=42,
    help="Random seed to use for dataset splitting",
    type=int,
)

parser.add_argument(
    "--num_workers",
    default=16,
    help="Number of workers to use for tokenization",
    type=int,
)

parser.add_argument(
    "--hf_model_name",
    default="answerdotai/ModernBERT-base",
    help="Huggingface model name for tokenizer",
    type=str,
)

parser.add_argument(
    "--batch_size",
    default=1000,
    help="Batch size for tokenization",
    type=int,
)

"""
test_split_pct: float = 0.005
context_length: int = 1024
path_to_data_store: str
huggingface_cache_dir: str = None
dataset_split_seed: int = 42
num_workers: int = 16
hf_model_name: str = "answerdotai/ModernBERT-base"
batch_size: int = 1000
"""

"""
python prepare_pretrain_data.py \
    --test_split_pct 0.005 \
    --context_length 1024 \
    --path_to_data_store ./data/tokenized_gutenberg_dataset \
    --huggingface_cache_dir ./hf_cache \
    --dataset_split_seed 42 \
    --num_workers 16 \
    --hf_model_name answerdotai/ModernBERT-base \
    --batch_size 1000
"""

def prepare_data(args):
    context_length = args.context_length
    path_to_save = args.path_to_data_store
    cache_dir = args.huggingface_cache_dir

    # Load Tokenizer
    tokenizer = get_tokenizer(args.hf_model_name)

    # Load Dataset
    dataset = load_dataset("manu/project_gutenberg",
                           split="en",
                           cache_dir=cache_dir,
                           num_proc=args.num_workers)
    
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

    def compute_tokens(examples):
        
        tokenized = tokenizer(
            examples["text"],
            return_attention_mask=False,
            add_special_tokens=True,
            max_length=None,
            truncation=False,
        )

        # Chunk Text
        input_ids_list = []
        for ids in tokenized["input_ids"]:
            for i in range(0, len(ids), context_length):
                chunk = ids[i:i + context_length]
                if len(chunk) < context_length:
                    continue
                input_ids_list.append(chunk)
        
        return {"input_ids": input_ids_list}
    
    tokenized_data = dataset.map(
        compute_tokens,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        remove_columns="text",
    )

    # Save Daataset
    print(f"Saving tokenized dataset to {path_to_save}")
    tokenized_data.save_to_disk(path_to_save)

if __name__ == "__main__":
    args = parser.parse_args()
    start_time = time.time()
    prepare_data(args)
    end_time = time.time()
    print(f"Data preparation completed in {end_time - start_time} seconds.")

    print("Test loading the dataset from disk...")
    dataset = load_from_disk(args.path_to_data_store)
    print("Loaded dataset with length:", len(dataset))
    print("Dataset loaded successfully.")