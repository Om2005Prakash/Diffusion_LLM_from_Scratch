%%writefile prepare_sft_data.py

from datasets import load_dataset, load_from_disk
import time
import argparse

from tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description="Prepare SFT data")

parser.add_argument(
    "--test_split_pct",
    default=0.01,
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

def prepare_data(args):

    context_length = args.context_length
    path_to_save = args.path_to_data_store
    cache_dir = args.huggingface_cache_dir

    tokenizer = get_tokenizer(args.hf_model_name)

    dataset = load_dataset(
        "csv",
        data_files="/kaggle/input/quotes-500k/quotes.csv",
        cache_dir=cache_dir,
    )

    # #####################################################################
    dataset = dataset["train"]
    print("Sample Data:", dataset[0])
    # #####################################################################

    def apply_chat_template(query, response):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ],
            tokenize=True,
            add_special_tokens=True,
        )
    
    def preprocess(example):

        instruction = ", ".join(example["category"]) if isinstance(example["category"], list) else example["category"]
        instruction = f"Generate a quote on {instruction}."

        output = example["quote"] + f"\n— {example['author']}"

        tokenized = apply_chat_template(instruction, output)
        return {"input_ids": tokenized, "length": len(tokenized)}

    def remove_missing_examples(example):
        return example["quote"] is not None and example["author"] is not None and example["category"] is not None
    
    dataset = dataset.filter(remove_missing_examples, num_proc=args.num_workers)

    dataset = dataset.train_test_split(
        test_size=args.test_split_pct,
        seed=args.dataset_split_seed,
    )

    tokenizied_dataset = dataset.map(
        preprocess,
        num_proc=args.num_workers,
        remove_columns=["quote", "author", "category"],
    )

    # print("Sample tokenized data:")
    # print(tokenizer.decode(tokenizied_dataset["train"][0]["input_ids"], skip_special=False))

    def keep_within_context_length(example):
        return example["length"] <= context_length

    print("Number of Samples in Dataset:", len(tokenizied_dataset["train"]))
    tokenizied_dataset = tokenizied_dataset.filter(keep_within_context_length, num_proc=args.num_workers)
    tokenizied_dataset = tokenizied_dataset.remove_columns(["length"])
    print("Number of Samples after filtering by context length:", len(tokenizied_dataset["train"]))

    def get_answer_mask(example):
        tokenized = example["input_ids"]

        query_mask = []
        occurance = 0
        is_answer = False

        for t in tokenized:
            check = (t==tokenizer.convert_tokens_to_ids("<END_ID>"))
            if not is_answer:
                query_mask.append(0)
            else:
                query_mask.append(1)
            
            if check:
                if occurance == 0:
                    occurance += 1
                else:
                    is_answer = True

        example["answer_mask"] = query_mask

        return example
    
    tokenizied_dataset = tokenizied_dataset.map(
        get_answer_mask,
        num_proc=args.num_workers,
    )

    print("Sample tokenized data with answer mask:")
    print(tokenizer.decode(tokenizied_dataset["train"][0]["input_ids"], skip_special=False))
    print("Answer Mask:", tokenizied_dataset["train"][0]["answer_mask"])

    print(f"Saving tokenized dataset to {path_to_save}")
    tokenizied_dataset.save_to_disk(path_to_save)

if __name__ == "__main__":
    args = parser.parse_args()
    prepare_data(args)

    # Test loading the dataset
    print("Test loading the dataset from disk...")
    dataset = load_from_disk(args.path_to_data_store)
    print("Number of samples in loaded dataset:", len(dataset["train"]))
    print("Dataset loaded successfully.")
    print(dataset["train"])