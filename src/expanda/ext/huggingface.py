import os
import shutil
from typing import Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

login(os.getenv("HUGGINGFACE_API_KEY"))

def _process_huggingface_dataset(
    input_file: str, output_file: str, temporary: str, args: Dict[str, Any]
):
    # Load the dataset from Hugging Face Hub with cache directory
    dataset = load_dataset(
        input_file, split=args["split"], cache_dir=args["cache_dir"]
    )
    dsname = args["dataset_name"]

    # Create temporary directory if not exists
    os.makedirs(temporary, exist_ok=True)

    # Process the dataset and save to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, total=len(dataset), desc=f"Processing {dsname}"):
            text: str = example[args["text_column"]]
            f.write(text.replace("\n\n", " ") + "\n")


__extension__ = {
    "name": "huggingface dataset processor",
    "version": "1.0",
    "description": "process a dataset from Hugging Face Hub.",
    "author": "mandar2812+GitHubCoPilot",
    "main": _process_huggingface_dataset,
    "arguments": {
        "dataset_name": {"type": str, "default": "wikipedia"},
        "split": {"type": str, "default": "train"},
        "text_column": {"type": str, "default": "text"},
        "cache_dir": {"type": str, "default": "./cache"},
    },
}
