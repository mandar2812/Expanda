import os
import json
from typing import Dict, Any
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login

login(os.getenv("HUGGINGFACE_API_KEY"))


def _process_hf_chem_std(
    input_file: str, output_file: str, temporary: str, args: Dict[str, Any]
):
    # Load the dataset from Hugging Face Hub with cache directory
    dataset = load_dataset(input_file, split=args["split"], cache_dir=args["cache_dir"])

    # Create temporary directory if not exists
    os.makedirs(temporary, exist_ok=True)

    df: pd.DataFrame = dataset.to_pandas().sort_values(
        ["conversation_id", "message_id"]
    )

    with open(output_file, "a", encoding="utf-8") as f:
        for _, conv in tqdm(
            df.groupby("conversation_id"), total=len(df["conversation_id"].unique())
        ):
            text: str = ""
            for msg in conv.itertuples():
                role = "<user>" if msg.message_type == "input" else "<assistant>"
                text += f"<chmsg>{role}<s>{msg.message}</s></chmsg>"
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")


__extension__ = {
    "name": "HydraLM standardized chemistry dataset processor",
    "version": "1.0",
    "description": "Process the HydraLM/chemistry_dataset_standardized dataset.",
    "author": "mandar2812",
    "main": _process_hf_chem_std,
    "arguments": {
        "split": {"type": str, "default": "train"},
        "cache_dir": {"type": str, "default": "./cache"},
    },
}
