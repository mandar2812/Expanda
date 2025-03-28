import os
import json
from typing import Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

login(os.getenv("HUGGINGFACE_API_KEY"))


def _process_hf_chem700K(
    input_file: str, output_file: str, temporary: str, args: Dict[str, Any]
):
    # Load the dataset from Hugging Face Hub with cache directory
    dataset = load_dataset(input_file, split=args["split"], cache_dir=args["cache_dir"])

    # Create temporary directory if not exists
    os.makedirs(temporary, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        for conv in tqdm(
            dataset, total=len(dataset), desc="Processing Chem700K"
        ):
            text: str = ""
            history: list[list[str]] = conv["history"]
            instruction: str = conv["instruction"]
            input_text: str = conv["input"]
            output: str = conv["output"]

            # First concatenate the history
            for hist in history:
                for idx, msg in enumerate(hist):
                    role = "<user>" if idx % 2 == 0 else "<assistant>"
                    text += f"<chmsg>{role}<s>{msg.strip()}</s></chmsg>"
            # Then concatenate the instruction
            text += f"<chmsg><user><s>{instruction.strip()} {input_text.strip()}</s></chmsg>"
            # Concatenate the output
            text += f"<chmsg><assistant><s>{output.strip()}</s></chmsg>"

            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")


__extension__ = {
    "name": "Chem700K dataset processor",
    "version": "1.0",
    "description": "Process the AI4Chem/ChemData700K dataset.",
    "author": "mandar2812",
    "main": _process_hf_chem700K,
    "arguments": {
        "split": {"type": str, "default": "train"},
        "cache_dir": {"type": str, "default": "./cache"},
    },
}
