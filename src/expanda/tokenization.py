import os
import json
import tqdm
import argparse
import shutil
from typing import List
from configparser import ConfigParser
from .utils import random_filename
from tokenizers import Tokenizer, models, decoders
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.pre_tokenizers import ByteLevel


def _split_subset_from_file(input_file: str, subset_file: str, subset_size: int):
    with open(input_file, "rb") as src, open(subset_file, "wb") as dst:
        while True:
            line = src.readline()

            # If all sentences are read, stop copying data.
            if not line:
                break

            dst.write(line)

            # If total amount of copied data is more than `subset_size`, stop
            # copying data.
            if src.tell() > subset_size:
                break


def jsonl_text_iterator(file_path, batch_size=10000):
    """Yields batches of extracted text from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.
        batch_size (int): Number of lines to process per batch.

    Yields:
        list: A batch of extracted text strings.
    """
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)  # Parse JSON
                if "text" in data:  # Extract text if available
                    batch.append(data["text"])

                if len(batch) >= batch_size:
                    yield batch  # Yield a full batch
                    batch = []  # Reset batch
            except json.JSONDecodeError:
                continue  # Skip malformed lines

    if batch:
        yield batch  # Yield remaining items if any


def train_tokenizer(
    input_file: str,
    vocab_file: str,
    merges_file: str,
    tokenizer_file: str,
    temporary: str,
    subset_size: int = 512000000,
    vocab_size: int = 8000,
    limit_alphabet: int = 6000,
    unk_token: str = "<unk>",
    control_tokens: List[str] = [],
    added_tokens: list[str] = [],
):
    r"""Train **BPE** tokenizer and save trained vocabulary.

    Note:
        Since tokenizers_ reads whole file data in training, this function
        could occur memory errors if `input_file` is too large. Under the
        assumption that `input_file` is shuffled randomly, the subset of input
        corpus will be used in training.

    Caution:
        The subset of input corpus is saved in `temporary` directory. Please be
        careful not to delete the file while executing this function.

    Arguments:
        input_file (str): Input file path.
        vocab_file (str): Output vocabulary file path.
        merges_file (str): Output merges file path.
        temporary (str): Temporary directory where the subset of corpus would
            be saved.
        subset_size (int): The maximum number of lines in the subset.
        vocab_size (int): The number of subwords in the vocabulary.
        limit_alphabet (int): The maximum number of alphabets in vocabulary.
        unk_tokens (str): Unknown token in the vocabulary.
        control_tokens (list): Control tokens in the vocabulary.
        added_tokens (list): Optional list of tokens to keep in vocabulary.

    .. _tokenizers: https://github.com/huggingface/tokenizers
    """
    # Create **BPE** model and add normalizer and pre-tokenizer.
    # BERT-specific normalizer and pre-tokenizer are used.
    model = models.BPE()
    tokenizer = Tokenizer(model)

    # Set normalizers (Unicode normalization)
    tokenizer.normalizer = NFKC()

    # Set pre-tokenizer to byte-level (like GPT)
    tokenizer.pre_tokenizer = ByteLevel()

    tokenizer.decoder = decoders.ByteLevel()

    num_sp_tks = tokenizer.add_special_tokens([unk_token] + control_tokens)
    print(f"[*] added {num_sp_tks} special tokens to the vocabulary.")

    if added_tokens:
        num_add_tks = tokenizer.add_tokens(added_tokens)
        print(f"[*] added {num_add_tks} tokens to the vocabulary.")

    # Split the head of input corpus file and save in `temporary` directory.
    subset_file = random_filename(temporary)
    _split_subset_from_file(input_file, subset_file, subset_size)

    # Train the model with splitted subset of corpus.
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        limit_alphabet=limit_alphabet,
        special_tokens=[unk_token] + control_tokens,
    )

    tokenizer.train_from_iterator(
        jsonl_text_iterator(subset_file),
        trainer=trainer,
    )

    # Save trained subword vocabulary in `temporary` directory and rename to
    # `vocab_file`.
    saved_files = tokenizer.model.save(os.path.dirname(vocab_file))
    saved_vocab_file = [f for f in saved_files if "vocab" in f][0]
    saved_merges_file = [f for f in saved_files if "merges" in f][0]
    os.rename(saved_vocab_file, vocab_file)
    os.rename(saved_merges_file, merges_file)
    tokenizer.save(tokenizer_file)

    # Remove temporary subset corpus.
    os.remove(subset_file)


def tokenize_corpus(
    input_file: str,
    output_file: str,
    tokenizer_file: str,
    unk_token: str = "<unk>",
    control_tokens: List[str] = [],
):
    r"""Tokenize corpus sentences through trained **WordPiece** model.

    Arguments:
        input_file (str): Input corpus file path.
        output_file (str): Output file path.
        vocab_file (str): Trained vocabulary file path.
        merges_file (str): Trained merges file path.
        unk_token (str): Unknown token in the vocabulary.
        control_tokens (list): Control tokens in the vocabulary.
    """
    # Create `BPE` model and add special tokens. Note that `unk_token`
    # is also a special token.normalizer and pre-tokenizer.
    tokenizer = Tokenizer.from_file(tokenizer_file)

    with open(input_file, "r", encoding="utf-8") as src, open(
        output_file, "w", encoding="utf-8"
    ) as dst:
        # Count total lines in corpus.
        total_lines = 0
        for _ in src:
            total_lines += 1

        # Move the corpus file to first.
        src.seek(0)

        buffer = []
        for line in tqdm.tqdm(src, desc="[*] tokenize corpus", total=total_lines):
            text: str = json.loads(line)["text"]
            buffer.append(text)

            # Tokenize buffered sentences and write to `output_file`.
            if len(buffer) > 10000:
                for t in tokenizer.encode_batch(buffer):
                    json.dump({"tokens": " ".join(t.tokens)}, dst, ensure_ascii=False)
                    dst.write("\n")
                buffer.clear()

        # Process the remained buffer.
        if buffer:
            for t in tokenizer.encode_batch(buffer):
                json.dump({"tokens": " ".join(t.tokens)}, dst, ensure_ascii=False)
                dst.write("\n")


def _main():
    parser = argparse.ArgumentParser(
        prog="expanda-tokenization", description="manage tokenizations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # command line: expanda.tokenization train ...
    train_parser = subparsers.add_parser(
        "train", help="train tokenizer by using corpus"
    )
    train_parser.add_argument("workspace", help="workspace directory")
    train_parser.add_argument(
        "config", default="expanda.cfg", nargs="?", help="expanda configuration file"
    )

    # command line: expanda.tokenization tokenize ...
    tokenize_parser = subparsers.add_parser(
        "tokenize", help="tokenize and split corpus"
    )
    tokenize_parser.add_argument("workspace", help="workspace directory")
    tokenize_parser.add_argument(
        "config", default="expanda.cfg", nargs="?", help="expanda configuration file"
    )

    args = parser.parse_args()
    config = ConfigParser()
    config.read(os.path.join(args.workspace, args.config))

    temporary = os.path.join(
        args.workspace, config["build"].get("temporary-path", "tmp")
    )
    reuse_vocab = config["build"].get("input-vocab", None)
    reuse_merges = config["build"].get("input-merges", None)
    vocab = os.path.join(
        args.workspace, config["build"].get("output-vocab", "build/vocab.json")
    )
    merges = os.path.join(
        args.workspace, config["build"].get("output-merges", "build/merges.txt")
    )
    tokenizer_file = os.path.join(
        args.workspace, config["build"].get("output-tokenizer", "build/tokenizer.json")
    )
    split_ratio = config["build"].getfloat("split-ratio", 0.1)

    train_corpus = os.path.join(
        args.workspace,
        config["build"].get("output-train-corpus", "build/corpus.train.txt"),
    )
    test_corpus = os.path.join(
        args.workspace,
        config["build"].get("output-test-corpus", "build/corpus.test.txt"),
    )
    raw_corpus = os.path.join(
        args.workspace, config["build"].get("output-raw-corpus", "build/corpus.raw.txt")
    )

    subset_size = config["tokenization"].getint("subset-size", fallback=1000000000)
    vocab_size = config["tokenization"].getint("vocab-size", fallback=8000)
    limit_alphabet = config["tokenization"].getint("limit-alphabet", fallback=1000)
    unk_token = config["tokenization"].get("unk-token", "<unk>")

    control_tokens = config["tokenization"].get("control-tokens", "")
    control_tokens = [token.strip() for token in control_tokens.splitlines() if token]

    added_tokens = config["tokenization"].get("added-tokens", "")
    added_tokens = [token.strip() for token in added_tokens.splitlines() if token]

    # Create temporary directory if not exists.
    if not os.path.exists(temporary):
        os.makedirs(temporary)

    if args.command == "train":
        if reuse_vocab:
            print(f"[*] use the given vocabulary file [{reuse_vocab}].")
            shutil.copyfile(reuse_vocab, vocab)
        if reuse_merges:
            print(f"[*] use the given merges file [{reuse_merges}].")
            shutil.copyfile(reuse_merges, merges)


        print("[*] train tokenizer.")
        # Train the tokenizer.
        train_tokenizer(
            raw_corpus,
            vocab,
            merges,
            tokenizer_file,
            temporary,
            subset_size=subset_size,
            vocab_size=vocab_size,
            limit_alphabet=limit_alphabet,
            unk_token=unk_token,
            control_tokens=control_tokens,
            added_tokens=added_tokens,
        )
    elif args.command == "tokenize":
        print("[*] create tokenized corpus.")
        tokenize_filename = random_filename(temporary)
        # Tokenize the input corpus file.
        tokenize_corpus(
            raw_corpus,
            tokenize_filename,
            tokenizer_file,
            unk_token=unk_token,
            control_tokens=control_tokens,
        )

        print("[*] split the corpus into train and test dataset.")
        with open(tokenize_filename, "rb") as src:
            total_lines = 0
            for _ in src:
                total_lines += 1

            src.seek(0)

            # Write to test dataset
            with open(test_corpus, "wb") as dst:
                for i, line in enumerate(src):
                    dst.write(line)
                    if i >= total_lines * split_ratio:
                        break

            # Write to train dataset
            with open(train_corpus, "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.remove(tokenize_filename)

    # Remove temporary directory
    print("[*] remove temporary directory.")
    shutil.rmtree(temporary)

    print("[*] finish building corpus.")


if __name__ == "__main__":
    _main()
