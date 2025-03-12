import os
import json
import tqdm
import argparse
from typing import List
from .utils import random_filename
from tokenizers import Tokenizer, models, decoders
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.pre_tokenizers import ByteLevel


def _split_subset_from_file(input_file: str, subset_file: str,
                            subset_size: int):
    with open(input_file, 'rb') as src, \
            open(subset_file, 'wb') as dst:
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

def jsonl_text_iterator(file_path, batch_size=1000):
    """Yields batches of extracted text from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file.
        batch_size (int): Number of lines to process per batch.
    
    Yields:
        list: A batch of extracted text strings.
    """
    batch = []
    with open(file_path, "r", encoding='utf-8') as f:
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
        temporary: str,
        subset_size: int = 512000000,
        vocab_size: int = 8000,
        limit_alphabet: int = 6000,
        unk_token: str = '<unk>',
        control_tokens: List[str] = []):
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

    .. _tokenizers: https://github.com/huggingface/tokenizers
    """
    # Create **BPE** model and add normalizer and pre-tokenizer.
    # BERT-specific normalizer and pre-tokenizer are used.
    model = models.BPE()
    tokenizer = Tokenizer(model)

    # Set normalizers (Unicode normalization + lowercasing)
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])

    # Set pre-tokenizer to byte-level (like GPT)
    tokenizer.pre_tokenizer = ByteLevel()

    # Split the head of input corpus file and save in `temporary` directory.
    subset_file = random_filename(temporary)
    _split_subset_from_file(input_file, subset_file, subset_size)

    # Train the model with splitted subset of corpus.
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        limit_alphabet=limit_alphabet,
        special_tokens=[unk_token] + control_tokens)

    tokenizer.train_from_iterator(
        jsonl_text_iterator(subset_file),
        trainer=trainer,
    )

    # Save trained subword vocabulary in `temporary` directory and rename to
    # `vocab_file`.
    saved_files = model.save(os.path.dirname(vocab_file))
    saved_vocab_file = [f for f in saved_files if 'vocab' in f][0]
    saved_merges_file = [f for f in saved_files if 'merges' in f][0]
    os.rename(saved_vocab_file, vocab_file)
    os.rename(saved_merges_file, merges_file)

    # Remove temporary subset corpus.
    os.remove(subset_file)


def tokenize_corpus(
        input_file: str,
        output_file: str,
        vocab_file: str,
        merges_file: str,
        unk_token: str = '<unk>',
        control_tokens: List[str] = []):
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
    tokenizer = Tokenizer(models.BPE.from_file(vocab_file, merges_file, unk_token=unk_token))
    tokenizer.add_special_tokens([unk_token] + control_tokens)

    # Set normalizers (Unicode normalization + lowercasing)
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    # Set pre-tokenizer to byte-level (like GPT)
    tokenizer.pre_tokenizer = ByteLevel()

    tokenizer.decoder = decoders.BPEDecoder()

    with open(input_file, 'r', encoding='utf-8') as src, \
            open(output_file, 'w', encoding='utf-8') as dst:
        # Count total lines in corpus.
        total_lines = 0
        for _ in src:
            total_lines += 1

        # Move the corpus file to first.
        src.seek(0)

        buffer = []
        for line in tqdm.tqdm(src,
                              desc='[*] tokenize corpus',
                              total=total_lines):
            text: str = json.loads(line)['text']
            buffer.append(text)

            # Tokenize buffered sentences and write to `output_file`.
            if len(buffer) > 10000:
                for t in tokenizer.encode_batch(buffer):
                    json.dump({'tokens': ' '.join(t.tokens)}, dst, ensure_ascii=False)
                    dst.write('\n')
                buffer.clear()

        # Process the remained buffer.
        if buffer:
            for t in tokenizer.encode_batch(buffer):
                json.dump({'tokens': ' '.join(t.tokens)}, dst, ensure_ascii=False)
                dst.write('\n')


def _main():
    parser = argparse.ArgumentParser(
        prog='expanda-tokenization',
        description='manage tokenizations')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # command line: expanda.tokenization train ...
    train_parser = subparsers.add_parser(
        'train', help='train tokenizer by using corpus')
    train_parser.add_argument('input')
    train_parser.add_argument('vocab', default='vocab.txt', nargs='?',
                              help='output vocabulary file')
    train_parser.add_argument('merges', default='merges.txt', nargs='?',
                              help='output merges file')
    train_parser.add_argument('--tmp', default='tmp',
                              help='temporary directory path')
    train_parser.add_argument('--subset_size', default=512000000, type=int,
                              help='maximum number of lines in subset')
    train_parser.add_argument('--vocab_size', default=8000, type=int,
                              help='number of subwords in vocabulary')
    train_parser.add_argument('--unk_token', default='<unk>',
                              help='unknown token name')
    train_parser.add_argument('--control_tokens', default=[], nargs='*',
                              help='control token names except unknown token')

    # command line: expanda.tokenization tokenize ...
    tokenize_parser = subparsers.add_parser(
        'tokenize', help='')
    tokenize_parser.add_argument('input')
    tokenize_parser.add_argument('output')
    tokenize_parser.add_argument('vocab')
    tokenize_parser.add_argument('merges')
    tokenize_parser.add_argument(
        '--unk_token', default='<unk>', help='unknown token name')
    tokenize_parser.add_argument(
        '--control_tokens', default=[], nargs='*',
        help='control token names except unknown token')

    args = parser.parse_args()
    if args.command == 'train':
        # Create temporary directory if not exists.
        remove_after_training = False
        if not os.path.exists(args.tmp):
            os.makedirs(args.tmp)
            remove_after_training = True

        # Train the tokenizer.
        train_tokenizer(args.input, args.vocab, args.merges, args.tmp, args.subset_size,
                        args.vocab_size, args.unk_token, args.control_tokens)

        # Remove created temporary directory.
        if remove_after_training:
            os.removedirs(args.tmp)
    elif args.command == 'tokenize':
        # Tokenize the input corpus file.
        tokenize_corpus(
            args.input, args.output, args.vocab, args.merges,
            args.unk_token, args.control_tokens)


if __name__ == '__main__':
    _main()
