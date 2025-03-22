import re
import os
import math
import shutil
import argparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from configparser import ConfigParser
from .extension import Extension
from .shuffling import shuffle
from .tokenization import train_tokenizer, tokenize_corpus
from typing import List
from .utils import random_filename, random_filenames

load_dotenv(verbose=True)


def _show_extension_details(module_name: str):
    # Show the details of extension.
    ext = Extension(module_name)
    print(
        f"Extension [{ext.module_name}]\n"
        f"Name             : {ext.ext_name}\n"
        f"Version          : {ext.version}\n"
        f"Description      : {ext.description}\n"
        f"Author           : {ext.author}"
    )

    # Print required parameters.
    print("Parameters")
    for name, opt in ext.arg_reqs.items():
        print("    {:12s} : {:6s}".format(name, opt["type"].__name__), end="")
        if "default" in opt:
            print(f' (default: {opt["default"]})')
        else:
            print("")


def _show_required_extension_list(config_file: str):
    # Read config file.
    config = ConfigParser()
    config.read(config_file)

    # Parse `input-files` option to get required extension list.
    input_files = config["build"].get("input-files", fallback="")
    exts = set(
        re.match(r"--(.*?)\s", line).group(0)
        for line in input_files.splitlines(False)
        if line
    )

    # Show the extension list tidily.
    print("{:25s}{:10s}".format("Extension", "Version"))
    print("=" * 35)
    for ext in exts:
        version = Extension(ext).version
        print(f"{ext[:25]:25s}{version[:10]:10s}")


def _balancing_corpora(input_files: List[str], corpus_names: List[str], temporary: str):
    corpus_size = [os.path.getsize(input_file) for input_file in input_files]

    # Get maximum size and calculate repetition rate.
    max_size = max(corpus_size)
    expand_rate = [math.floor(max_size / size) for size in corpus_size]

    print("[*] balance the size for extracted texts.")
    for input_file, corpus_name, rate in zip(input_files, corpus_names, expand_rate):
        # Skip if no repetition case.
        if rate == 1:
            continue

        # Rename the origin file.
        print(f"[*] corpus [{corpus_name}] will be repeated {rate} times.")

        # Repeat the texts and save to the path of origin file.
        repeat_filename = random_filename(temporary)
        with open(input_file, "rb") as src, open(repeat_filename, "wb") as dst:
            for _ in range(rate):
                src.seek(0)
                shutil.copyfileobj(src, dst)

        # Remove the origin file.
        os.remove(input_file)
        os.rename(repeat_filename, input_file)


def _build_corpus(workspace: str, config_file: str):
    # Read config file
    config = ConfigParser()
    config.read(os.path.join(workspace, config_file))

    # Read arguments from configuration file
    input_files = config["build"].get("input-files")
    input_files = [
        re.match(r"--(.*?)\s+(.*)", line.strip()).groups()
        for line in input_files.splitlines(False)
        if line
    ]
    reuse_vocab = config["build"].get("input-vocab", None)
    reuse_merges = config["build"].get("input-merges", None)
    reuse_tokenizer = config["build"].get("input-tokenizer", None)

    temporary = os.path.join(workspace, config["build"].get("temporary-path", "tmp"))
    vocab = os.path.join(
        workspace, config["build"].get("output-vocab", "build/vocab.json")
    )
    merges = os.path.join(
        workspace, config["build"].get("output-merges", "build/merges.txt")
    )
    tokenizer_file = os.path.join(
        args.workspace, config["build"].get("output-tokenizer", "build/tokenizer.json")
    )
    split_ratio = config["build"].getfloat("split-ratio", 0.1)

    train_corpus = os.path.join(
        workspace, config["build"].get("output-train-corpus", "build/corpus.train.txt")
    )
    test_corpus = os.path.join(
        workspace, config["build"].get("output-test-corpus", "build/corpus.test.txt")
    )
    raw_corpus = os.path.join(
        workspace, config["build"].get("output-raw-corpus", "build/corpus.raw.txt")
    )

    subset_size = config["tokenization"].getint("subset-size", fallback=1000000000)
    vocab_size = config["tokenization"].getint("vocab-size", fallback=8000)
    limit_alphabet = config["tokenization"].getint("limit-alphabet", fallback=1000)
    unk_token = config["tokenization"].get("unk-token", "<unk>")

    control_tokens = config["tokenization"].get("control-tokens", "")
    control_tokens = [token.strip() for token in control_tokens.splitlines() if token]

    # Create directories if not exists
    def create_dir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    create_dir(os.path.dirname(vocab))
    create_dir(os.path.dirname(train_corpus))
    create_dir(os.path.dirname(test_corpus))
    create_dir(os.path.dirname(raw_corpus))
    create_dir(temporary)

    # Extract raw corpus file to plain sentences
    extract_filenames = random_filenames(temporary, len(input_files))
    for (ext, input_file), name in zip(input_files, extract_filenames):
        print(f"[*] execute extension [{ext}] for [{input_file}]")
        Extension(ext).call(
            os.path.join(workspace, input_file),
            name,
            temporary,
            dict(config.items(ext)),
        )

    # Balance the size of each corpus
    if config["build"].get("balancing", "").lower() == "true":
        _balancing_corpora(
            extract_filenames, [name for _, name in input_files], temporary
        )

    # Gather the extracted plain text
    print("[*] merge extracted texts.")
    integrate_filename = random_filename(temporary)
    with open(integrate_filename, "wb") as dst:
        for name in extract_filenames:
            with open(name, "rb") as src:
                shutil.copyfileobj(src, dst)
            os.remove(name)

    if config["build"].get("shuffle", "true").lower() == "true":
        # Shuffle the text
        print("[*] start shuffling merged corpus...")
        shuffle(integrate_filename, raw_corpus, temporary)
        os.remove(integrate_filename)
    else:
        shutil.move(integrate_filename, raw_corpus)
    # Train subword tokenizer and tokenize the corpus
    print("[*] complete preparing corpus. start training tokenizer...")

    should_train_tokenizer = True
    if reuse_tokenizer:
        print(f"[*] use the given tokenizer file [{reuse_tokenizer}].")
        shutil.copyfile(reuse_tokenizer, tokenizer_file)
        should_train_tokenizer = False
    if reuse_vocab and reuse_merges:
        # If re-using pretrained vocabulary file, skip training tokenizer
        print(f"[*] use the given vocabulary file [{reuse_vocab}] and merges [{reuse_merges}].")
        shutil.copyfile(reuse_vocab, vocab)
        shutil.copyfile(reuse_merges, merges)
        should_train_tokenizer = False

    if should_train_tokenizer:
        print("[*] train tokenizer.")
        train_tokenizer(
            raw_corpus,
            vocab,
            merges,
            tokenizer_file,
            temporary,
            subset_size,
            vocab_size,
            limit_alphabet,
            unk_token,
            control_tokens,
        )

    print("[*] create tokenized corpus.")
    tokenize_filename = random_filename(temporary)
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


def _download_latest_wikipedia_dumps(workspace: str, limit: int):
    # Get the latest Wikipedia dump URL
    response = requests.get("https://dumps.wikimedia.org/enwiki/latest/", timeout=60)
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a", href=True)
    dump_files = [
        link["href"]
        for link in links
        if link["href"].endswith("-pages-articles.xml.bz2")
        and "multistream" not in link["href"]
        and "meta" not in link["href"]
    ]

    # Limit the number of files to download
    dump_files = dump_files[:limit]

    # Create src directory if not exists
    src_dir = os.path.join(workspace, "src")
    os.makedirs(src_dir, exist_ok=True)

    # Download the dump files with resume capability
    for dump_file in dump_files:
        dump_url = f"https://dumps.wikimedia.org/enwiki/latest/{dump_file}"
        dump_path = os.path.join(src_dir, dump_file)
        print(f"Downloading {dump_url} to {dump_path}")

        # Check if file already exists and get its size
        file_size = 0
        if os.path.exists(dump_path):
            file_size = os.path.getsize(dump_path)

        headers = {"Range": f"bytes={file_size}-"}
        with requests.get(
            dump_url, headers=headers, stream=True, timeout=(180, 180)
        ) as r:
            r.raise_for_status()
            with open(dump_path, "ab") as f:
                for chunk in r.iter_content(chunk_size=8192):  # 8 KB chunks
                    if chunk:
                        f.write(chunk)

    # Generate expanda.cfg file
    config = ConfigParser()
    config["expanda.ext.wikipedia"] = {"num-cores": f"{os.cpu_count() or 1}"}
    config["tokenization"] = {
        "unk-token": "<unk>",
        "control-tokens": "<s>\n</s>\n<pad>",
    }
    config["build"] = {
        "input-files": "\n".join(
            [f"--expanda.ext.wikipedia     src/{dump_file}" for dump_file in dump_files]
        )
    }

    with open(os.path.join(workspace, "expanda.cfg"), "w") as configfile:
        config.write(configfile)

    print(f"Generated expanda.cfg in {workspace}")


def _combine_workspaces(output_workspace: str, input_workspaces: List[str]):
    """Combine multiple workspaces into a new workspace."""
    # Create output workspace and src directory
    os.makedirs(output_workspace, exist_ok=True)
    src_dir = os.path.join(output_workspace, "src")
    os.makedirs(src_dir, exist_ok=True)

    # Initialize merged config
    merged_config = ConfigParser()

    # Process each workspace
    all_input_files = []
    for workspace in input_workspaces:
        config = ConfigParser()
        config_path = os.path.join(workspace, "expanda.cfg")
        config.read(config_path)

        # Merge non-build sections
        for section in config.sections():
            if section != "build":
                if not merged_config.has_section(section):
                    merged_config.add_section(section)
                for key, value in config.items(section):
                    if not merged_config.has_option(section, key):
                        merged_config.set(section, key, value)

        # Process input files
        input_files = config["build"].get("input-files", "").splitlines()
        input_files = [f for f in input_files if f.strip()]

        for input_file in input_files:
            match = re.match(r"--(.*?)\s+(.*)", input_file.strip())
            if match:
                ext, src_path = match.groups()
                abs_src_path = os.path.join(workspace, src_path)
                new_src_path = os.path.join(
                    src_dir,
                    f"{os.path.basename(workspace)}_{os.path.basename(src_path)}",
                )

                # Create symlink
                if os.path.exists(abs_src_path):
                    os.symlink(os.path.abspath(abs_src_path), new_src_path)
                    all_input_files.append(
                        f"--{ext}     {os.path.relpath(new_src_path, output_workspace)}"
                    )

    # Set build section in merged config
    if not merged_config.has_section("build"):
        merged_config.add_section("build")

    merged_config.set("build", "input-files", "\n\t" + "\n\t".join(all_input_files))

    # Copy common build outputs
    common_outputs = [
        "output-vocab",
        "output-merges",
        "output-train-corpus",
        "output-test-corpus",
        "output-raw-corpus",
    ]

    for workspace in input_workspaces:
        config = ConfigParser()
        config.read(os.path.join(workspace, "expanda.cfg"))
        if "build" in config:
            for key in common_outputs:
                if config["build"].get(key) and not merged_config["build"].get(key):
                    merged_config.set("build", key, config["build"][key])

    # Write merged config
    with open(os.path.join(output_workspace, "expanda.cfg"), "w") as f:
        merged_config.write(f)

    print(f"[*] Combined {len(input_workspaces)} workspaces into {output_workspace}")



parser = argparse.ArgumentParser(
    prog="expanda", description="Expanda - A universal integrated corpus generator"
)
subparsers = parser.add_subparsers(dest="command", required=True)

# command line: expanda list [config]
list_parser = subparsers.add_parser(
    "list", help="list required extensions in the workspace"
)
list_parser.add_argument(
    "config", default="expanda.cfg", nargs="?", help="expanda configuration file"
)

# command line: expanda show [extension]
show_parser = subparsers.add_parser("show", help="show extension information")
show_parser.add_argument("extension", help="module name of certain extension")

# command line: expanda build [workspace] [config]
build_parser = subparsers.add_parser(
    "build", help="build dataset through given corpora"
)
build_parser.add_argument("workspace", help="workspace directory")
build_parser.add_argument(
    "config", default="expanda.cfg", nargs="?", help="expanda configuration file"
)

# command line: expanda download [workspace] [--limit LIMIT]
download_parser = subparsers.add_parser(
    "download", help="download latest Wikipedia dumps and generate expanda.cfg"
)
download_parser.add_argument("workspace", help="workspace directory")
download_parser.add_argument(
    "--limit", type=int, default=5, help="limit the number of files to download"
)

# command line: expanda combine output_workspace workspace1 workspace2 ...
combine_parser = subparsers.add_parser(
    "combine", help="combine multiple workspaces into a new workspace"
)
combine_parser.add_argument("output_workspace", help="output workspace directory")
combine_parser.add_argument(
    "input_workspaces", nargs="+", help="input workspace directories to combine"
)

args = parser.parse_args()
if args.command == "list":
    _show_required_extension_list(args.config)
elif args.command == "show":
    _show_extension_details(args.extension)
elif args.command == "build":
    _build_corpus(args.workspace, args.config)
elif args.command == "download":
    _download_latest_wikipedia_dumps(args.workspace, args.limit)
elif args.command == "combine":
    _combine_workspaces(args.output_workspace, args.input_workspaces)
