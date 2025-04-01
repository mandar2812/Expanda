import shutil
from typing import Dict, Any


def _identity_mapper(
    input_file: str, output_file: str, temporary: str, args: Dict[str, Any]
):
    # Copy the input file as-is into the output file
    # No need for temporary directory in this case
    shutil.copy(input_file, output_file)


__extension__ = {
    "name": "identity map",
    "version": "1.0",
    "description": "Copy input file as-is into output",
    "author": "mandar2812",
    "main": _identity_mapper,
    "arguments": {},
}
