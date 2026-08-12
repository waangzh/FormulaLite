"""Build the fixed FormulaLite tokenizer artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formulalite.data.tokenizer import FormulaTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizer"),
        help="Hugging Face artifact output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = FormulaTokenizer.build().save_pretrained(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
