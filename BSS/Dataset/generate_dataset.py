from __future__ import annotations

import argparse
from pathlib import Path

from .builder import build_dataset
from .config import DatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genere un dataset synthetique pour la BSS.")
    parser.add_argument("--config", required=True, type=Path, help="Configuration JSON du dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Dossier de sortie.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplace le dossier de sortie s'il n'est pas vide.",
    )
    return parser.parse_args()


    
def main() -> None:
    args = parse_args()
    config = DatasetConfig.from_json(args.config)
    output = build_dataset(config, args.output, overwrite=args.overwrite)
    print(f"Dataset genere dans {output.resolve()}")


if __name__ == "__main__":
    main()
