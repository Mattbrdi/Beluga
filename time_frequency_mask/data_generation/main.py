import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from time_frequency_mask.config import Parameters
from time_frequency_mask.data_generation.core.data_generator import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic beluga mask generator")
    parser.add_argument(
        "--config",
        help="Configuration used for time_frequency_mask",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--showcase",
        action="store_true",
        help="Showcase mode generates a few examples"
        "but does not store them in the provided output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parameters = Parameters.from_json(args.config)
    is_showcase = args.showcase
    generator = Generator(parameters, args.showcase)
    generator.run()
