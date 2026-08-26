import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def labelme_json_to_mask(json_path, *, label=None, stft_order=True):
    """Convert LabelMe polygon annotations to a binary mask.

    Args:
        json_path: Path to a LabelMe JSON file.
        label: Optional shape label to keep, for example "Whistle".
        stft_order: If True, flip vertically so row 0 is the lowest frequency bin.
            If False, keep normal image row order, where row 0 is the top pixel.

    Returns:
        A boolean mask shaped (frequency_bins, time_frames) when stft_order=True,
        or (image_height, image_width) when stft_order=False.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    height = int(data["imageHeight"])
    width = int(data["imageWidth"])
    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        if label is not None and shape.get("label") != label:
            continue

        points = np.asarray(shape["points"], dtype=np.float32)
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)

        cv2.fillPoly(mask, [points], color=1)

    if stft_order:
        mask = np.flipud(mask)

    return mask.astype(bool)


def save_mask(mask, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".npy":
        np.save(output_path, mask)
    else:
        cv2.imwrite(str(output_path), mask.astype(np.uint8) * 255)


def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe polygon JSON to a binary mask.")
    # parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--path", type=str, help="path towards polygons")
    parser.add_argument(
        "--image-order",
        action="store_true",
        help="Keep image row order instead of flipping to STFT frequency order.",
    )
    args = parser.parse_args()
    print("test")


    for json_path in Path(args.path).glob("*.json"):
 
        mask = labelme_json_to_mask(
            json_path,
            label=args.label,
            stft_order=False,
        )

        mask_path = json_path.parent.parent / "mask"
        print(str(mask_path))
        output = mask_path / f"{json_path.stem}_mask.png"
        save_mask(mask,  output)
        print(f"saved {output} with shape {mask.shape}")


if __name__ == "__main__":
    main()
