import onnxruntime as ort
import sys
from typing import List
import argparse
from pathlib import Path
import cv2
import os
import tqdm
import numpy as np

model_size = (
    960,
    1280,
)


def _merge_images(orignal, add):
    org_shape = orignal.shape
    add_shape = add.shape

    if add_shape[0] >= org_shape[0] and add_shape[1] >= org_shape[1]:
        orignal = add[: org_shape[0], : org_shape[1]]
    elif add_shape[0] <= org_shape[0] and add_shape[1] <= org_shape[1]:
        orignal[: add_shape[0], : add_shape[1]] = add
    else:
        assert (
            False
        ), f"Inconsistent shapes, cannot crop or fill {add_shape} vs {model_size}"
    return orignal


def main(args):
    model = ort.InferenceSession(
        str(args.model),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    for img_name in tqdm.tqdm(os.listdir(args.input_folder)):
        img_path = os.path.join(args.input_folder, img_name)
        if img_path.endswith(".png"):
            img_u8 = cv2.imread(img_path)
            if img_u8.shape[-1] == 3:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

            norm_value = (
                np.iinfo(np.uint16).max
                if np.max(img_u8) > np.iinfo(np.uint8).max
                else np.iinfo(np.uint8).max
            )
            img_fp = img_u8.astype(dtype=np.float32) / norm_value
            input = np.zeros((1, 960, 1280, 1), dtype=np.float32)

            input[0, :, :, 0] = _merge_images(input[0, :, :, 0], img_fp)
            output = model.run(None, {"image_input": input})[0][0]
            mask_model_size = np.array(255 * (output[:, :, 0] >= 0.5), dtype=np.uint8)

            mask = np.zeros(img_fp.shape)
            mask = _merge_images(mask, mask_model_size)
            cv2.imwrite(os.path.join(args.output_folder, f"mask_{img_name}"), mask)

            img_3ch = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            overlay = _merge_images(img_3ch[:, :, 2], mask)
            img_3ch[:, :, 2] = np.where(overlay, 255, img_3ch[:, :, 2])
            cv2.imwrite(
                os.path.join(args.output_folder, f"overlay_{img_name}"), img_3ch
            )


def parse_args(args: List[str]):

    parser = argparse.ArgumentParser(
        "Run model inference", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the dataset folder",
    )

    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Path to read the pngs from",
    )

    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Path to dump the data",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
