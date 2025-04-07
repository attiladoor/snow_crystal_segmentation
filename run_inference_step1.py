"""""""""
This script is created to evaluate the converted "*.png" images from the B-ICI instrument.
Creates a txt file with the output of:
name of the particle
AR - aspect ratio
particle size (in pixels)
AED - Area Equivalent Diameter
min_dia
max_dia
min_circle_diameter / maximum_dimension
x - length
y - width
border - Particle is on border(1) or not(0)?
"""""""""
import onnxruntime as ort
import sys
from typing import List
import argparse
from pathlib import Path
import cv2
import os
import tqdm
import numpy as np
from pathlib import Path
import math
import model.dataset as dataset
import shutil

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

def model_version(input):
    with open("mode_version.txt", "w") as f:
        f.write(f"Model version: {input}")

def main(args):
    model = ort.InferenceSession(
        str(args.model),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    # CREATE FOLDER IF ID DOESNT EXIST
    folder_names = ["mask"]

    if os.path.exists(args.output_folder): 
        shutil.rmtree(args.output_folder)
        print("\n Folder exists. Deleting...")
    os.mkdir(args.output_folder)
    print(f"\n Creating folder: {args.output_folder}")
    
    for folder_name in folder_names:
        folder_path = os.path.join(args.output_folder, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Creating folder: {folder_name}")

    ### CREATE TEXTFILE
    for image_files in tqdm.tqdm(os.listdir(args.input_folder)):
        img_path = (os.path.join(args.input_folder, image_files))
        img_name = Path(img_path).stem
        if img_path.endswith(".png"):
            img_u8c = cv2.imread(img_path)
            if img_u8c.shape[-1] == 3:
                img_u8 = cv2.cvtColor(img_u8c, cv2.COLOR_BGR2GRAY)

            norm_value = (
                np.iinfo(np.uint16).max
                if np.max(img_u8) > np.iinfo(np.uint8).max
                else np.iinfo(np.uint8).max
                )

            img_fp = img_u8.astype(dtype=np.float32) / norm_value
            input = np.zeros((1, 960, 1280, 1), dtype=np.float32)

            input[0, :, :, 0] = _merge_images(input[0, :, :, 0], img_fp)
            output = model.run(None, {"image_input": input})[0][0]
                
            mask_model_size = np.array(255 * (output[:, :, 0] >= 0.99), dtype=np.uint8)
                
            mask = np.zeros(img_fp.shape)
            mask = _merge_images(mask, mask_model_size)
            mask = cv2.bitwise_not(mask) # Invert B/W
            cv2.imwrite(
                    str(Path(args.output_folder) /
                        "mask" /
                        f"{image_files}"),
                        mask
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
    model_version(args)
