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

    # CREATE FOLDER IF ID DOESNT EXIST
    folder_names = ["particles", "mask", "bbox", "overlay", "mask_particle"]

    if not os.path.exists(args.output_folder): 
        os.mkdir(args.output_folder)
        print(f"Creating folder: {args.output_folder}")
    
    for folder_name in folder_names:
        folder_path = os.path.join(args.output_folder, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Creating folder: {folder_name}")

    # CREATE TEXTFILE
    with open(str(args.output_folder) + "/particles.txt", "w") as file:
        file.write("image_id " + "aspect_ratio " + "size " + "aed " + "min_dia " + "max_dia "  + "\n")
    
        for img_name in tqdm.tqdm(os.listdir(args.input_folder)):
            img_path = os.path.join(args.input_folder, img_name)
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
                mask_model_size = np.array(255 * (output[:, :, 0] >= 0.5), dtype=np.uint8)
                
                mask = np.zeros(img_fp.shape)
                mask = _merge_images(mask, mask_model_size)
                cv2.imwrite(
                        str(Path(args.output_folder) /
                            "mask" /
                            f"mask_{img_name}"),
                        mask
                        )

                # FIND CONTOURS/BOUNDING BOX
                edged = cv2.Canny(mask, 30, 200)
                contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # draw contours
                contour = cv2.drawContours(img_u8c, contours, -1, (0,255,0), 1)
                # Iterate over the contours
                scale = 1.5
                counter = 1
                for contour_box in contours:
                    # get the bounding box of the contours
                    x, y, w, h = cv2.boundingRect(contour_box)
                    scaled_x = int(x - (scale -1) * w / 2)
                    scaled_y = int(y - (scale -1) * h / 2)
                    scaled_w = int(w * scale)
                    scaled_h = int(h * scale)
                    # Draw rectangle around the contour
                    box = cv2.rectangle(
                            contour, (scaled_x, scaled_y), (x + scaled_w, y + scaled_h), (255,0,0),1
                            )
                # EXTRACT THE CROPPED REGION INSIDE THE BOUNDING BOX
                    particle = img_u8[ scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]
                    particle_mask = mask[ y : y + h, x : x + w ]
                    px_res = 1.65 # resolution of 1.65um/pixel

                    if particle.size > (10/px_res)**2*math.pi : # Limit particles to 10um+
                        cv2.imwrite(
                                str(Path(args.output_folder) /
                                    "particles" /
                                    f"particle_{str(img_name)[:18]}_{counter}.png"),
                                particle
                                )
                        cv2.imwrite(
                                str(Path(args.output_folder) /
                                    "mask_particle" /
                                    f"particle_mask{str(img_name)[:18]}_{counter}.png"),
                                particle_mask
                                )
                        ## calculate the aspect ratio of particles that is defined by the ratio of the 2 length of the bounding box that encapsulates the particle.
                        aspect_ratio = w/h

                        ## calculate the size of the particle
                        # AED - Area Equivalent Diameter converted to um
                        aed = 2 * math.sqrt(np.count_nonzero(particle_mask == 0) / math.pi) * px_res

                        # Find the indices of all black pixels in the image
                        obj_indices = np.where(particle_mask == 0)
                        # calculate the size of the black object in the x and y dimensions
                        dim_x = np.amax(obj_indices[1]) - np.amin(obj_indices[1]) + 1
                        dim_y = np.amax(obj_indices[0]) - np.amin(obj_indices[0]) + 1
                        min_dia = 2 * math.sqrt((dim_x * dim_y) / math.pi) * px_res
                        max_dia = math.sqrt((dim_x * dim_x) + (dim_y * dim_y)) * px_res

                        file.write(f"{str(img_name)[:18]}_{counter} " +
                                f"{aspect_ratio} " +
                                f"{particle.size} " +
                                f"{aed} " +
                                f"{min_dia} " +
                                f"{max_dia} " +
                                "\n" )

                    else:
                        file.write(f"{str(img_name)[:18]}: "+ "no_particle" + "\n")
                    counter = counter + 1

#                cv2.imwrite(str(Path(args.output_folder) /
#                    "bbox" /
#                    f"box_{img_name}"),
#                    box
#                    )
                
                img_3ch = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
                overlay = _merge_images(img_3ch[:, :, 2], mask)
                img_3ch[:, :, 2] = np.where(overlay, 255, img_3ch[:, :, 2])
                cv2.imwrite(
                    str(Path(args.output_folder) /
                        "overlay" /
                        f"overlay_{img_name}"),
                        img_3ch
                        )
                
                # write the contour file
                cv2.imwrite(str(Path(args.output_folder) /
			"bbox" / 
			f"contour_{img_name}"),
			contour)

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
