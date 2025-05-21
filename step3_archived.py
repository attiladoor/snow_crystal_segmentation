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

input_folder = input("Location of the original images(LTU22 folder /113 images): ")
contour_folder = input("Location of the contour images: ")
output_folder = input("desired output folder: ")

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
 
def calculate_max_dim(binary_image_path): # diameter of the smallest circle that encloses the particle     
     # Read the binary image
     binary_image = binary_image_path

     # Find contours
     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     # Fit a circle to the contours
     (x, y), radius = cv2.minEnclosingCircle(contours[0])
     center = (int(x), int(y))
     radius = int(radius)

     output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

     # Calculate the diameter times 1.65 um/px
     diameter = radius * 2 * 1.65
 
  # Define the top-left and bottom-right coordinates of the bounding box
     top_left = (max(0, int(x - radius * 1.1)), max(0, int(y - radius * 1.1)))
     bottom_right = (min(output_image.shape[1], int(x + radius * 1.1)), min(output_image.shape[0], int    (    y + radius * 1.1)))

     # Extract the circle area from the image
     circle_area = output_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

     # Return the diameter
     return diameter

    # CREATE Output FOLDER IF ID DOESNT EXIST
folder_names = ["particles", "mask", "bbox", "overlay", "mask_particle", "circle"]

if not os.path.exists(output_folder): 
    os.mkdir(output_folder)
    print(f"Creating folder: {args.output_folder}")
   
for folder_name in folder_names:
    folder_path = os.path.join(output_folder, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"Creating folder: {folder_name}")

    ### CREATE TEXTFILE
with open(str(output_folder) + "/particles.txt", "w") as file:
    file.write(
        "Particle_ID " + 
        "AR " + 
        "size " + 
        "aed " + 
        "min_dia " +
        "max_dia " +
        "max_dim " + 
        "x " +
        "y " +
        "border " +
        "\n"
    )

    for img_name in tqdm.tqdm(os.listdir(input_folder)):
        img_path = (os.path.join(input_folder, img_name))
        if img_path.endswith(".png"):
            img_u8c = cv2.imread(img_path)
            if img_u8c.shape[-1] == 3:
                img_u8 = cv2.cvtColor(img_u8c, cv2.COLOR_BGR2GRAY)

            norm_value = (
                np.iinfo(np.uint16).max
                if np.max(img_u8) > np.iinfo(np.uint8).max
                else np.iinfo(np.uint8).max
            )

            print(img_name)

            img_fp = img_u8.astype(dtype=np.float32) / norm_value

            contour_path = os.path.join(contour_folder, img_name)
            output = cv2.imread(contour_path)  # model.run(None, {"image_input": input})[0][0]

            mask_model_size = np.array(255 * (output[:, :, 0] >= 0.99), dtype=np.uint8)

            mask = np.zeros(img_fp.shape)
            mask = _merge_images(mask, mask_model_size)
            cv2.imwrite(
                str(Path(output_folder) /
                    "mask" /
                    f"mask_{img_name}"),
                mask
            )

            ### FIND CONTOURS/BOUNDING BOX
            edged = cv2.Canny(mask, 30, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # draw contours
            contour = cv2.drawContours(img_u8c, contours, -1, (0, 255, 0), 1)
            contour2 = contour

            # Iterate over the contours
            scale = 1.75
            counter = 1
            for contour_box in contours:
                # get the bounding box of the contours
                rect = cv2.minAreaRect(contour_box)
                center, size, angle = rect
                x, y = map(int, center)

                # Rotation matrix
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Rotate the image
                rotated = cv2.warpAffine(img_u8, M, (img_u8.shape[0], img_u8.shape[1]))
                rotated_mask = cv2.warpAffine(mask, M, (mask.shape[0], mask.shape[1]))

                w = int(size[0] * scale)
                h = int(size[1] * scale)

                if w > 0 and h > 0:
                    # Calculate the bounding box coordinates
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # Check if the bounding box crosses the border
                    if (x - w / 2) < 0 or (x + w / 2) > img_u8.shape[1] or (y - h / 2) < 0 or (y + h / 2) > img_u8.shape[0]:
                        border = 1
                        print(f"Bounding box {counter} crosses the border of the original image")
                    else:
                        border = 0

                    # positioning the scaled rectangle to encapsulate the particle
                    start_x = max(int(center[0] - w // 2), 0)
                    start_y = max(int(center[1] - w // 2), 0)
                    end_x = min(int(center[0] + w // 2), rotated.shape[1])
                    end_y = min(int(center[1] + w // 2), rotated.shape[0])

                    # Draw rectangle around the contour
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    contour = cv2.drawContours(contour, [box], 0, (0, 255, 255), 2)

                    # EXTRACT THE SCALED REGION INSIDE THE BOUNDING BOX
                    particle = rotated[start_y:end_y, start_x:end_x]
                    particle_mask = rotated_mask[start_y:end_y, start_x:end_x]

                    px_res = 1.65  # resolution of 1.65um/pixel

                    height, width = particle.shape[:2]  # Extracting height and width of the particle
                    if particle.shape[0] > 10 and particle.shape[1] > 10:  # LIMIT THE OBJECT SIZE
                        cv2.imwrite(
                            str(Path(output_folder) /
                                "particles" /
                                f"particle_{str(img_name)[:18]}_{counter}.png"),
                            particle
                        )
                        cv2.imwrite(
                            str(Path(output_folder) /
                                "mask_particle" /
                                f"particle_mask{str(img_name)[:18]}_{counter}.png"),
                            particle_mask
                        )
                        # calculate the aspect ratio of particles that is defined by the ratio of the 2 lengths of the bounding box that encapsulates the particle.
                        AR = max(h, w) / min(h, w)

                        # Calculate the Area ratio

                        # calculate the size of the particle
                        # AED - Area Equivalent Diameter converted to um
                        aed = 2 * math.sqrt(np.count_nonzero(particle_mask == 255) / math.pi) * px_res

                        # Find the indices of all black pixels in the image
                        obj_indices = np.where(particle_mask == 0)
                        # calculate the size of the black object in the x and y dimensions
                        dim_x = np.amax(obj_indices[1]) - np.amin(obj_indices[1]) + 1
                        dim_y = np.amax(obj_indices[0]) - np.amin(obj_indices[0]) + 1
                        min_dia = 2 * math.sqrt((dim_x * dim_y) / math.pi) * px_res
                        max_dia = math.sqrt((dim_x * dim_x) + (dim_y * dim_y)) * px_res

                        # maximum_dimension / ferret
                        max_dim = calculate_max_dim(particle)

                        # Area ratio
                        # ferret_area = (ferret/2) **2 * np.pi
                        # particle_area = cv2.contourArea(contour_box)
                        # area_ratio = ferret_area / particle_area

                        file.write(f"{str(img_name)[:18]}_{counter} " +
                                   f"{AR} " +
                                   f"{particle.size} " +
                                   f"{aed} " +
                                   f"{min_dia} " +
                                   f"{max_dia} " +
                                   f"{max_dim} " +
                                   f"{x} " +
                                   f"{y} " +
                                   f"{border} " +
                                   "\n")
                    else:
                        file.write(f"{str(img_name)[:18]} " + "particle smaller than 10um" + "\n")

                    counter = counter + 1

            cv2.imwrite(str(Path(output_folder) /
                            "bbox" /
                            f"box_{img_name}"),
                        contour
                        )

            ### FIND THE SMALLEST CIRCLE/FERRET DIAMETER
            for bounding_circle in contours:
                (circle_x, circle_y), radius = cv2.minEnclosingCircle(bounding_circle)
                center = (int(circle_x), int(circle_y))
                radius = int(radius)

                contour2 = cv2.circle(contour2, center, radius, (0, 0, 255), 2)

            cv2.imwrite(str(Path(output_folder) /
                            "circle" /
                            f"circle_{img_name}"),
                        contour2
                        )

            img_3ch = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            overlay = _merge_images(img_3ch[:, :, 2], mask)
            img_3ch[:, :, 2] = np.where(overlay, 255, img_3ch[:, :, 2])
            cv2.imwrite(
                str(Path(output_folder) /
                    "overlay" /
                    f"overlay_{img_name}"),
                img_3ch
            )
