import cv2
import numpy as np
import os
import shutil
from pathlib import Path


# Folder paths
#original_folder = "/home/stejan/snow_crystal_segmentation/ltu18-21"
#original_folder = Path("/home/stejan/snow_crystal_segmentation/data/cropped_/batch_1/cropped_original_png")
original_folder = Path("/home/stejan/snow_crystal_segmentation/ltu23")
#original_folder = Path("/home/stejan/snow_crystal_segmentation/ltu16_paper1")
#original_folder = Path("/home/stejan/hand_analyzed/ltu16_paper2/ltu16_bg")

#contour_folder = Path("/home/stejan/snow_crystal_segmentation/scs_out/ltu21_m331_05/mask")
#contour_folder = Path("/home/stejan/snow_crystal_segmentation/step2_output/ltu16/")
contour_folder = Path("/home/stejan/snow_crystal_segmentation/scs_out/ltu23_m332_05_paper3/mask")
#contour_folder = Path("/home/stejan/snow_crystal_segmentation/data/cropped_/batch_1/cropped_contours")
#contour_folder = Path("/home/stejan/hand_analyzed/ltu16_paper2/ltu16")

#output_folder = "/home/stejan/snow_crystal_segmentation/step1_output/ltu18-21"
#output_folder = Path("/home/stejan/snow_crystal_segmentation/step2_output/ltu21_nodust")
output_folder = Path("/home/stejan/snow_crystal_segmentation/step2_output/ltu23_m332_05")
#output_folder = Path("/home/stejan/snow_crystal_segmentation/step2_output/training_data")
#output_folder = Path("/home/stejan/hand_analyzed/ltu16_paper2/ltu16_step2")

## Get image lists (assuming they have the same names)
# read in the model name that is an extra suffix on the contour images
original_images = sorted(os.listdir(original_folder))
contour_images = sorted(os.listdir(contour_folder))

#  Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


# delete the extra files in case it was deleted already by another run over the data
check_original = {f for f in original_folder.iterdir() if f.is_file()}
check_contour = {f for f in contour_folder.iterdir() if f.is_file()}

normalized_contours = {
        "_".join(p.stem.split("_")[:-2]) + p.suffix
        for p in check_contour
        }

normalized_originals = {p.name for p in check_original}

extra_files = normalized_contours - normalized_originals
print(f"{extra_files=}")
#if extra_files:
#    for files in extra_files:
#        file_path = os.path.join(contour_folder, files)
#        os.remove(file_path)
#        print(f"Removed file {file_path}")
#else:
#    print("No extra files to delete")

# Mouse drawing variables
#drawing = False
#draw_color = (0, 0, 0)  # Default color (black)
#brush_size = 2  # Default size for drawing
polygon_points = []
draw_color = (255, 255, 255) #initial color
paint_one_pixel_mode = False
paint_value = 0
draw_mode = "contour"

# Mouse callback function
def draw_contours(event, x, y, flags, param):
    global polygon_points, drawing, draw_color, contour
    global draw_mode  # <-- add this global (set draw_mode = "contour" or "pixel")

    # Only act on left-click
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # -------------------------
    # N-MODE: paint 1 pixel
    # -------------------------
    if draw_mode == "pixel":
        h, w = contour.shape[:2]
        # Define 2x2 square (top-left anchored)
        for dy in (0, 1):
            for dx in (0, 1):
                px, py = x + dx, y + dy
                if 0 <= px < w and 0 <= py < h:
                    if len(contour.shape) == 3:
                        contour[py, px] = draw_color
                    else:
                        contour[py, px] = draw_color
        return

    # -------------------------
    # DEFAULT: polygon
    # -------------------------
    polygon_points.append((x, y))
    drawing = True

    # Draw a small circle at each point
    cv2.circle(contour, (x, y), 1, (0, 0, 0), -1)

    # Draw lines connecting the points
    if len(polygon_points) > 1:
        cv2.line(contour, polygon_points[-2], polygon_points[-1], draw_color, 1)

    # Close and fill polygon if clicking near the first point
    if len(polygon_points) > 2 and np.linalg.norm(np.array(polygon_points[0]) - np.array((x, y))) < 10:
        cv2.fillPoly(contour, [np.array(polygon_points, np.int32)], draw_color)
        for px, py in polygon_points:  # Change the previous polygon points to draw_color
            cv2.circle(contour, (px, py), 1, draw_color, -1)
        polygon_points.clear()
        drawing = False

def toggle_color(key):
    global draw_color
    if key == ord('p'):
        draw_color = (0, 0, 0) if draw_color == (255, 255, 255) else (255, 255, 255)
        print("Drawing color changed to Black" if draw_color == (0,0,0) else "Color changed to White")

def on_mouse(event, x, y, flags, param):
    global contour, original, display
    global paint_one_pixel_mode, paint_value, contour_path

    if event == cv2.EVENT_LBUTTONDOWN and paint_one_pixel_mode:
        # Safety clamp (in case)
        h, w = contour.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            contour[y, x] = paint_value  # paint exactly one pixel

            # Save immediately (optional; or you can save only when quitting)
            cv2.imwrite(contour_path, contour)

            # Refresh overlay display
            display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
            cv2.imshow("Edit Contour", display)

# Loop through all images
for orig_name, contour_name in zip(original_images, contour_images):
    # Load images
    original = cv2.imread(os.path.join(original_folder, orig_name))
    contour = cv2.imread(os.path.join(contour_folder, contour_name))
    original_path = os.path.join(original_folder, orig_name)
    contour_path = os.path.join(contour_folder, contour_name)

    # Ensure images have the same dimensions
    if original.shape != contour.shape:
        print(f"Skipping {orig_name}: Size mismatch")
        continue

    # Create a window and set mouse callback
    cv2.namedWindow("Original image")
    cv2.namedWindow("Edit Contour")
    cv2.setMouseCallback("Edit Contour", draw_contours)
 
    while True:
        # Convert grayscale to colour
        contour_display = contour.copy()
        black_pixels = (contour[:, :, 0] == 0) & (contour[:, :, 1] == 0) & (contour[:, :, 2] == 0) 
        contour_display[black_pixels] = [175, 0, 75]
 
        # Blend images for display
        display = cv2.addWeighted(original, 0.5, contour_display, 0.5, 0)
        cv2.imshow("Original image", original)
        cv2.imshow("Edit Contour", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save and move to next image
            cv2.imwrite(os.path.join(output_folder, contour_name), contour)
            #            shutil.move(contour_path, os.path.join(output_folder, "cropped_contours", contour_name))
#            shutil.move(original_path, os.path.join(output_folder, "cropped_original_png", orig_name))
            polygon_points.clear()
            print(f"Acceptable image, copy {orig_name} to {output_folder}")
            break

        elif key == ord('q'):  # Quit
            exit()

        elif key == ord("p"): # Change the drawing color
            toggle_color(key)

        elif key == ord("d"): # Pass
            if os.path.exists(original_path):
#                os.remove(contour_path)
                polygon_points.clear()
                print(f"Passing {contour_path}")
                break

        elif key == ord("r"):  # Crops from the RIGHT
            num_str = input("Enter number of pixels to mask from RIGHT: ")  # Get user input for pixel height
            if num_str.isdigit():  # Ensure it's a valid number
                mask_height = int(num_str)
                if mask_height > 0:
                # Apply white mask to the top `mask_height` pixels
                    contour[:, -mask_height:] = 255  # White mask on top N pixels
                # Overwrite the image
                    cv2.imwrite(contour_path, contour)
                    print(f"Applied and overwrote {contour_path}")
                # Update display with the modified contour
                    display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
                    cv2.imshow("Edit Contour", display)
                    print(f"Applied white mask of {mask_height} pixels from RIGHT")
            else:
                print("Invalid input, please enter a number.")

        elif key == ord("l"):  # Crops from the LEFT
            num_str = input("Enter number of pixels to mask from LEFT: ")  # Get user input for pixel height
            if num_str.isdigit():  # Ensure it's a valid number
                mask_height = int(num_str)
                if mask_height > 0:
                # Apply white mask to the top `mask_height` pixels
                    contour[:, :mask_height] = 255  # White mask on top N pixels
                # Overwrite the image
                    cv2.imwrite(contour_path, contour)
                    print(f"Applied and overwrote {contour_path}")
                # Update display with the modified contour
                    display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
                    cv2.imshow("Edit Contour", display)
                    print(f"Applied white mask of {mask_height} pixels from LEFT")
            else:
                print("Invalid input, please enter a number.")

        elif key == ord("t"):  # CCrops from the TOP
            num_str = input("Enter number of pixels to mask from TOP: ")  # Get user input for pixel height
            if num_str.isdigit():  # Ensure it's a valid number
                mask_height = int(num_str)
                if mask_height > 0:
                # Apply white mask to the top `mask_height` pixels
                    contour[:mask_height, :] = 255  # White mask on top N pixels
                # Overwrite the image
                    cv2.imwrite(contour_path, contour)
                    print(f"Applied and overwrote {contour_path}")
                # Update display with the modified contour
                    display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
                    cv2.imshow("Edit Contour", display)
                    print(f"Applied white mask of {mask_height} pixels from TOP")
            else:
                print("Invalid input, please enter a number.")

        elif key == ord("b"):  # Crops from the BOTTOM
            num_str = input("Enter number of pixels to mask from BOTTOM: ")  # Get user input for pixel height
            if num_str.isdigit():  # Ensure it's a valid number
                mask_height = int(num_str)
                if mask_height > 0:
                # Apply white mask to the top `mask_height` pixels
                    contour[-mask_height:, :] = 255  # White mask on top N pixels
                # Overwrite the image
                    cv2.imwrite(contour_path, contour)
                    print(f"Applied and overwrote {contour_path}")
                # Update display with the modified contour
                    display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
                    cv2.imshow("Edit Contour", display)
                    print(f"Applied white mask of {mask_height} pixels from BOTTOM")
            else:
                print("Invalid input, please enter a number.")

        elif key == ord("n"): # Paint pixels
            draw_mode = "pixel" if draw_mode != "pixel" else "contour"
            print("Mode: ", draw_mode)
