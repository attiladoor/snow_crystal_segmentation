import os
import re

# Specify the directory path (or list of directories)
directories = [
    '/home/stejan/snow_crystal_segmentation/comparing_ML_m323']#,
#'/home/stejan/snow_crystal_segmentation/comparing_output_HAND/', 
#   '/home/stejan/snow_crystal_segmentation/comparing_output_ML_m33007/']  # Replace with your folder paths

# Regular expression pattern to match '_m' followed by numbers, underscore, and more numbers
pattern = r'_m\d+'

for directory in directories:
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist, skipping...")
        continue
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the pattern
        if re.search(pattern, filename):
            # Construct the new filename by removing the matched pattern
            new_filename = re.sub(pattern, '', filename)
            
            # Create full file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            
            # Rename the file
            try:
                os.rename(old_file, new_file)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")
