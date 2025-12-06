import os
from PIL import Image
import cv2
import numpy as np
import random
import shutil

import os
from PIL import Image

def convert_32bit_to_24bit(image_dir,output_dir):
    """
    Convert 32-bit(RGBA)PNG images in the specified directory to 24-bit (RGB) format.

    Args:
        image_dir (str): The directory path where the 32-bit PNG images are located.
        output_dir (str): The directory path where the 24-bit PNG images will be saved.
        
    Returns:
        None: The function saves the converted images in place.
    
    Example usage:
        image_dir = "Path/to/images/directory"
        output_dir = "path/to/output/dir"
        convert_32bit_to_24bit(image_dir,output_dir)
    """
    
    # Loop through all files in the directory
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            # Load the image
            image_path = os.path.join(image_dir, filename)
            img = Image.open(image_path)

            # Convert to 24-bit color depth
            img_24bit = img.convert("RGB")
           
            # SaveS the converted image in the output_dir
            output_filename = os.path.splitext(filename)[0] + '.png'
            img_24bit.save(os.path.join(output_dir, output_filename))



def convert_masks_to_semantic_labels(input_folder, output_folder):
    """
    Converts RGB color image masks to class image masks based on predefined RGB values.

    Args:
        input_folder (str): The path to the folder containing the RGB color image masks.
        output_folder (str): The path to the folder where the class image masks will be saved.
        
    Returns:
        None: The function saves the labeled images in the output folder.
    Example:
        input_folder = "/path/to/input/folder"
        output_folder = "/path/to/output.folder"
       convert_masks_to_semantic_label(input_folder, output_folder)
    """
    rgb_values = {
        (170, 255, 255): 0,    # wall
        (170, 170, 255): 1,    # table
        (255, 85, 255): 2,     # storage
        (170, 85, 255): 3,     # TV Unit
        (255, 170, 255): 4,    # chair
        (85, 255, 255): 5,     # sofa
        (255, 255, 170): 6,    # curtain
        (85, 255, 170): 7,     # ceiling
        (170, 170, 170): 8,    # rug
        (85, 170, 170): 9,     # floors
        (0, 0, 0): 10,         # others
    }

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of RGB color image masks in the input folder
    rgb_mask_files = os.listdir(input_folder)

    for rgb_mask_file in rgb_mask_files:
        # Load the RGB color image mask
        rgb_mask_path = os.path.join(input_folder, rgb_mask_file)
        rgb_mask = cv2.imread(rgb_mask_path)
        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)

        # Create the class image mask
        class_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        for rgb, class_label in rgb_values.items():
            mask = np.all(rgb_mask == np.array(rgb), axis=2)
            class_mask[mask] = class_label

        # Save the class image mask as PNG to preserve pixel values
        class_mask_path = os.path.join(output_folder, rgb_mask_file)
        cv2.imwrite(class_mask_path, class_mask)



def rename_files_in_folder(folder_path, new_name_prefix, start_index):
    
    """
    Renames the files in a folder with a specified prefix and starting index.

    Args:
        folder_path (str): The path to the folder containing the files.
        new_name_prefix (str): The prefix to prepend to the new file names.
        start_index (int): The starting index for renaming the files.
        
    Returns:
        None: The function saves the renamed images in place.
        
     Example:
       folder_path = "/path/to/folder"
       new_name_prefix = "image_"
       start_index = 541
       rename_files_in_folder(folder_path, new_name_prefix, start_index)

    """
    
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Sort the files in the desired order
    sorted_files = sorted(files)

    # Rename the files in the desired order
    for index, file_name in enumerate(sorted_files):
        file_extension = os.path.splitext(file_name)[1]
        new_file_name = f"{new_name_prefix}{start_index + index}{file_extension}"
        src = os.path.join(folder_path, file_name)
        dst = os.path.join(folder_path, new_file_name)
        os.rename(src, dst)

def train_test_split(input_folder, train_folder, test_folder, split_ratio, random_seed=None):
    
    """
    Splits the files in the input folder into two separate folders, train_folder and test_folder,
       based on the provided split_ratio.

    Args:
        input_folder (str): Path to the input folder containing the files to be split.
        train_folder (str): Path to the train folder where a portion of the files will be copied.
        test_folder (str):  Path to the test folder where the remaining files will be copied.
        split_ratio (float): Ratio of files to be allocated for the train_folder.
                            The value should be between 0 and 1, where 0 represents no files in the train_folder, and 1 represents all files in the train_folder.
        random_seed (int or None, optional): Seed value for the random shuffling of files.
                                             If provided, the shuffling will be consistent across multiple function calls.
                                             Defaults to None.
                                             
    Returns:
        None: The function copies the files to the train_folder and test_folder.
        
    Eample:
        input_folder = '/path/to/input_folder'
        train_folder = '/path/to/train_folder'
        test_folder = '/path/to/test_folder'
        split_ratio = 0.8
        random_seed = 42
        train_test_split(input_folder, train_folder, test_folder, split_ratio, random_seed)
    """

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)
    
    # Calculate the number of files for the first folder based on the split ratio
    num_files_1 = int(len(files) * split_ratio)
    
    # Set the random seed for consistent shuffling
    if random_seed is not None:
        random.seed(random_seed)
    
    # Shuffle the files randomly
    random.shuffle(files)
    
    # Create the output folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Copy files to the train folder
    for file in files[:num_files_1]:
        src = os.path.join(input_folder, file)
        dst = os.path.join(train_folder, file)
        shutil.copy(src, dst)
    
    # Copy files to the test folder
    for file in files[num_files_1:]:
        src = os.path.join(input_folder, file)
        dst = os.path.join(test_folder, file)
        shutil.copy(src, dst)
    