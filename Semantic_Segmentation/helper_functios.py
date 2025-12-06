from PIL import Image
import numpy as np

def get_image_mode(image_path):
    """
    Get the mode of an image such as image is RGBA OR RGB OR L(GRAYSCALE).

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The mode of the image, or None if an error occurs.
    Example:
        image_path = "path/to/image"
        mode = get_image_mode(image_path)
        print("Image Mode:", mode)

    """
    try:
        image = Image.open(image_path)
        return image.mode
    except Exception as e:
        print("Unable to open image:", image_path)
        print("Error:", str(e))
        return None



def find_min_max_pixel_value(image_path):
    """
    Finds the minimum and maximum pixel values in an image.

    Args:
        image_path (str): The path to the image file.
    Returns:
        tuple: A tuple containing the minimum and maximum pixel values.
    Example:
        image_path = 'path/to/image'
        min_value, max_value = find_min_max_pixel_value(image_path)
        print("Minimum pixel value:", min_value)
        print("Maximum pixel value:", max_value)
    """
    
     # Open the image using PIL
    img = Image.open(image_path)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Find the maximum and minimum pixel values in the array
    max_value = img_array.max()
    min_value = img_array.min()

    return min_value, max_value

