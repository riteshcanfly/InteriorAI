import os
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
def rgb_semantic_segmentation_mask_to_binary_mask(segmented_mask, class_name, save_path):
    """
    Converts a segmented mask image to a binary mask for a specific class.

    Args:
        segmented_mask (numpy.ndarray): Segmented mask image.
        class_name (str): Name of the class to extract the binary mask for.
        save_path (str): File path to save the binary mask image.

    Returns:
        The function saves the binary mask at the provided path

    Raises:
        KeyError: If the provided class name is not found in the dictionary.

    Example:
        segmented_mask = 'path/to/segmented_mask.png'
        class_name = 'Wall'
        save_path = 'path/to/binary_mask.png'
        convert_to_binary_mask(segmented_mask, class_name, save_path)

    """

    class_colors = {
        "Wall": np.array([20, 215, 197]),
        "Table": np.array([207, 248, 132]),
        "Storage": np.array([183, 244, 155]),
        "TV Unit": np.array([144, 71, 111]),
        "Chair": np.array([128, 48, 71]),
        "Sofa": np.array([50, 158, 75]),
        "Curtain": np.array([241, 169, 37]),
        "Ceiling": np.array([222, 181, 51]),
        "Rug": np.array([244, 104, 161]),
        "Floor": np.array([31, 133, 226]),
        "Others": np.array([204, 47, 7])
    }
    
    # Get the class color RGB value based on the class name
    class_color_rgb = class_colors.get(class_name)

    # Check if the class color RGB value is None, indicating that the class name is not found in the dictionary
    if class_color_rgb is None:
        # Create a string of available classes by joining the keys of the class_colors dictionary with commas
        available_classes = ", ".join(class_colors.keys())
        # Raise a KeyError with an informative error message
        raise KeyError(f"Class '{class_name}' not found in the dictionary. Available classes: {available_classes}")
    
    segmented_image = cv2.imread(segmented_mask)

    # segmented mask to grayscale
    gray_mask = cv2.cvtColor( segmented_image, cv2.COLOR_BGR2GRAY)

    # Convert RGB to BGR format
    class_color_bgr = class_color_rgb[::-1]

    # Create the binary mask by comparing pixel values with the class color
    binary_mask = np.zeros_like(gray_mask)
    binary_mask[np.all( segmented_image == class_color_bgr, axis=-1)] = 255

    # Save the binary mask as an image
    cv2.imwrite(save_path, binary_mask)


def resize_style_image(image_path,new_width,new_height):
    """
    The function resizes a single image

    Args:
        folder_path (str): Path to the folder containing the PNG images to be resized.
        new_width (int): The desired width of the resized images.
        new_height (int): The desired height of the resized images.
    Return:
        None: The function saves the resized images.
    Example:
        new_width = 1920
        new_height = 1080
        image_path = "/path/to/image"
        resize_style_image(image_path,new_width,new_height)
        
    """
    image = Image.open(image_path)
    resized_image = image.resize((new_width, new_height))
    resized_image.save(image_path)
    
def apply_class_specific_style_transfer(input_image, binary_mask, style_image, output_path):
    """
    Applies class-specific style transfer to an input image based on a binary mask and a style image.

    Args:
        input_image (str): File path of the input image.
        binary_mask (str): File path of the binary mask image.
        style_image (str): File path of the style image.
        output_path (str): File path to save the stylized output image.

    Returns:
        The Function saves the stylized output image in the output path provided

    Example:
        input_image = "input_image.jpg"
        binary_mask = "binary_mask.png"
        style_image = "style_image.jpg"
        output_path = "stylized_output.jpg"
        apply_object_specific_style_transfer(input_image, binary_mask, style_image, output_path)
    """

    # Load input image, binary mask, and style image
    input_image = cv2.imread(input_image)
    binary_mask = cv2.imread(binary_mask, 0)
    style_image = cv2.imread(style_image)

    # Resize binary mask and style image to match input image size
    binary_mask_resized = cv2.resize(binary_mask, (input_image.shape[1], input_image.shape[0]))
    style_image_resized = cv2.resize(style_image, (input_image.shape[1], input_image.shape[0]))

    # Convert binary mask to color channels
    binary_mask_channels = cv2.cvtColor(binary_mask_resized, cv2.COLOR_GRAY2BGR)

    # Invert binary mask
    inverted_mask = cv2.bitwise_not(binary_mask_channels)

    # Multiply style image with binary mask
    rc_ts = cv2.bitwise_and(binary_mask_channels, style_image_resized)

    # Multiply input image with inverted binary mask
    one_minus_rc_i = cv2.bitwise_and(inverted_mask, input_image)

    # Add the style and non-style parts
    stylized_output = cv2.add(rc_ts, one_minus_rc_i)

    # Save the stylized output image
    cv2.imwrite(output_path, stylized_output)

def nst_load_img(path_to_img):
    """
    Loads an image from the given file path and performs preprocessing.

    Args:
        path_to_img (str): File path of the input image.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    max_dim = 512

    # Read image file
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    # Resize image
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img
