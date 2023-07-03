# Interior AI

## Image Segmentation with Class based styling

### Implementation of Semantic segmentation using keras model and applying class based style transfer

<p align="center">
  <img src="Documentation Images/basicflowchart.png" alt="Interior AI" width="50%" height="50%">
</p>



**Semantic segmentation** is a computer vision task that involves assigning a specific class label to each pixel in an image, enabling the identification and differentiation of objects and regions based on their semantic meaning.

## Model Architecture 
There are indeed several models for semantic segmentation, such as Fully Convolutional Networks (FCN), SegNet, DeepLab, and Mask R-CNN. However, **VGG-UNet** is a popular and widely used architecture for Semantic segmentation.<p>VGG-UNet combines the strengths of two different models, namely **VGG16** and **UNet**, to achieve better performance in semantic segmentation tasks.</p>
The VGG-UNet model architecture combines VGG and UNet into a single network for semantic segmentation:
- VGG serves as the encoder layer: VGG, known for its high accuracy in image classification, is used as the encoder to extract feature representations from the input image. It captures hierarchical and abstract features through a series of convolutional layers.
- UNet functions as the decoder: UNet, specifically designed for semantic segmentation tasks, acts as the decoder. It employs a symmetrical U-shaped architecture with skip connections. These connections allow the integration of both high-level and low-level features while preserving the spatial information of the input image, enabling accurate segmentation results.

## Creating Environment
```
# Step 1: Install Anaconda or Miniconda (skip this if already installed)

# Step 2: Navigate to the directory containing the YAML file
cd /path/to/yaml_directory

# Step 3: Create the conda environment from the YAML file
conda env create -f requiremnets.yaml

# Step 4: Activate the conda environment
conda activate my_env
```
## Preparing the dataset for training
You need to make two folders:

- Images Folder - For all the images
   - Check the mode of the image and convert it to 24-bit(RGB) ,if it is initially in 32-bit(RGBA) format
- Mask Folder - For the corresponding ground truth segmentation images
    - Convert the ground truth segmentation into the semantic labels (*assigning class categories to each pixel according to the rgb value of the classes in the given ground truth* ) 
    - Save the labels in the image form in 8-bit(L Mode)
- Make sure that the naming and size is same of the image and their corresponding mask and semantic labels
- Divide the images and the corresponding semantic labels into separate test and train folders
- For the segmentation maps, do not use the jpg format as jpg is lossy and the pixel values might change. Use png format instead

## [Train the model and make Predictions](https://github.com/riteshcanfly/InteriorAI/blob/Atithi/Semantic_Segmentation/Semantic_Segmentation/Semantic_Segmentation_Model.ipynb)
To train the model and make predictions using it, please refer to the provided link for detailed instructions. Ensure that your dataset has been pre-processed and prepared adequately, following the recommended steps, before initiating the training process. Additionally, save your predictions in PNG format, which can be further utilized for class-based styling.

## Semantic Segmentation Result

<p align="center">
  <img src="Documentation Images/semanitc_segmentation_output.png" alt="Semantic Segmentation Output" width="50%" height = "50%">
</p>

---
To perform class-based style transfer, three steps need to be followed:

- **Classs Identification** 
   - Utilize the predictions from the semantic segmentation model to identify the classes of interest. Determine the class to which you want to apply style transfer.
   - Based on the identified class, create a binary mask where the pixels belonging to the class of interest are marked as foreground (white) and the remaining pixels are considered as background (black). This binary mask can be generated by setting the pixels corresponding to the class of interest to white and the rest to black, using the predictions obtained from the semantic segmentation model.

   <p align="center">
    <img src="Documentation Images/binary mask.png" alt = "Class Identification" width="50%" height = "50%">
  </p>


- **Style Selection**
    - Once the classes have been identified, you can select a desired artistic style for each class, determining the specific visual appearance you wish to achieve. This style selection process allows you to choose a unique aesthetic for each class region, influencing how they will be stylized in the final output.

- **Neural Style Transfer** 
   - Neural Style Transfer (NST) is a technique that combines the content of one image with the artistic style of another image to create a visually appealing output.
   - To apply the chosen styles, a pretrained NST model is employed. This model, trained on a large dataset, has learned to extract and transfer style representations from reference images to target images.
   -  By utilizing a pretrained model in Neural Style Transfer (NST), which has learned style representations from a large dataset, efficient style transfer can be applied to the entire input image.
   - The resulting stylized image from NST can serve as a foundation for applying class-based style transfer techniques, enabling the customization of specific regions or objects within the image with unique artistic styles.
   <p align="center">
    <img src="Documentation Images/nst.png" alt = "Class Identification" width="50%" height = "50%">
  </p>

- **Class based Style Transfer**

    <p align="center">
    <img src="Documentation Images/class_based_syle_transfer.png" alt = "Class Identification" width="50%" height = "50%">
  </p>

    - The basic formula used to perform this task is **U(c, S) = (Rc ∗ TS) + (1 − Rc)∗ I** where,
        - U(c, S) represents the stylized output image for a specific class c, given a style image S.
        - Rc is the binary mask for the class c, where Rc = 1 for pixels belonging to class c (foreground) and Rc = 0 for pixels outside class c (background).
        - 1-Rc  is the inverted form of Rc where foreground becomes background and vica-versa
        - TS is the style transfer result obtained by applying neural style transfer using the style image S on the input image I.
        - I is the input image or the original image before stylization.

    - The steps in the formula U(c, S) = (Rc * TS) + ((1 - Rc) * I) are as follows:
        - Rc * TS: Multiply the style image (TS) with the binary mask (Rc). This step applies the artistic style to the specific class regions defined by the binary mask. The result is the stylized image for the class regions
        - (1 - Rc) * I: Subtract the binary mask (Rc) from 1 and multiply it with the input image (I). This step preserves the original appearance of the input image outside the class regions. The result is the non-stylized image for the background regions.
        - Add the results from steps 1 and 2: Add the stylized image from step 1 and the non-stylized image from step 2 together. This step combines the stylized class regions with the preserved background regions. The result is the final output image after class-specific style transfer.
        - Make sure that all the images used in the formula are of same size.
    
    <p align="center">
    <img src="Documentation Images/final_output_cbs.png" alt = "Class Based Style Transfer" >
    </p>
