import os
import logging
from PIL import Image
from typing import List, Optional, Tuple
import torch
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def resize_image(
    new_width: int,
    new_height: int,
    image_path: str,
    output_folder: Optional[str] = None,
    inplace: bool = False,
    verbose: bool = True
) -> None:
    """
    Resizes an image to have the specified width and height.
    
    Args:
        new_width (int): The new width of the image.
        new_height (int): The new height of the image.
        image_path (str): The path to the image.
        output_folder (Optional[str], optional): The destination folder for the resized image.
            Defaults to None, in which case the resized image is placed in the same folder as the original image.
        inplace (bool, optional): If True, overwrite the original image. Defaults to False.
        verbose (bool, optional): If True, a log message is created upon successful resizing. Defaults to True.
    """
    try:
        with Image.open(image_path) as image:
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            
            if inplace:
                resized_image.save(image_path)
                logging.info(f"Resized image saved in-place at {image_path}")
            else:
                directory, filename = os.path.split(image_path)
                basename, extension = os.path.splitext(filename)
                new_file_name: str = f"{basename}_{new_width}x{new_height}{extension}"
                # If output_folder is provided, use it; otherwise, use the original directory.
                output_path: str = os.path.join(output_folder, new_file_name) if output_folder else os.path.join(directory, new_file_name)
                resized_image.save(output_path)
                if verbose:
                	logging.info(f"Resized image saved to {output_path}")
    except FileNotFoundError:
        logging.error(f"Error: Image file not found at {image_path}")
    except Exception as e:
        logging.error(f"An error occurred while resizing image at {image_path}: {e}")

def resize_images(
    new_width: int,
    new_height: int,
    image_folder: str = './',
    output_folder: Optional[str] = None,
    inplace: bool = False
) -> None:
    """
    Resizes all images in a folder to have the specified width and height.
    
    Args:
        new_width (int): The new width of the images.
        new_height (int): The new height of the images.
        image_folder (str, optional): The path to the folder containing the images.
            Defaults to the current working directory.
        output_folder (Optional[str], optional): The destination folder for the resized images.
            If provided and doesn't exist, it will be created. Defaults to None, in which case the resized images
            are placed in the same folder as the original images.
        inplace (bool, optional): If True, overwrite the original images. Defaults to False.
    """
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Output folder created or already exists: {output_folder}")
    
    resized_count = 0
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path: str = os.path.join(image_folder, filename)
            resize_image(new_width, new_height, image_path, output_folder, inplace, False)
            resized_count += 1
    
    if not resized_count:
        logging.warning(f"No images found in folder: {image_folder}")
    else:
    	logging.info(f"Successfully resized {resized_count} images to size {new_width}x{new_height}.")

def check_image_sizes(
    target_width: int,
    target_height: int,
    image_folder: str = "./"
) -> bool:
    """
    Checks if all images in a folder have the specified width and height.
    
    Args:
        target_width (int): The expected width of the images.
        target_height (int): The expected height of the images.
        image_folder (str, optional): The folder containing the images.
            Defaults to the current working directory.
    
    Returns:
        bool: True if all images have the target size, False otherwise.
    """
    incorrect = 0
    error_processing = 0
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            filepath: str = os.path.join(image_folder, filename)
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    if width != target_width or height != target_height:
                        logging.warning(f"Image '{filename}' has size {width}x{height}, expected {target_width}x{target_height}")
                        incorrect += 1
            except Exception as e:
                logging.error(f"Error processing '{filename}': {e}")
                error_processing += 1
    if incorrect or error_processing:
    	logging.info(f"{incorrect} image(s) found with unexpected size, unable to process {error_processing} image(s).") 
    return not (incorrect or error_processing)

def load_images_tensor(
    folder_path: str = './',
    image_size: Optional[Tuple[int, int]] = None,
    mode: str = "RGBA"
) -> torch.Tensor:
    """
    Loads all images in the given folder, converts them to the specified mode (and optionally to the specified size),
    and returns a single tensor of shape (num_images, C, H, W).
    Intended to be passed to TensorDataset.
    
    Args:
        folder_path (str, optional): The directory where the images are stored.
        	Defaults to the current working directory.
        image_size (Optional[Tuple[int, int]]): If provided, resize each image to this size (width, height).
        mode (str): The color mode to convert the images to (e.g., "RGBA" or "RGB").
        
    Returns:
        torch.Tensor: A tensor of shape (num_images, C, H, W), where C is 4 for "RGBA" or 3 for "RGB".
    """
    image_tensors = []
    
    transform_list = []
    if image_size is not None:
        transform_list.append(transforms.Resize(image_size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Lambda(lambda x: x * 2 - 1))
    transform = transforms.Compose(transform_list)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                if img.mode != mode:
                    img = img.convert(mode)
                tensor = transform(img)
                image_tensors.append(tensor)
    
    if not image_tensors:
        raise ValueError("No images found in the provided folder.")
    
    return torch.stack(image_tensors)

# Testing block
if __name__ == "__main__":
    # Example test for load_images_tensor
    try:
        images = load_images_tensor("./images", image_size=(64, 64), mode="RGBA")
        logging.info(f"Loaded tensor shape: {images.shape}")
    except Exception as e:
        logging.error(f"Error loading images: {e}")

