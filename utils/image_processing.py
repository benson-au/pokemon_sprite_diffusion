import io
import os
import logging
from PIL import Image
from typing import List, Optional, Tuple, Union
import torch
from torchvision import transforms
from tqdm import tqdm
import time
from IPython.display import Image as IPyImage

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
) -> bool:
    """
    Resizes an image to have the specified width and height.
    If the image already has the specified width and height, it is not resized.
    
    Args:
        new_width (int): The new width of the image.
        new_height (int): The new height of the image.
        image_path (str): The path to the image.
        output_folder (Optional[str], optional): The destination folder for the resized image.
            Defaults to None, in which case the resized image is placed in the same folder as the original image.
        inplace (bool, optional): If True, overwrite the original image. Defaults to False.
        verbose (bool, optional): If True, a log message is created upon successful resizing. Defaults to True.
    
    Returns:
        bool: True if able to successfully resize image, False otherwise.
    """
    try:
        with Image.open(image_path) as image:
            # Check if the image already has the desired dimensions.
            if image.size == (new_width, new_height):
                if verbose:
                    logging.info(f"Image {image_path} already has dimensions {new_width}x{new_height}. Skipping resize.")
                return True
                
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
        return True
    except FileNotFoundError:
        logging.error(f"Error: Image file not found at {image_path}")
    except Exception as e:
        logging.error(f"An error occurred while resizing image at {image_path}: {e}")
    return False

def resize_images(
    new_width: int,
    new_height: int,
    image_folder: str = './',
    output_folder: Optional[str] = None,
    inplace: bool = False
) -> None:
    """
    Resizes all images in a folder to have the specified width and height.
    If an image already has the specified width and height, it is not resized.
    
    Args:
        new_width (int): The new width of the images.
        new_height (int): The new height of the images.
        image_folder (str, optional): The path to the folder containing the images.
            Defaults to the current working directory.
        output_folder (Optional[str], optional): The path for the resized images.
            If provided and doesn't exist, it will be created. Defaults to None, in which case the resized images
            are placed in the same folder as the original images.
        inplace (bool, optional): If True, overwrite the original images. Defaults to False.
    """
    file_list = [filename for filename in os.listdir(image_folder)
             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not file_list:
        logging.warning(f"No images found in folder: {image_folder}")
        return
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Output folder created or already exists: {output_folder}")
    
    resized_count = 0
    skipped_count = 0
    failed_count = 0
    
    for filename in tqdm(file_list, desc="Processing images"):
        image_path = os.path.join(image_folder, filename)
        try:
            with Image.open(image_path) as image:
                # Check if the image already has the desired dimensions.
                if image.size == (new_width, new_height):
                    skipped_count += 1
                    continue
        except Exception as e:
            logging.error(f"Could not open image {image_path}: {e}")
            failed_count += 1
            continue
        
        success = resize_image(new_width, new_height, image_path, output_folder, inplace, verbose=False)
        if success:
            resized_count += 1
        else:
            failed_count += 1

    logging.info(f"Processing complete. Resized: {resized_count} images, Skipped: {skipped_count} images, Failed: {failed_count} images.")

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
    mode: str = "RGB"
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
            For a given image and a given channel, the values of the tensor lie in the interval [-1, 1].
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

def tensor_to_pil(
    tensor: torch.Tensor
) -> Image.Image:
    """
    Converts a (single) image tensor with values in [-1, 1] to a PIL Image.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W) with values in [-1, 1].
        
    Returns:
        Image.Image: A PIL image with pixel values in [0, 255].
    """
    # Invert the transformation: map [-1,1] to [0,1]
    tensor = (tensor + 1) / 2.0
    # Clamp the tensor to ensure the values are within [0,1]
    tensor = tensor.clamp(0, 1)
    # Convert to a PIL image using ToPILImage.
    pil_image = transforms.ToPILImage()(tensor.cpu())
    return pil_image

def batch_to_pil(
    batch: torch.Tensor,
    output_folder: Optional[str] = None,
    file_format: Optional[str] = None,
    base_file_name: str = 'image',
) -> List[Image.Image]:
    """
    Converts a batch of image tensors with values in [-1, 1] to a list of PIL images.
    Intended to invert the transformation from load_images_tensor().
    
    Args:
        batch (torch.Tensor): A tensor of shape (num_images, C, H, W) with values in [-1, 1].
        output_folder (Optional[str], optional): The output folder for the PIL images.
            If provided and doesn't exist, it will be created. Defaults to None, in which case the PIL images
            are not saved.
        file_format (Optional[str], optional): The format for the PIL images if they are to be saved (e.g., "PNG" or "JPEG").
        base_file_name (Optional[str], optional): The base file name for the PIL images if they are to be saved.
            PIL images will be saved as f"{base_file_name}_{idx:03d}.{file_format.lower()}".
    
    Returns:
        List[Image.Image]: A list of PIL images.
    """
    pil_images = []
    # Ensure the batch is on CPU and iterate over the batch dimension.
    for img_tensor in batch.cpu():
        pil_images.append(tensor_to_pil(img_tensor))
    
    # If output_folder is provided, save PIL images to output_folder.
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Output folder created or already exists: {output_folder}")
        for idx, img in enumerate(pil_images):
            filename = f"{base_file_name}_{idx:03d}.{file_format.lower()}"
            output_path = os.path.join(output_folder, filename)
            img.save(output_path, file_format)
        
    return pil_images

def pils_to_gif(
    frames: List[Image.Image],
    display: bool = False,
    filename: Optional[str] = None,
    duration: int = 200,
    loop: int = 0
) -> Union[None, IPyImage]:
    """
    Converts a list of PIL images to a GIF.
    
    Args:
        frames (List[Image.Image]): A list of images to be overlaid into a GIF.
        display (bool, optional): If True, displays the resulting GIF (assumes an IPython environment). Defaults to False.
        filename (Optional[str], optional): The filename of the GIF.
            If not provided, the file name will be the time of creation.
            If provided, the time will not be added to the file name.
        duration (int, optional): The amount of time for each frame of the GIF.
            Can also input a list of durations for non-uniform times.
        loop (int, optional): The number of times to loop the GIF.
            The default option of 0 corresponds to indefinite looping.
    
    Returns:
        Union[None, IPyImage]:
            If the display option is set to True, then an IPyImage is returned.
    """    
    logdir = "./"
    if filename:
        filepath = os.path.join(logdir, filename + "_" + '.gif')
    else:
        filepath = os.path.join(logdir, 'animation_' + time.strftime("%d-%m-%Y_%H-%M-%S") + '.gif')
        
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop
    )
    
    with open(filepath, 'wb') as f:
        f.write(buffer.getvalue())

    if display:
        return IPyImage(data=buffer.getvalue(), format='gif')
    
def pil_grid(
    images: List[Image.Image], 
    cols: int
) -> Image.Image:
    """
    Combines a list of PIL images into a single PIL image.
    
    Args:
        images (List[Image.Image]): A list of images to be concatenated into a grid.
        cols (int): The number of columns in the grid.
        
    Returns:
        Image.Image: A PIL image.
    """    
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        x, y = (idx % cols) * w, (idx // cols) * h
        canvas.paste(img, (x, y))
    return canvas

# Testing block
if __name__ == "__main__":
    # Example test for load_images_tensor
    try:
        images = load_images_tensor("./images", image_size=(64, 64), mode="RGBA")
        logging.info(f"Loaded tensor shape: {images.shape}")
    except Exception as e:
        logging.error(f"Error loading images: {e}")
