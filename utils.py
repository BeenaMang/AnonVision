### Utility functions for image processing and file handling


import os
from PIL import Image
import cv2


def get_image_files(directory):
    """
    Get list of image files in a directory
    
    Args:
        directory (str): Path to directory
        
    Returns:
        list: List of image file paths
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    if not os.path.exists(directory):
        return []
    
    image_files = []
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(directory, filename))
    
    return image_files


def validate_image_path(image_path):
    """
    Validate if a path points to a valid image file
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        bool: True if valid image file
    """
    if not os.path.exists(image_path):
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    ext = os.path.splitext(image_path)[1].lower()
    
    return ext in valid_extensions


def resize_image_for_display(image, max_width=800, max_height=600):
    """
    Resize image to fit within display area while maintaining aspect ratio
    
    Args:
        image: PIL Image or numpy array
        max_width (int): Maximum width in pixels
        max_height (int): Maximum height in pixels
        
    Returns:
        PIL.Image: Resized image
    """
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        # Assume it's a numpy array from OpenCV
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Get current dimensions
    width, height = image.size
    
    # Calculate scaling factor
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize image
    if scale_factor < 1.0:
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image
    
    return image


def create_directory(directory_path):
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path (str): Path to directory
    """
    os.makedirs(directory_path, exist_ok=True)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test directory creation
    create_directory('test_output')
    print("✓ Directory creation works")
    
    # Test image file listing
    image_files = get_image_files('test_images')
    print(f"✓ Found {len(image_files)} image files")
    
    print("Utility functions working correctly!")