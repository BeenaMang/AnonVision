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


def resize_image_for_display(image, max_width=800, max_height=600, fill_background=(217, 217, 217)):
    """
    Resize image to fit within display area while maintaining aspect ratio,
    and pad with background color to fill the entire display area
    
    Args:
        image: PIL Image or numpy array
        max_width (int): Maximum width in pixels
        max_height (int): Maximum height in pixels
        fill_background (tuple or str): Background color for padding - RGB tuple like (217, 217, 217) for gray85 equivalent
        
    Returns:
        PIL.Image: Resized and padded image that fills the display area
    """
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        # Assume it's a numpy array from OpenCV
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Get current dimensions
    width, height = image.size
    
    # Calculate scaling factor to fit within the display area
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale_factor = min(width_ratio, height_ratio)
    
    # Calculate new dimensions after scaling
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert color specification if needed
    if isinstance(fill_background, str):
        # Convert common color names to RGB tuples
        color_map = {
            'gray85': (217, 217, 217),
            'lightgray': (211, 211, 211),
            'gray': (128, 128, 128),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        fill_background = color_map.get(fill_background.lower(), (217, 217, 217))
    
    # Create a new image with the target dimensions and background color
    display_image = Image.new('RGB', (max_width, max_height), fill_background)
    
    # Calculate position to center the resized image
    x_offset = (max_width - new_width) // 2
    y_offset = (max_height - new_height) // 2
    
    # Paste the resized image onto the background
    display_image.paste(resized_image, (x_offset, y_offset))
    
    return display_image


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