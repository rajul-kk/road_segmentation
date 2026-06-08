import cv2
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image

def clean_mask(mask_array, kernel_size=5):
    """
    Perform morphological cleaning on a binary road mask.
    
    Args:
        mask_array: Numpy array (0 or 255)
        kernel_size: Size of the structuring element
        
    Returns:
        Cleaned numpy array
    """
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Opening to remove noise (small islands)
    opening = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
    
    # Closing to fill gaps (holes in roads)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    return closing

def get_skeleton(mask_array):
    """
    Extract the 1-pixel wide centerline from a thick road mask.
    
    Args:
        mask_array: Numpy array (0 or 255)
        
    Returns:
        Skeletonized numpy array (0 or 255)
    """
    # Normalize to 0-1 for skimage
    binary = mask_array > 127
    
    # Skeletonize
    skeleton = skeletonize(binary)
    
    # Convert back to 0-255
    return (skeleton * 255).astype(np.uint8)

def post_process_image(image_path, output_dir=None):
    """
    Full pipeline: Clean and Skeletonize an image.
    """
    # Load
    mask = np.array(Image.open(image_path).convert('L'))
    
    # Process
    cleaned = clean_mask(mask)
    skeleton = get_skeleton(cleaned)
    
    return cleaned, skeleton

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python src/post_process.py path/to/mask.png")
        sys.exit(1)
        
    input_path = sys.argv[1]
    cleaned, skeleton = post_process_image(input_path)
    
    # Save examples
    base = os.path.splitext(input_path)[0]
    Image.fromarray(cleaned).save(f"{base}_cleaned.png")
    Image.fromarray(skeleton).save(f"{base}_skeleton.png")
    print(f"[OK] Processed {input_path}")
