import cv2
import numpy as np
from PIL import Image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    Works for both grayscale and RGB images.
    
    Args:
        image: PIL Image or numpy array (RGB or L)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image of the same type as input (PIL or Numpy)
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_np = np.array(image)
    else:
        img_np = image.copy()

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img_np.shape) == 2:  # Grayscale
        enhanced = clahe.apply(img_np)
    else:  # RGB
        # Convert to LAB color space
        # L = Lightness, A/B = Color channels
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L-channel
        l_enhanced = clahe.apply(l)
        
        # Merge back and convert to RGB
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    if is_pil:
        return Image.fromarray(enhanced)
    return enhanced

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py path/to/image.jpg")
        sys.exit(1)
        
    input_path = sys.argv[1]
    img = Image.open(input_path).convert('RGB')
    enhanced = apply_clahe(img)
    
    base = os.path.splitext(input_path)[0]
    enhanced.save(f"{base}_clahe.jpg")
    print(f"✅ Saved CLAHE enhanced image to {base}_clahe.jpg")
