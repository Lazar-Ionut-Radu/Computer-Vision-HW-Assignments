import numpy as np

from PIL import Image
from pathlib import Path

def load_images(dir_path: str | Path) -> dict[str, np.ndarray]:
    """
    Loads all images from a directory as NumPy arrays.
    Returns a dict mapping file names to NumPy arrays.
    """
    
    dir = Path(dir_path)
    images = {}
    
    for path in dir.iterdir():
        if not path.is_file():
            continue
        
        try:
            with Image.open(path) as image:
                images[path.name] = np.array(image)
        except:
            continue
        
    return images

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB NumPy array to grayscale.
    """
    
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
 