from PIL import Image
import numpy as np

def preprocess_image(file_path, target_size=(30, 30)):
    """Load and preprocess image for model prediction"""
    img = Image.open(file_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    return np.array(img)