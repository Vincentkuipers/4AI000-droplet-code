import os
from PIL import Image
import numpy as np

def add_noise(image, intensity):
    np.random.seed(42)  # Set a seed for reproducibility
    image_array = np.array(image)
    noise = np.random.normal(scale=intensity, size=image_array.shape).astype(np.uint8)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Specify the input and output directories
# input_dir = "./import os
input_dir = "./Test_sets/Data_Normal_98"
from PIL import Image
import numpy as np

def add_noise(image, intensity):
    np.random.seed(42)  # Set a seed for reproducibility
    image_array = np.array(image)
    noise = np.random.normal(scale=intensity, size=image_array.shape).astype(np.uint8)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Specify the input and output directories
input_dir = "./Test_sets/Data_Normal_98"
output_dir = "./Test_sets/Data_Noise5_98"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Load the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)
        
        # Add noise to the image with intensity of 3
        noisy_image = add_noise(image, intensity=5)
        
        # Save the noisy image to the output directory
        output_path = os.path.join(output_dir, filename)
        noisy_image.save(output_path)
output_dir = "./Test_sets/Data_Noise5_98"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Load the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)
        
        # Add noise to the image with intensity of 3
        noisy_image = add_noise(image, intensity=3)
        
        # Save the noisy image to the output directory
        output_path = os.path.join(output_dir, filename)
        noisy_image.save(output_path)
