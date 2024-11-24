import os
import pickle
import numpy as np
from PIL import Image

# Define paths
dataset_path = r'c:\Users\gargr\OneDrive\Documents\spikingResformer\SpikingResformer\cifar-10-python\cifar-10-batches-py'  # Your CIFAR-10 extracted directory
output_path = r'c:\Users\gargr\OneDrive\Documents\spikingResformer\SpikingResformer\dataset'  # Path where you want to save the new structure

# CIFAR-10 classes
label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Helper function to load batch files
def load_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Create folders for train and val
os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)

# Load and save training batches
for i in range(1, 6):
    batch = load_batch(os.path.join(dataset_path, f'data_batch_{i}'))
    for j, (data, label) in enumerate(zip(batch[b'data'], batch[b'labels'])):
        img = data.reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape and transpose to HWC
        img = Image.fromarray(img)
        class_dir = os.path.join(output_path, 'train', label_names[label])
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f'{i}_{j}.png'))

# Load and save test batch
test_batch = load_batch(os.path.join(dataset_path, 'test_batch'))
for j, (data, label) in enumerate(zip(test_batch[b'data'], test_batch[b'labels'])):
    img = data.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img)
    class_dir = os.path.join(output_path, 'val', label_names[label])
    os.makedirs(class_dir, exist_ok=True)
    img.save(os.path.join(class_dir, f'test_{j}.png'))
