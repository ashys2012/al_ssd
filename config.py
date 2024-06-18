import torch

BATCH_SIZE = 8 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 2 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# Training images and XML files directory.
TRAIN_DIR = '/home/achazhoor/Documents/2024/ssd/data/Train'
# Validation images and XML files direcata/Train/Train/JPEGImagetory.
VALID_DIR = '/home/achazhoor/Documents/2024/ssd/data/Val'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = '/home/achazhoor/Documents/2024/ssd/outputs'