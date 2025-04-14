import torch
import os

# This is to run linux trained model on windows to avoid posixpath errors
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# This is to suppress the FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model (change path to your local best.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.25  # confidence threshold

print("Model loaded from:", 'best.pt')
print("Model classes:", model.names)

# Set folder path containing images
image_folder = 'golfball_images'  # <-- update this

# Loop through all image files
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)
        results = model(img_path)

        print(f"\n Results for {filename}:")
        results.print()