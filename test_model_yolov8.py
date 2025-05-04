from ultralytics import YOLO
import os
import warnings

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv8 model
model = YOLO('bestv8.pt')  # Replace with the actual path to your YOLOv8 model
print("YOLOv8 model loaded.")

# Set folder path containing images
image_folder = 'golfball_images'  # Update this if needed

# Loop through all image files
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)
        results = model(img_path)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"Detected class {cls_id} with confidence {conf:.3f} at {xyxy}")

        print(f"\nResults for {filename}:")
        results[0].show()