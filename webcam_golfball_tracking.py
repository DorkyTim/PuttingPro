import torch
import cv2
import time

# This is to run linux trained model on windows to avoid posixpath errors
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# This is to suppress the FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

confidence = 0.25
show_fps = 0

# Load YOLOv5 model (change path to your local best.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = confidence  # confidence threshold

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 1 or 2 if 0 doesn't work

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60.0)

print("FPS reported:", cap.get(cv2.CAP_PROP_FPS))

if(show_fps):
  prev_time = time.time()

while cap.isOpened():
  
  if(show_fps):
    start_time = time.time()
  
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.resize(frame, (854, 480))
  
  # Run detection
  results = model(frame)
  #print("Inference done.")
  
  #detections = results.pandas().xyxy[0] 
  
  #print(detections)
  
  # Draw results
  annotated_frame = results.render()[0].copy()  # render returns list of frames
  
  # === FPS Calculation ===
  if(show_fps):
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

  
  # # Draw Circles
  # for _, row in detections.iterrows():
  #   if row['confidence'] < confidence:
  #     continue  # skip low confidence detections
    
  #   # Get bounding box
  #   x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

  #   # Calculate center and radius
  #   center_x = (x1 + x2) // 2
  #   center_y = (y1 + y2) // 2
  #   radius = int(max(x2 - x1, y2 - y1) / 2)

  #   # Draw circle in orange
  #   cv2.circle(annotated_frame, (center_x, center_y), radius, (0, 165, 255), 2)
    
  #   label = f"{row['name']} {row['confidence']:.2f}"
  #   cv2.putText(annotated_frame, label, (x1, y1 - 10),
  #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        

  cv2.imshow("Golf Ball Detection", annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()