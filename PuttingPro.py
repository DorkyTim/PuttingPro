import cv2
import numpy as np
from BallTracker import BallTracker
from TrailManager import TrailManager

# Set up camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60.0)

# Initialize tracker and trail manager
tracker = BallTracker(yolo_path='bestv8.pt', confidence=0.3)
trail_manager = TrailManager()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for display
    resized_frame = cv2.resize(frame, (854, 480))

    # Detect using original frame
    ball_data = tracker.detect_ball(frame)

    # Create blank canvas and copy for display
    overlay = np.zeros((resized_frame.shape[0], resized_frame.shape[1], 3), dtype=np.uint8)
    display = resized_frame.copy()

    # Scale factors for coordinate mapping
    scale_x = resized_frame.shape[1] / ball_data['frame_size'][0]
    scale_y = resized_frame.shape[0] / ball_data['frame_size'][1]

    # Draw balls and update trail
    for ball_id, info in ball_data['balls'].items():
        pos = info['prediction']  # Use Kalman-filtered center for smooth display
        #pos = info['pos'] # Use raw position - supposedly more jittery
        radius = info['radius']
    
        scaled_pos = (int(pos[0] * scale_x), int(pos[1] * scale_y))
        scaled_radius = int(radius * (scale_x + scale_y) / 2)

        cv2.circle(display, scaled_pos, scaled_radius, (0, 255, 0), 2)
        cv2.putText(display, f"ID {ball_id}", (scaled_pos[0], scaled_pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.circle(overlay, scaled_pos, scaled_radius, (0, 255, 0), 2)
        cv2.putText(overlay, f"ID {ball_id}", (scaled_pos[0], scaled_pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        trail_manager.add_point(ball_id, scaled_pos)

    # Draw trails if enabled
    overlay = trail_manager.draw(overlay)

    # Show both windows
    cv2.imshow("Video Feed", display)
    cv2.imshow("Overlay Only", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        trail_manager.toggle()

cap.release()
cv2.destroyAllWindows()