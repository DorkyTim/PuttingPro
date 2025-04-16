import cv2
from BallTracker import BallTracker
from TrailManager import TrailManager

trail_manager = TrailManager()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 1 or 2 if 0 doesn't work

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60.0)
#cap.set(cv2.CAP_PROP_SETTINGS, 1)

tracker = BallTracker(yolo_path='best.pt', confidence=0.6, trail_manager=trail_manager)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output = tracker.detect_ball(frame)
    cv2.imshow("Tracking", output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        trail_manager.toggle()

cap.release()
cv2.destroyAllWindows()