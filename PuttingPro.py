from BallTracker import BallTracker
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 1 or 2 if 0 doesn't work

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60.0)

tracker = BallTracker(yolo_path='best.pt', confidence=0.6)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output = tracker.detect_ball(frame)
    cv2.imshow("Tracking", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()