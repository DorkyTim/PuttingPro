import cv2
import numpy as np
import time
import math
from collections import deque

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FPS, 60.0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Kalman filter setup
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Deque of (center, speed) tuples
trajectory = deque(maxlen=50)
prev_center = None
prev_time = time.time()

def get_color_for_speed(speed):
    min_speed = 5
    max_speed = 100
    speed = max(min_speed, min(speed, max_speed))
    ratio = (speed - min_speed) / (max_speed - min_speed)
    if ratio < 0.5:
        blue = int(255 * (1 - 2 * ratio))
        green = int(255 * (2 * ratio))
        red = 0
    else:
        blue = 0
        green = int(255 * (2 * (1 - ratio)))
        red = int(255 * (2 * (ratio - 0.5)))
    return (blue, green, red)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (854, 480))
    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=100, param2=50, minRadius=20, maxRadius=50
    )

    prediction = kalman.predict()
    predicted_pt = (int(prediction[0].item()), int(prediction[1].item()))

    current_time = time.time()

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        largest = max(circles, key=lambda c: c[2])
        x, y, r = largest
        center = (x, y)
        last_radius = r 
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        kalman.correct(measurement)

        # Estimate speed
        speed = 0
        if prev_center is not None:
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]
            dist = math.hypot(dx, dy)
            time_diff = current_time - prev_time
            if 0 < time_diff < 0.5:
                speed = dist / time_diff

        trajectory.appendleft((center, speed))
        prev_center = center
        prev_time = current_time

        cv2.circle(output, center, r, (0, 255, 0), 2)
        cv2.circle(output, center, 2, (0, 0, 255), 3)
        cv2.putText(output, f"Speed: {speed:.2f} px/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, get_color_for_speed(speed), 2)
    else:
        if prev_center is not None and 'last_radius' in locals():
            trajectory.appendleft((predicted_pt, 0))
            cv2.circle(output, predicted_pt, last_radius, (0, 255, 255), 2)
            cv2.circle(output, predicted_pt, 2, (0, 255, 255), -1)
            cv2.putText(output, "Predicted", (predicted_pt[0] + 10, predicted_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            trajectory.appendleft((predicted_pt, 0))

    for i in range(1, len(trajectory)):
        (pt1, spd1) = trajectory[i - 1]
        (pt2, spd2) = trajectory[i]
        if pt1 is None or pt2 is None:
            continue
        avg_speed = (spd1 + spd2) / 2
        color = get_color_for_speed(avg_speed)
        thickness = int(np.sqrt(trajectory.maxlen / float(i + 1)) * 2.5)
        cv2.line(output, pt1, pt2, color, thickness)

    cv2.imshow("Kalman + Speed Trail Tracker", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()