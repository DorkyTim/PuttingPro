
import cv2
import numpy as np
import time
import math
import pathlib
from ultralytics import YOLO
import warnings
from scipy.optimize import linear_sum_assignment

# Ensure compatibility on Windows for paths
pathlib.PosixPath = pathlib.WindowsPath
warnings.filterwarnings("ignore", category=FutureWarning)

class BallTracker:
    def __init__(self, yolo_path='bestv8.pt', confidence=0.3):
        self.model = YOLO(yolo_path)
        self.model.conf = confidence

        self.next_ball_id = 0
        self.tracked_balls = {}
        self.fade_duration = 1.0
        self.dist_check = 80  # Increased from 50

    def _create_kalman_filter(self, init_x, init_y):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.statePre = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
        kalman.statePost = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
        return kalman

    def detect_ball(self, frame):
        frame = cv2.resize(frame, (854, 480))
        current_time = time.time()

        for ball_id, info in self.tracked_balls.items():
            prediction = info['kalman'].predict()
            self.tracked_balls[ball_id]['prediction'] = (int(prediction[0]), int(prediction[1]))

        results = self.model.predict(frame, verbose=False)[0]
        detections = []
        if results and results.boxes is not None:
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                radius = int(max(x2 - x1, y2 - y1) / 2)
                detections.append({'center': center, 'radius': radius})

        matched_ids = set()
        pred_ids = list(self.tracked_balls.keys())
        predictions = [self.tracked_balls[bid]['prediction'] for bid in pred_ids]

        if predictions and detections:
            cost_matrix = np.zeros((len(detections), len(predictions)), dtype=np.float32)
            for i, det in enumerate(detections):
                for j, pred in enumerate(predictions):
                    dist = np.linalg.norm(np.array(det['center']) - np.array(pred))
                    cost_matrix[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_preds = set()

            for det_idx, pred_idx in zip(row_ind, col_ind):
                dist = cost_matrix[det_idx, pred_idx]
                ball_id = pred_ids[pred_idx]
                if dist < self.dist_check and current_time - self.tracked_balls[ball_id]['last_seen'] < self.fade_duration:
                    det = detections[det_idx]
                    kalman = self.tracked_balls[ball_id]['kalman']
                    prev_pos = self.tracked_balls[ball_id]['pos']
                    dt = current_time - self.tracked_balls[ball_id]['last_seen']
                    dx = det['center'][0] - prev_pos[0]
                    dy = det['center'][1] - prev_pos[1]
                    if dt > 0 and (abs(dx) > 1 or abs(dy) > 1):
                        vx = dx / dt
                        vy = dy / dt
                        alpha = 0.4
                        kalman.statePost[2] = alpha * vx + (1 - alpha) * kalman.statePost[2]
                        kalman.statePost[3] = alpha * vy + (1 - alpha) * kalman.statePost[3]
                    kalman.correct(np.array([[np.float32(det['center'][0])], [np.float32(det['center'][1])]]))

                    self.tracked_balls[ball_id].update({
                        "pos": det['center'],
                        "radius": det['radius'],
                        "last_seen": current_time,
                        "prediction": det['center']
                    })
                    matched_ids.add(ball_id)
                    assigned_preds.add(pred_idx)

            for i, det in enumerate(detections):
                if i not in row_ind:
                    matched_id = self.next_ball_id
                    self.next_ball_id += 1
                    self.tracked_balls[matched_id] = {
                        "kalman": self._create_kalman_filter(det['center'][0], det['center'][1]),
                        "pos": det['center'],
                        "last_seen": current_time,
                        "radius": det['radius'],
                        "prediction": det['center'],
                        "confidence": 1.0
                    }
                    matched_ids.add(matched_id)
        elif detections:
            for det in detections:
                matched_id = self.next_ball_id
                self.next_ball_id += 1
                self.tracked_balls[matched_id] = {
                    "kalman": self._create_kalman_filter(det['center'][0], det['center'][1]),
                    "pos": det['center'],
                    "last_seen": current_time,
                    "radius": det['radius'],
                    "prediction": det['center'],
                    "confidence": 1.0
                }
                matched_ids.add(matched_id)

        for ball_id in list(self.tracked_balls.keys()):
            if ball_id not in matched_ids:
                self.tracked_balls[ball_id].setdefault("confidence", 1.0)
                self.tracked_balls[ball_id]["confidence"] *= 0.9
                if self.tracked_balls[ball_id]["confidence"] < 0.3:
                    del self.tracked_balls[ball_id]
                    continue

        return {
            'frame_size': (frame.shape[1], frame.shape[0]),
            'balls': {
                ball_id: {
                    'pos': info['pos'],
                    'radius': info['radius'],
                    'prediction': info['prediction'],
                    'last_seen': info['last_seen']
                }
                for ball_id, info in self.tracked_balls.items()
                if current_time - info['last_seen'] < self.fade_duration
            }
        }
