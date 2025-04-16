from collections import deque
import cv2

class TrailManager:
    def __init__(self, max_length=1000):
        self.enabled = False
        self.max_length = max_length
        self.trails = {}  # ball_id: deque of points

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.reset()  # Clear trails when turning off

    def add_point(self, ball_id, point):
        if not self.enabled:
            return
        if ball_id not in self.trails:
            self.trails[ball_id] = deque(maxlen=self.max_length)
        self.trails[ball_id].append(point)

    def draw(self, frame):
        if not self.enabled:
            return frame
        for trail in self.trails.values():
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 100, 255), 2)
        return frame

    def reset(self):
        self.trails.clear()