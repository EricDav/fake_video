import cv2
import mediapipe as mp
import numpy as np

class FacialTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def get_landmarks(self, frame):
        """Process a frame and return facial landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        landmarks = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
        return landmarks

    def draw_landmarks(self, frame, landmarks):
        """Draw landmarks on the frame for debugging."""
        if landmarks:
            h, w = frame.shape[:2]
            for lm in landmarks:
                x, y = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        return frame

    def close(self):
        """Release resources."""
        self.face_mesh.close()