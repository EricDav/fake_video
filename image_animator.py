import cv2
import numpy as np
from facial_tracker import FacialTracker  # Add this import

class ImageAnimator:
    def __init__(self, image_path):
        self.base_image = cv2.imread(image_path)
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Failed to load image")
        self.h, self.w = self.image.shape[:2]
        self.neutral_landmarks = None
        self.tracker = FacialTracker()

    def get_base_image(self):
        """Return the base image without animation."""
        return self.base_image.copy() if self.base_image is not None else None

    def set_neutral_landmarks(self):
        """Process the static image to get its landmarks (neutral state)."""
        results = self.tracker.get_landmarks(self.image)
        if results:
            self.neutral_landmarks = np.array(results)
        else:
            raise ValueError("No face detected in uploaded image")

    def animate(self, webcam_landmarks):
        """Warp the image based on webcam landmarks."""
        if self.neutral_landmarks is None:
            self.set_neutral_landmarks()

        if webcam_landmarks is None:
            return self.image.copy()

        src_pts = self.neutral_landmarks[:, :2] * [self.w, self.h]
        dst_pts = np.array(webcam_landmarks)[:, :2] * [self.w, self.h]

        try:
            matrix = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)[0]
            warped = cv2.warpAffine(self.image, matrix, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
        except:
            warped = self.image.copy()

        return warped