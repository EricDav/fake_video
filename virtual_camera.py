import pyvirtualcam
import numpy as np
import cv2

class VirtualCamera:
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None

    def start(self):
        """Start the virtual camera."""
        self.cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps)
        print(f"Virtual camera started: {self.cam.device}")

    def send_frame(self, frame):
        """Send a frame to the virtual camera."""
        if self.cam is None:
            return
        # Resize and convert to RGB
        frame = cv2.resize(frame, (self.width, self.height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cam.send(frame_rgb)
        self.cam.sleep_until_next_frame()

    def stop(self):
        """Stop the virtual camera."""
        if self.cam:
            self.cam = None
            print("Virtual camera stopped")