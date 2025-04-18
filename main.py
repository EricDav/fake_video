import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from facial_tracker import FacialTracker
from image_animator import ImageAnimator
from virtual_camera import VirtualCamera

class VideoCallAnimator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Call Animator")
        self.setGeometry(100, 100, 400, 300)

        # Initialize components
        self.tracker = FacialTracker()
        self.animator = None
        self.virtual_cam = VirtualCamera()
        self.cap = cv2.VideoCapture(0)
        self.image_path = None
        self.running = False
        self.last_landmarks = None
        self.movement_threshold = 0.005  # Adjust based on testing
        self.is_moving = False

        # GUI setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_btn)

        self.start_btn = QPushButton("Start Virtual Camera")
        self.start_btn.clicked.connect(self.start_stop)
        self.start_btn.setEnabled(False)
        self.layout.addWidget(self.start_btn)

        self.status_label = QLabel("No image loaded")
        self.layout.addWidget(self.status_label)

        # Timer for video processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def upload_image(self):
        """Handle image upload."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            try:
                self.image_path = file_name
                self.animator = ImageAnimator(file_name)
                self.status_label.setText(f"Image loaded: {file_name}")
                self.start_btn.setEnabled(True)
            except Exception as e:
                self.status_label.setText(f"Error: {str(e)}")
                self.start_btn.setEnabled(False)

    def start_stop(self):
        """Toggle virtual camera on/off."""
        if not self.running:
            try:
                self.virtual_cam.start()
                self.timer.start(50)  # ~20fps
                self.running = True
                self.start_btn.setText("Stop Virtual Camera")
                self.upload_btn.setEnabled(False)
            except Exception as e:
                self.status_label.setText(f"Error starting camera: {str(e)}")
        else:
            self.timer.stop()
            self.virtual_cam.stop()
            self.running = False
            self.start_btn.setText("Start Virtual Camera")
            self.upload_btn.setEnabled(True)
            self.status_label.setText("Virtual camera stopped")

    def process_frame(self):
        """Process webcam frame and send to virtual camera."""
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Camera error")
            return

        landmarks = self.tracker.get_landmarks(frame)
        output_frame = frame

        if landmarks:
            # Extract key landmarks (e.g., mouth) for movement detection
            mouth_top = landmarks[13] if len(landmarks) > 13 else None
            mouth_bottom = landmarks[14] if len(landmarks) > 14 else None
            if mouth_top and mouth_bottom:
                current_landmarks = np.array([mouth_top[0], mouth_top[1], mouth_bottom[0], mouth_bottom[1]])
                mouth_open = abs(mouth_top[1] - mouth_bottom[1])
                print(f"Mouth openness: {mouth_open}")

                # Detect movement
                if self.last_landmarks is not None:
                    movement = np.linalg.norm(current_landmarks - self.last_landmarks)
                    self.is_moving = movement > self.movement_threshold
                    print(f"Movement: {movement}, Is moving: {self.is_moving}")
                self.last_landmarks = current_landmarks

            # Animate only if moving
            if self.animator:
                if self.is_moving:
                    output_frame = self.animator.animate(landmarks)
                    if output_frame is None:
                        print("Animation failed: output_frame is None")
                        output_frame = frame
                else:
                    # Idle state: Use the base image without animation
                    output_frame = self.animator.get_base_image()  # Assume ImageAnimator has a method to return the base image
                    if output_frame is None:
                        print("Base image not available")
                        output_frame = frame
        else:
            print("No landmarks detected")
            if self.animator:
                output_frame = self.animator.get_base_image()  # Idle state
                if output_frame is None:
                    output_frame = frame

        self.virtual_cam.send_frame(output_frame)

        # Preview
        preview = cv2.resize(output_frame, (320, 240))
        cv2.imshow("Preview", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.start_stop()

    def closeEvent(self, event):
        """Clean up resources."""
        self.timer.stop()
        self.virtual_cam.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        self.tracker.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoCallAnimator()
    window.show()
    sys.exit(app.exec_())