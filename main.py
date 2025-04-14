import sys
import cv2
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
                self.timer.start(33)  # ~30fps
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
            return

        landmarks = self.tracker.get_landmarks(frame)
        if self.animator:
            output_frame = self.animator.animate(landmarks)
        else:
            output_frame = frame

        self.virtual_cam.send_frame(output_frame)

        # Preview (optional, for debugging)
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