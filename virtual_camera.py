import numpy as np
import cv2
import obsws_python as obs
from threading import Thread
import time
import os

class VirtualCamera:
    def __init__(self, width=640, height=480, fps=20):
        self.width = width
        self.height = height
        self.fps = fps
        self.client = None
        self.running = False
        self.thread = None
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.temp_file = "/tmp/animator_frame.jpg"

    def start(self):
        print("Attempting to connect to OBS WebSocket")
        try:
            self.client = obs.ReqClient(host="localhost", port=4455)
            self.client.set_current_program_scene("AnimatorScene")
            self.running = True
            self.thread = Thread(target=self._stream_frames)
            self.thread.start()
            print("Virtual camera started via OBS")
        except Exception as e:
            print(f"Error starting OBS virtual camera: {str(e)}")
            raise RuntimeError(f"Failed to start virtual camera: {str(e)}")

    def _stream_frames(self):
        while self.running:
            try:
                cv2.imwrite(self.temp_file, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
                time.sleep(1 / self.fps)
            except Exception as e:
                print(f"Error streaming frame: {str(e)}")
                break

    def send_frame(self, frame):
        if not self.running:
            return
        try:
            self.frame = cv2.resize(frame, (self.width, self.height))
        except Exception as e:
            print(f"Error sending frame: {str(e)}")

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            if self.client:
                try:
                    self.client.disconnect()
                except Exception as e:
                    print(f"Error disconnecting from OBS: {str(e)}")
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            print("Virtual camera stopped")