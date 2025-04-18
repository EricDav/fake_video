import pyvirtualcam
import traceback
import sys

print("Python version:", sys.version)
print("PyVirtualCam version:", pyvirtualcam.__version__)
print("PyVirtualCam path:", pyvirtualcam.__file__)
try:
    backends = pyvirtualcam.available_backends()
    print("Available backends:", backends)
    print("Attempting to create camera with native backend")
    cam = pyvirtualcam.Camera(width=640, height=480, fps=20, backend="native")
    print(f"Camera started: {cam.device}")
    input("Press Enter to exit")
    cam.close()
except Exception as e:
    print("Error details:")
    traceback.print_exc()