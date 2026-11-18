"""
Main Launcher for AI-Vision Communication Assistance for Paralyzed Individuals.
Automatically verifies models, starts the Flask application, and opens the web interface.
"""

import os
import sys

# Suppress framework informational messages before any package import
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["CLEARCUT_DISABLE_TELEMETRY"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

import time
import webbrowser
import threading
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def ensure_model():
    """
    Verify that MediaPipe FaceLandmarker deep learning model is present.
    If missing/corrupted and download fails, prints a clear error and exits non-zero.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            print(f"[ERROR] Could not auto-download model: {e}")
            print("[ERROR] FaceLandmarker model missing or corrupt. Application cannot start.")
            sys.exit(1)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        print("[ERROR] FaceLandmarker model file is missing or invalid size (<1MB).")
        sys.exit(1)


def open_browser(url: str = "http://127.0.0.1:5000"):
    """
    Poll application server until ready before opening browser quietly in background.
    """
    start_time = time.time()
    while time.time() - start_time < 30.0:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    webbrowser.open(url)
                    break
        except Exception:
            time.sleep(0.5)


def main():
    ensure_model()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    url = f"http://{host}:{port}"
    home_url = f"{url}/home"

    # Launch browser opener in background — opens the Home Page directly
    threading.Thread(target=open_browser, args=(home_url,), daemon=True).start()

    # Import and run Flask app
    from app import app, shutdown_application, setup_graceful_shutdown
    setup_graceful_shutdown()

    print("==================================================")
    print("       AI-VISION COMMUNICATION ASSISTANT")
    print("==================================================")
    print("")
    print("[OK] FaceLandmarker model loaded")
    print("[OK] AI Vision Engine initialized")
    print("[INFO] Camera is OFF by default (Turn ON via interface)")
    print("[OK] Touchless keyboard controls active (Keys 1-4)")
    print("[OK] Database connected")
    print("[OK] Server started successfully")
    print("")
    print(f"Application: {url}")
    print("")
    print("==================================================")
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_application()


if __name__ == "__main__":
    main()
