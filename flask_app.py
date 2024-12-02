import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
import os
import threading
import queue
import requests
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for now

# Constants
API_KEY = os.getenv("ROBOFLOW-INFERENCE-API-KEY")
GAZE_DETECTION_URL = f"http://{os.getenv('CONTAINER-PORT')}:9001/gaze/gaze_detection?api_key=" + API_KEY
MAX_YAW_LEFT = -0.5
MAX_YAW_RIGHT = 0.5
MAX_PITCH_UP = -0.5
MAX_PITCH_DOWN = 0.5

# Models
face_detector = FaceDetector("assets/face_detector.onnx")
mark_detector = MarkDetector("assets/face_landmarks.onnx")
pose_estimator = None

# Async Queues for Processing
input_queue = queue.Queue(maxsize=1)
output_queue = queue.Queue(maxsize=1)


def check_for_distraction(gaze):
    """Check if gaze is out of range (distraction)."""
    yaw = gaze["yaw"]
    pitch = gaze["pitch"]
    return yaw < MAX_YAW_LEFT or yaw > MAX_YAW_RIGHT or pitch < MAX_PITCH_UP or pitch > MAX_PITCH_DOWN


def detect_gazes(frame):
    """Detect gazes using the gaze detection API."""
    if frame is None or frame.size == 0:
        return []

    try:
        _, img_encode = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(img_encode)
        resp = requests.post(
            GAZE_DETECTION_URL,
            json={"api_key": API_KEY, "image": {"type": "base64", "value": img_base64.decode("utf-8")}},
        )
        resp.raise_for_status()
        return resp.json()[0]["predictions"]
    except Exception as e:
        print(f"Error in detect_gazes: {e}")
        return []


def async_frame_processing():
    """Background thread for frame processing."""
    global pose_estimator
    while True:
        frame = input_queue.get()
        if frame is None:  # Stop the thread
            break

        if pose_estimator is None:
            pose_estimator = PoseEstimator(frame.shape[1], frame.shape[0])

        # Run face detection
        faces, _ = face_detector.detect(frame, 0.7)
        combined_distraction = False

        if len(faces) > 0:
            face = refine(faces, frame.shape[1], frame.shape[0], 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]
            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Pose estimation
            head_distraction, _ = pose_estimator.detect_distraction(marks)

            # Gaze detection
            gazes = detect_gazes(frame)
            eye_distraction = False
            if gazes:
                for gaze in gazes:
                    eye_distraction = check_for_distraction(gaze)

            combined_distraction = head_distraction or eye_distraction

        output_queue.put(combined_distraction)


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """Process a single video frame."""
    try:
        # Decode the base64-encoded frame
        data = request.json["frame"]
        img = np.frombuffer(base64.b64decode(data), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Send the frame to the input queue
        if input_queue.empty():
            input_queue.put(frame)

        # Retrieve the distraction status from the output queue
        combined_distraction = output_queue.get() if not output_queue.empty() else False
        status = "Distracted" if combined_distraction else "Focused"
        return jsonify({"status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Start the async thread
thread = threading.Thread(target=async_frame_processing)
thread.daemon = True
thread.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
