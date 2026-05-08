from __future__ import annotations

import base64
import cgi
import io
import json
import mimetypes
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MODELS_DIR = ROOT / "models"

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
AGE_GROUPS = {
    "0-2": "Young",
    "4-6": "Young",
    "8-12": "Young",
    "15-20": "Young",
    "25-32": "Middle-aged",
    "38-43": "Middle-aged",
    "48-53": "Old",
    "60-100": "Old",
}

MODEL_LABELS = {
    "caffe": "OpenCV DNN pre-trained Caffe age model",
}


class AgeDetector:
    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(str(cascade_path))
        self.yunet_model = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
        self.yunet = None
        if self.yunet_model.exists() and hasattr(cv2, "FaceDetectorYN_create"):
            self.yunet = cv2.FaceDetectorYN_create(str(self.yunet_model), "", (320, 320), 0.65, 0.3, 5000)
        self.age_net = self._load_age_net()
        self.total_tests = 0
        self._stats_lock = threading.Lock()

    def _load_age_net(self):
        prototxt = MODELS_DIR / "age_deploy.prototxt"
        model = MODELS_DIR / "age_net.caffemodel"
        if prototxt.exists() and model.exists():
            return cv2.dnn.readNet(str(model), str(prototxt))
        return None

    @property
    def mode(self) -> str:
        if self.age_net is not None:
            return MODEL_LABELS["caffe"]
        return "OpenCV demo mode"

    def analyze(self, image_bytes: bytes, model_preference: str = "caffe") -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = self._detect_faces(frame)
        quality = self._quality_report(frame)

        detections = []
        for idx, (x, y, w, h, detection_confidence) in enumerate(faces, start=1):
            face = frame[y : y + h, x : x + w]
            prediction = self._predict_age(face, "caffe")
            detections.append(
                {
                    "id": idx,
                    "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "detectionConfidence": round(float(detection_confidence), 3),
                    **prediction,
                }
            )
            self._draw_detection(frame, x, y, w, h, prediction)

        annotated = self._encode_image(frame)
        with self._stats_lock:
            self.total_tests += 1
            total_tests = self.total_tests

        return {
            "mode": self._mode_for_preference(model_preference),
            "modelPreference": model_preference,
            "totalTests": total_tests,
            "testedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "faceCount": len(faces),
            "resultCount": len(detections),
            "detections": detections,
            "annotatedImage": annotated,
            "quality": quality,
            "note": self._note(model_preference),
        }

    def _selected_methods(self, model_preference: str) -> list[str]:
        return ["caffe"] if self.age_net is not None else ["demo"]

    def _mode_for_preference(self, model_preference: str) -> str:
        if model_preference == "caffe" and self.age_net is not None:
            return MODEL_LABELS["caffe"]
        return self.mode

    def _quality_report(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        issues = []
        if brightness < 75:
            issues.append("Image is dark")
        elif brightness > 205:
            issues.append("Image is too bright")
        if blur < 80:
            issues.append("Image looks blurry")
        return {
            "brightness": round(brightness, 1),
            "sharpness": round(blur, 1),
            "status": "Good" if not issues else "Needs clearer image",
            "issues": issues,
        }

    def _detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        height, width = frame.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((width, height))
            _, faces = self.yunet.detect(frame)
            if faces is not None:
                boxes = []
                for face in faces:
                    x, y, w, h = face[:4]
                    confidence = face[-1]
                    pad_x = int(w * 0.18)
                    pad_y = int(h * 0.22)
                    x1 = max(0, int(x) - pad_x)
                    y1 = max(0, int(y) - pad_y)
                    x2 = min(width, int(x + w) + pad_x)
                    y2 = min(height, int(y + h) + pad_y)
                    boxes.append((x1, y1, x2 - x1, y2 - y1, float(confidence)))
                return boxes

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(55, 55))
        return [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in faces]

    def _predict_age(self, face: np.ndarray, model_preference: str = "caffe") -> dict:
        senior_cues = self._senior_cues(face)

        if self.age_net is not None:
            blob = cv2.dnn.blobFromImage(
                face,
                scalefactor=1.0,
                size=(227, 227),
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False,
            )
            self.age_net.setInput(blob)
            probabilities = self.age_net.forward()[0]
            bucket_index = int(probabilities.argmax())
            bucket = AGE_BUCKETS[bucket_index]
            warning = "This model predicts age ranges, not exact years."
            if AGE_GROUPS[bucket] != "Old" and senior_cues["isSeniorLikely"]:
                bucket = "60-100"
                warning = "Adjusted to old: senior visual cues conflict with the Caffe range prediction."
            return {
                "age": None,
                "ageBucket": bucket,
                "ageGroup": AGE_GROUPS[bucket],
                "confidence": max(round(float(probabilities[bucket_index]), 3), 0.62 if bucket == "60-100" else 0),
                "method": MODEL_LABELS["caffe"],
                "warning": warning,
            }

        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = gray_face.shape[:2]
        texture = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        brightness = float(np.mean(gray_face))
        face_area = h * w

        score = 0
        score += 1 if texture > 620 else 0
        score += 1 if texture > 1050 else 0
        score += 1 if brightness < 95 else 0
        score += 1 if face_area > 42000 else 0

        if score <= 1:
            group, bucket, confidence = "Young", "8-20", 0.54
        elif score == 2:
            group, bucket, confidence = "Middle-aged", "25-43", 0.50
        else:
            group, bucket, confidence = "Old", "48-100", 0.48

        return {
            "age": None,
            "ageBucket": bucket,
            "ageGroup": group,
            "confidence": confidence,
            "method": "Demo heuristic until Caffe model files are added",
            "warning": "Demo fallback only.",
        }

    def _senior_cues(self, face: np.ndarray) -> dict:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        height = face.shape[0]
        lower = hsv[int(height * 0.45) :, :]
        upper = hsv[: int(height * 0.45), :]
        lower_white = ((lower[:, :, 1] < 55) & (lower[:, :, 2] > 135)).mean() if lower.size else 0
        upper_white = ((upper[:, :, 1] < 50) & (upper[:, :, 2] > 145)).mean() if upper.size else 0
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = 0
        score += 2 if lower_white > 0.18 else 0
        score += 1 if upper_white > 0.16 else 0
        score += 1 if texture > 900 else 0
        return {
            "lowerWhiteRatio": round(float(lower_white), 3),
            "upperWhiteRatio": round(float(upper_white), 3),
            "texture": round(float(texture), 1),
            "score": score,
            "isSeniorLikely": score >= 2,
        }

    def _age_bucket(self, age: float) -> str:
        if age < 25:
            return "18-24"
        if age < 35:
            return "25-34"
        if age < 45:
            return "35-44"
        if age < 55:
            return "45-54"
        if age < 65:
            return "55-64"
        return "65+"

    def _age_group(self, age: float) -> str:
        if age < 25:
            return "Young"
        if age < 45:
            return "Middle-aged"
        return "Old"

    def _age_confidence(self, age: float) -> float:
        if age < 18 or age > 75:
            return 0.42
        return 0.78

    def _draw_detection(self, frame: np.ndarray, x: int, y: int, w: int, h: int, prediction: dict) -> None:
        color = (36, 172, 242)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        if prediction.get("age") is not None:
            label = f"{prediction['ageGroup']} - {prediction['age']} yrs"
        else:
            label = f"{prediction['ageGroup']} ({prediction['ageBucket']})"
        cv2.rectangle(frame, (x, max(0, y - 36)), (x + max(w, 260), y), color, -1)
        cv2.putText(frame, label, (x + 10, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 25, 32), 2)

    def _encode_image(self, frame: np.ndarray) -> str:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError("Could not encode annotated image")
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

    def _note(self, model_preference: str = "caffe") -> str:
        if model_preference == "caffe" and self.age_net is not None:
            return "Using Python, OpenCV DNN, and a pre-trained Caffe model to classify the detected face into young, middle-aged, or old."
        return "Caffe model files were not found, so predictions use a lightweight demo heuristic."


detector = AgeDetector()


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if route.startswith("/static/"):
            requested = (ROOT / route.lstrip("/")).resolve()
            if STATIC_DIR in requested.parents and requested.exists():
                self._send_file(requested)
                return
        self._json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            self._json({"error": "Not found"}, status=404)
            return
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            file_item = form["image"] if "image" in form else None
            if file_item is None or not file_item.file:
                self._json({"error": "Upload an image first."}, status=400)
                return
            result = detector.analyze(file_item.file.read(), "caffe")
            self._json(result)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        if content_type is None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), AppHandler)
    print("Age Detection Model app")
    print(f"Mode: {detector.mode}")
    print("Open http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
