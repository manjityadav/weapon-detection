from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
import uuid
import winsound
import threading
from ultralytics import YOLO  # YOLOv8

app = Flask(__name__)

# ----------------------------
# Folders
# ----------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# Load YOLOv8 Model (pretrained)
# ----------------------------
model = YOLO('yolov8n.pt')  # small, fast model for testing

alarm_lock = threading.Lock()

# ----------------------------
# Alarm Function (Thread Safe)
# ----------------------------
def play_alarm():
    with alarm_lock:
        winsound.Beep(1000, 700)

# ----------------------------
# Detection Function
# ----------------------------
def detect_frame(img):
    results = model.predict(img, verbose=False)[0]  # Run inference
    weapon_detected = False

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Here we assume every detection could be a weapon
            if conf > 0.5:
                weapon_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    img,
                    f"Weapon {round(conf*100,2)}%",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    if weapon_detected:
        threading.Thread(target=play_alarm).start()

    return img, weapon_detected

# ----------------------------
# Home Route
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html", result_image=None, detected=False)

# ----------------------------
# Stop Detection Route
# ----------------------------
@app.route("/stop")
def stop_detection():
    return redirect(url_for("home"))

# ----------------------------
# Image Detection
# ----------------------------
@app.route("/detect_image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("home"))

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return redirect(url_for("home"))

    result_img, detected = detect_frame(img)

    output_path = os.path.join(UPLOAD_FOLDER, "result_" + filename)
    cv2.imwrite(output_path, result_img)

    return render_template("index.html", result_image=output_path, detected=detected)

# ----------------------------
# Video Detection
# ----------------------------
@app.route("/detect_video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("home"))

    filename = str(uuid.uuid4()) + ".mp4"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return redirect(url_for("home"))

    first_detected_frame_path = None
    weapon_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, detected = detect_frame(frame)
        if detected:
            weapon_found = True
            first_detected_frame_path = os.path.join(
                UPLOAD_FOLDER,
                "detected_" + str(uuid.uuid4()) + ".jpg"
            )
            cv2.imwrite(first_detected_frame_path, result_frame)
            break  # stop after first detection

    cap.release()

    return render_template("index.html", result_image=first_detected_frame_path, detected=weapon_found)

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)