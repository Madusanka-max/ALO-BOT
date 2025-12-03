import time
import csv
import cv2
import numpy as np
from inference import get_model
import supervision as sv
from datetime import datetime
import mss
import mss.tools
import pandas as pd

# ───────────────────────────────
# CONFIG
# ───────────────────────────────
MODEL_ID = "albion_detector-l4o1a-hzqc5/1"
CONF_THRESHOLD = 0.53  # adjust based on CSV reliability analysis
CSV_PATH = "detections_log.csv"

# screen region to capture (adjust as needed)
MONITOR = {"top": 100, "left": 100, "width": 1280, "height": 720}

# Model load
model = get_model(MODEL_ID)

# Supervision annotators
bounding_box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default())
label_annotator = sv.LabelAnnotator(text_color=sv.Color.white())

# ───────────────────────────────
# CSV Logging Setup
# ───────────────────────────────
def init_csv():
    try:
        pd.read_csv(CSV_PATH)
        print("[CSV] Existing log found.")
    except:
        print("[CSV] Creating new log file.")
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "class", "confidence", "x", "y", "width", "height"])


def log_detection(cls, conf, box):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            cls,
            round(conf, 3),
            box[0],
            box[1],
            box[2],
            box[3]
        ])

# ───────────────────────────────
# MAIN LOOP
# ───────────────────────────────
def main():
    init_csv()
    sct = mss.mss()

    while True:
        # Capture screenshot
        frame = np.array(sct.grab(MONITOR))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Model inference
        results = model.infer(frame)[0]

        detections = sv.Detections.from_inference(results)
        detections = detections[detections.confidence > CONF_THRESHOLD]

        # Log each detection
        for i, (box, cls, conf) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
            log_detection(detections.data['class_name'][i], float(conf), box)

        # Annotate
        labels = [
            f"{results.classes[i]} {conf:.2f}"
            for i, conf in zip(detections.class_id, detections.confidence)
        ]
        frame = bounding_box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, labels)

        # Show counts overlay
        class_counts = {}
        for cls_name in results.classes:
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        y = 30
        for cls_name, count in class_counts.items():
            cv2.putText(frame, f"{cls_name}: {count}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 22

        cv2.imshow("Albion Resource Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
