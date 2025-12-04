import time
import cv2
import numpy as np
import os
from datetime import datetime
import mss
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/train10/weights/best.pt"  # Path to your current best model
SAVE_DIR = "datasets/new_images"
CONF_RANGE = (0.3, 0.8)  # Save images where detection confidence is within this range
SAVE_INTERVAL = 2.0      # Minimum seconds between saves to avoid duplicates
MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # Load model
    try:
        model = YOLO(MODEL_PATH)
        print(f"[INFO] Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    sct = mss.mss()
    print(f"[INFO] Data collection started. Saving to {SAVE_DIR}...")
    print(f"[INFO] capturing images with detections between {CONF_RANGE[0]} and {CONF_RANGE[1]} confidence.")
    print("[INFO] Press 'q' to quit.")

    last_save_time = 0

    while True:
        # Capture screen
        img = np.array(sct.grab(MONITOR))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Run inference
        results = model(frame, verbose=False)[0]
        
        should_save = False
        max_conf = 0.0

        # Check detections
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
            
            # Condition to save: meaningful detection but not 100% sure
            if CONF_RANGE[0] <= conf <= CONF_RANGE[1]:
                should_save = True

        # Draw boxes for visualization (optional, so you see what's happening)
        annotated_frame = results.plot()

        # Save logic
        current_time = time.time()
        if should_save and (current_time - last_save_time > SAVE_INTERVAL):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{SAVE_DIR}/collect_{timestamp}.jpg"
            cv2.imwrite(filename, frame) # Save the clean frame, not annotated
            print(f"[SAVED] {filename} (Max Conf: {max_conf:.2f})")
            last_save_time = current_time

        # Display
        cv2.putText(annotated_frame, f"Collecting... (Saved: {len(os.listdir(SAVE_DIR))})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Data Collector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
