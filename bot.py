from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
import time
import pyautogui

# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/train10/weights/best.pt"

monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

CONF_THRESHOLD = 0.70       # minimum confidence allowed
HARVEST_TIMEOUT = 3        # max seconds to wait for resource to vanish
# TARGET_CLASSES = ["fiber", "ore", "hide", "stone", "tree"]  # you can add/remove
TARGET_CLASSES = ["ore"]  # you can add/remove

# =========================

def click_center(box):
    """Click the center of a bounding box."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    print(f"[CLICK] Harvesting at {cx},{cy}")
    pyautogui.moveTo(cx, cy)
    pyautogui.click()


def wait_until_gone(model, target_cls):
    """Wait until the harvested object disappears."""
    sct = mss()
    start = time.time()

    print("[INFO] Waiting for harvesting animation to finish...")

    while time.time() - start < HARVEST_TIMEOUT:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model(frame, verbose=False)[0]

        seen = False
        for box in results.boxes:
            if int(box.cls[0]) == target_cls:
                seen = True

        if not seen:
            print("[SUCCESS] Node gone. Harvest complete.")
            return

        time.sleep(0.3)

    print("[WARNING] Timeout — moving on.")


def main():
    model = YOLO(MODEL_PATH)
    sct = mss()

    print("[INFO] Detection started... Press Q to quit.")

    prev_time = 0

    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = model(img)[0]

        for box in results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Click + wait if resource
            if conf > CONF_THRESHOLD and label in TARGET_CLASSES:
                click_center(box)
                wait_until_gone(model, cls)

        # FPS
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if prev_time else 0
        prev_time = cur_time
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Albion Screen Detection", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
