from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
import time

# Path to your trained weights
MODEL_PATH = "runs/detect/train10/weights/best.pt"

# Screen region to capture (monitor 1 full screen)
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

def main():
    model = YOLO(MODEL_PATH)
    sct = mss()

    print("[INFO] Detection started... Press Q to quit.")

    prev_time = 0

    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = model(img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

                print(f"[DETECT] {label}: {conf:.2f}")

        # FPS
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if prev_time > 0 else 0
        prev_time = cur_time
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Albion Screen Detection", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
