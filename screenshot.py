import os
import time
from datetime import datetime
from PIL import ImageGrab

def take_screenshot():
    folder = "ss"
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(folder, f"{timestamp}.png")

    screenshot = ImageGrab.grab()
    screenshot.save(filepath)

    print(f"[+] Screenshot saved: {filepath}")

def main():
    print("[*] Auto screenshot started (every 5s). Press CTRL + C to stop.")

    while True:
        take_screenshot()
        time.sleep(5)  # wait 5 seconds

if __name__ == "__main__":
    main()
