# ALO-BOT (Albion Online Bot)

ALO-BOT is a computer vision-based automation tool designed for **Albion Online**. It utilizes YOLO (You Only Look Once) object detection models to identify in-game resources (such as ore, fiber, logs, etc.) and automate the harvesting process.

> **⚠️ WARNING:** Using automation tools, bots, or macros in Albion Online is against the [Terms of Service](https://albiononline.com/terms_and_conditions). This project is for educational and research purposes only. Use it at your own risk. The author is not responsible for any bans or penalties incurred.

## 🚀 Features

* **Automated Harvesting (`bot.py`):** Detects resources on the screen, moves the mouse to them, clicks to harvest, and waits for the action to complete.
* **Resource Detection & Logging (`detect_resources.py`):** Scans the screen for resources, highlights them with bounding boxes, and logs detection data (timestamp, class, confidence, location) to a CSV file.
* **Real-time Screen Detection (`screen_detect.py`):** A lightweight visualizer that shows what the bot sees in real-time, including bounding boxes, class labels, and confidence scores.
* **Custom Trained Model:** Uses a YOLO model trained on a specific dataset of Albion Online resources.

## 🛠️ Prerequisites

*   Python 3.8+
*   A CUDA-capable GPU is recommended for real-time performance.

## 📦 Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd ALO-BOT
    ```

2.  **Set up the Virtual Environment:**
    This project includes a `venv` directory. You should activate it before running the scripts.

    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **Mac/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *(Note: You may need to install PyTorch separately depending on your system configuration to enable GPU support.)*

## 💻 Usage

### 1. Automated Harvesting Bot

This script actively controls your mouse to harvest resources.

```bash
python bot.py
```

* **Controls:** Press `Q` to quit.
* **Configuration:** Edit `bot.py` to change `TARGET_CLASSES` (e.g., `["ore", "fiber"]`), `CONF_THRESHOLD`, or `HARVEST_TIMEOUT`.

### 2. Resource Logger

This script detects resources and saves the data to `detections_log.csv`. Useful for analyzing spawn rates or testing model accuracy.

```bash
python detect_resources.py
```

* **Controls:** Press `Esc` to quit.

### 3. Visual Detector

A simple visualizer to see detection performance without any automation or logging.

```bash
python screen_detect.py
```

* **Controls:** Press `Q` to quit.

## 📂 Project Structure

* `bot.py`: Main automation script.
* `detect_resources.py`: Detection and logging script.
* `screen_detect.py`: Real-time detection visualizer.
* `data.yaml`: Dataset configuration file.
* `runs/`: Directory containing trained model weights (e.g., `best.pt`).
* `train/`, `test/`, `valid/`: Dataset directories.

## 🤖 Model Classes

The model is trained to detect the following classes:

* fiber
* hide
* mob
* ore
* player
* red
* silver
* stone
* tree
* wisp caged

## 📄 License

This project uses a dataset exported from Roboflow. See `README.roboflow.txt` for dataset license details (CC BY 4.0).
