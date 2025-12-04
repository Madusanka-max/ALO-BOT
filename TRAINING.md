# 🏋️ Continuous Training Guide

This guide explains how to improve your ALO-BOT model over time by collecting new data, labeling it, and retraining. This process allows you to increase the `CONF_THRESHOLD` (e.g., to 0.8) while maintaining high detection accuracy.

## 1. Collect Data (`collect_data.py`)

Run the data collection script to capture "hard" examples (images where the model is uncertain).

```bash
python collect_data.py
```

*   **What it does:** It captures screenshots where the model's confidence is between 0.3 and 0.8.
*   **Output:** Images are saved to `datasets/new_images`.
*   **Action:** Play the game normally. The script will automatically save interesting frames.

## 2. Label Data

You need to manually label the new images.

1.  **Upload to Roboflow:**
    *   Create a project on [Roboflow](https://roboflow.com/).
    *   Upload the images from `datasets/new_images`.
2.  **Annotate:**
    *   Draw bounding boxes around all resources (Ore, Fiber, etc.) in the new images.
    *   *Tip:* Use Roboflow's "Label Assist" to use your previous model to help label!
3.  **Export Dataset:**
    *   Generate a new version of the dataset.
    *   Export it in **YOLOv8** format.
    *   Download and unzip it to your project folder (replacing or merging with `train/`, `valid/`, `test/`).

## 3. Train the Model

Run the training command to fine-tune your model on the new data.

```bash
yolo task=detect mode=train model=runs/detect/train10/weights/best.pt data=data.yaml epochs=50 imgsz=640
```

*   `model=...`: Points to your *previous* best model (starting point).
*   `data=data.yaml`: Points to your dataset configuration.
*   `epochs=50`: Number of training cycles (adjust as needed).

## 4. Update the Bot

After training finishes:

1.  Find the new best model in `runs/detect/trainXX/weights/best.pt`.
2.  Update `bot.py` to point to this new model path:
    ```python
    MODEL_PATH = "runs/detect/train11/weights/best.pt" # Example
    ```
3.  Increase `CONF_THRESHOLD` in `bot.py` (e.g., to 0.80) and test!
