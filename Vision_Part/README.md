# AI Football App - Vision Part

This repository contains the complete Vision Pipeline for the AI Football Analysis app. It handles everything from training YOLO/TrackNet models to extracting real-world pitch coordinates via homography.

## Project Structure

- **`app/`**: Contains the interactive PySide6 graphical interface (`app.py`) for visual calibration and running the pipeline.
- **`pipeline/`**: The core headless processing engine. Uses a Chain-of-Responsibility pattern to handle detection, tracking, team classification, ball interpolation, and export.
- **`training/`**: Scripts used to extract datasets, augment images, and train the YOLO and TrackNet models from scratch.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your configuration by copying the template:
   ```bash
   cp pipeline/config.example.json pipeline/config.json
   ```
3. Place your model weights inside `pipeline/models/`:
   - `yolo_players.pt`
   - `yolo_ball.pt`
4. Place any API keys (like Roboflow) in a `.env` file at the root.

## Usage

Run the graphical interface to manually calibrate the pitch and kick off tracking:
```bash
python app/app.py
```

Run the pipeline purely headlessly (make sure `config.json` is set up):
```bash
python pipeline/main.py
```
