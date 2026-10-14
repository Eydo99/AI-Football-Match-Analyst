import cv2
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from handlers.detection_handler import YOLOHandler
from handlers.tracking_handler import TrackerHandler
from handlers.team_class_handler import TeamClassifierHandler
from handlers.ball_logic_handler import BallInterpolationHandler
from handlers.export_handler import ExportHandler

import json

def load_config(config_path="config.json"):
    if not Path(config_path).exists():
        print(f"[main] ERROR: config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)

from inference import get_model

def build_roboflow_client(api_key, model_id):
    if not api_key:
        print("[main] No Roboflow API key found in environment -> homography disabled, pitch_x/pitch_y will be null.")
        return None
    try:
        return get_model(model_id=model_id, api_key=api_key)
    except Exception as e:
        print(f"[main] Failed to load local Roboflow model (non-fatal): {e}")
        return None

def main():
    print("=========================================")
    print(" Starting OOP Stage 1 Tracking Pipeline ")
    print("=========================================")

    config = load_config("config.json")
    
    video_path = config.get("video_path")
    output_video = config.get("output_video")
    output_csv = config.get("output_csv")
    PROCESS_FULL_VIDEO = config.get("process_full_video", False)
    FRAME_LIMIT = config.get("frame_limit", 100)
    
    ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
    ROBOFLOW_MODEL_ID = "object-detection-traffic/1"

    if not Path(video_path).exists():
        print(f"[main] ERROR: video not found at {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[main] ERROR: could not open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("[main] WARNING: video reported invalid FPS, defaulting to 25.0")
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    roboflow_client = build_roboflow_client(ROBOFLOW_API_KEY, ROBOFLOW_MODEL_ID)

    detector = YOLOHandler(
        player_model_path=r"models\yolo_players.pt",
        ball_model_path=r"models\yolo_ball.pt",
        roboflow_client=roboflow_client,
        roboflow_model_id=ROBOFLOW_MODEL_ID,
        homography_every_n=10,
    )
    tracker = TrackerHandler()
    team_classifier = TeamClassifierHandler(n_teams=3)
    ball_interpolator = BallInterpolationHandler(max_gap=15, min_real_detections=3)
    exporter = ExportHandler(output_video, output_csv, fps, (w, h))

    detector.set_next(tracker).set_next(team_classifier).set_next(ball_interpolator).set_next(exporter)

    cap = cv2.VideoCapture(video_path)
    video_data = {"original_video_path": video_path, "frames": []}

    print("\n--- Phase 1: Frame-by-Frame Detection & Tracking ---")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not PROCESS_FULL_VIDEO and frame_idx >= FRAME_LIMIT:
            break

        frame_data = {"frame_idx": frame_idx, "frame": frame}
        detector.process_frame(frame_data)

        del frame_data["frame"]
        video_data["frames"].append(frame_data)

        frame_idx += 1
        if frame_idx % 25 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap.release()

    if frame_idx == 0:
        print("[main] ERROR: no frames were read from the video.")
        sys.exit(1)

    print("\n--- Phase 2: Whole-Video Post-Processing (Teams, Interpolation, Render) ---")
    detector.post_process(video_data)

    print(f"\nPipeline Finished! Output video: {output_video}")
    print(f"Output CSV (Stage 2 handoff): {output_csv}")

if __name__ == "__main__":
    main()
