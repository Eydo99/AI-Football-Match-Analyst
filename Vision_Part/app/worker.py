import os
import sys
import json
import cv2
from PySide6.QtCore import QThread, Signal
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from handlers.detection_handler import YOLOHandler
from handlers.tracking_handler import TrackerHandler
from handlers.team_class_handler import TeamClassifierHandler
from handlers.ball_logic_handler import BallInterpolationHandler
from handlers.export_handler import ExportHandler

class PipelineWorker(QThread):
    progress_updated = Signal(int, str)
    finished_successfully = Signal(str, str)                       
    error_occurred = Signal(str)

    def __init__(self, video_path, precomputed_homography, frame_fraction=1.0):
        super().__init__()
        self.video_path = video_path
        self.precomputed_homography = precomputed_homography
        self.frame_fraction = frame_fraction
        self._is_cancelled = False

    def run(self):
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "pipeline", "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)

            roboflow_model_id = "object-detection-traffic/1"
            api_key = os.environ.get("ROBOFLOW_API_KEY")

            roboflow_client = None
            if api_key:
                try:
                    from inference import get_model
                    roboflow_client = get_model(model_id=roboflow_model_id, api_key=api_key)
                except:
                    pass

            stage1_dir = os.path.join(os.path.dirname(__file__), "..", "pipeline")
            player_model_path = os.path.join(stage1_dir, "models", "yolo_players.pt")
            ball_model_path = os.path.join(stage1_dir, "models", "yolo_ball.pt")

            output_video = config.get("output_video", "output_tracking.mp4")
            output_csv = config.get("output_csv", "tracking_data.csv")

            detector = YOLOHandler(
                player_model_path=player_model_path,
                ball_model_path=ball_model_path,
                roboflow_client=roboflow_client,
                roboflow_model_id=roboflow_model_id,
                precomputed_homography=self.precomputed_homography
            )
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            tracker = TrackerHandler()
            tc_method = config.get("team_classification_method", "hsv")
            team_classifier = TeamClassifierHandler(method=tc_method)
            ball_logic = BallInterpolationHandler()
            exporter = ExportHandler(output_video, output_csv, fps, (w, h))

            detector.set_next(tracker).set_next(team_classifier).set_next(ball_logic).set_next(exporter)

            limit = config.get("frame_limit", 100)
            if getattr(self, "frame_fraction", 1.0) < 1.0:
                total_frames = max(1, int(total_frames * self.frame_fraction))
            elif not config.get("process_full_video", False):
                total_frames = limit

            frames_list = []

            for frame_idx in range(total_frames):
                if self._is_cancelled:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                frame_data = {'frame_idx': frame_idx, 'frame': frame}
                detector.process_frame(frame_data)

                if 'frame' in frame_data:
                    del frame_data['frame']
                frames_list.append(frame_data)

                if frame_idx % 25 == 0 or frame_idx == total_frames - 1:
                    percent = int((frame_idx / max(1, total_frames)) * 90)                          
                    self.progress_updated.emit(percent, f"Processing frame {frame_idx}/{total_frames}")

            cap.release()

            self.progress_updated.emit(95, "Running post-processing (teams, interpolation, export)...")

            video_data = {
                'original_video_path': self.video_path,
                'total_frames': frame_idx,
                'fps': fps,
                'width': w,
                'height': h,
                'output_video_path': output_video,
                'frames': frames_list
            }
            detector.post_process(video_data)

            final_csv = exporter.output_csv_path
            final_vid = exporter.output_video_path

            abs_csv = os.path.abspath(final_csv)
            abs_vid = os.path.abspath(final_vid)
            self.finished_successfully.emit(abs_csv, abs_vid)

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print("PIPELINE CRASHED:")
            print(err)
            self.error_occurred.emit(str(e))
