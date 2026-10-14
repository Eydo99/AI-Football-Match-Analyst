import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

class PitchKeypointDetector:
    def __init__(self, model_repo="cdqqqqqq/Soccana_Keypoint", filename="Model/weights/best.pt"):
        print(f"[PitchKeypoint] Downloading/Loading {model_repo}/{filename}...")
        try:
            model_path = hf_hub_download(repo_id=model_repo, filename=filename)
            self.model = YOLO(model_path)
            self.available = True
            print("[PitchKeypoint] Model loaded successfully.")
        except Exception as e:
            print(f"[PitchKeypoint] Failed to load model: {e}")
            self.available = False

        self.keypoint_mapping = {
            0: ("sideline_top_left", (0.0, 0.0)),
            1: ("big_rect_left_top_pt1", (0.0, 13.84)),
            2: ("big_rect_left_top_pt2", (16.5, 13.84)),
            3: ("big_rect_left_bottom_pt1", (16.5, 54.16)),
            4: ("big_rect_left_bottom_pt2", (0.0, 54.16)),
            5: ("small_rect_left_top_pt1", (0.0, 24.84)),
            6: ("small_rect_left_top_pt2", (5.5, 24.84)),
            7: ("small_rect_left_bottom_pt1", (5.5, 43.16)),
            8: ("small_rect_left_bottom_pt2", (0.0, 43.16)),
            9: ("sideline_bottom_left", (0.0, 68.0)),
            10: ("left_semicircle_right", (20.15, 34.0)),              
            11: ("center_line_top", (52.5, 0.0)),
            12: ("center_line_bottom", (52.5, 68.0)),
            13: ("center_circle_top", (52.5, 24.85)),
            14: ("center_circle_bottom", (52.5, 43.15)),
            15: ("field_center", (52.5, 34.0)),
            16: ("sideline_top_right", (105.0, 0.0)),
            17: ("big_rect_right_top_pt1", (105.0, 13.84)),
            18: ("big_rect_right_top_pt2", (88.5, 13.84)),
            19: ("big_rect_right_bottom_pt1", (88.5, 54.16)),
            20: ("big_rect_right_bottom_pt2", (105.0, 54.16)),
            21: ("small_rect_right_top_pt1", (105.0, 24.84)),
            22: ("small_rect_right_top_pt2", (99.5, 24.84)),
            23: ("small_rect_right_bottom_pt1", (99.5, 43.16)),
            24: ("small_rect_right_bottom_pt2", (105.0, 43.16)),
            25: ("sideline_bottom_right", (105.0, 68.0)),
            26: ("right_semicircle_left", (84.85, 34.0)),                  
            27: ("center_circle_left", (43.35, 34.0)),
            28: ("center_circle_right", (61.65, 34.0)),
        }

    def detect(self, frame_bgr, conf_thresh=0.5):
        if not self.available:
            return {}

        results = self.model(frame_bgr, verbose=False)
        landmarks = {}

        if len(results) > 0 and results[0].keypoints is not None:

            keypoints = results[0].keypoints.data[0].cpu().numpy()                               

            for idx, (x, y, conf) in enumerate(keypoints):
                if conf > conf_thresh and idx in self.keypoint_mapping:
                    name = self.keypoint_mapping[idx][0]
                    landmarks[name] = (float(x), float(y), float(conf))

        return landmarks
