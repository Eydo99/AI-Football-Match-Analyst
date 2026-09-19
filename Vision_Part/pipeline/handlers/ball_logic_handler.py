import pandas as pd
import numpy as np
from .base_handler import BaseHandler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core_logic"))
from ball_tracker import BallTracker

class BallInterpolationHandler(BaseHandler):
    def __init__(self, max_gap=15, min_real_detections=3):
        super().__init__()
        self.max_gap = max_gap
        self.min_real_detections = min_real_detections

        self.tracker = BallTracker(yolo_conf_thresh=0.0)

    def process_frame(self, frame_data):
        fused = frame_data.get('raw_ball')                                    
        source = frame_data.get('ball_source')
        if fused is not None:
            self.tracker.history.append({
                "x": float(fused[0]), "y": float(fused[1]),
                "source": source or "fused", "conf": float(fused[2]),
                "is_interpolated": False,
            })
        else:
            self.tracker.history.append(None)

        if self._next_handler:
            return self._next_handler.process_frame(frame_data)
        return frame_data

    def post_process(self, video_data):
        print("Applying Ball Interpolation (linear, ghost-pruned)...")
        interp = self.tracker.get_interpolated_trajectory(
            max_gap=self.max_gap, min_real_detections=self.min_real_detections
        )

        rows = []
        for h in interp:
            if h is None:
                rows.append({"x": np.nan, "y": np.nan, "conf": 0.0, "source": "lost"})
            else:
                rows.append({
                    "x": h["x"], "y": h["y"], "conf": h.get("conf", 0.0),
                    "source": h.get("source", "unknown"),
                })
        video_data['interpolated_ball'] = rows
        video_data['ball_stats'] = self.tracker.get_trajectory_stats()
        print(f"  Ball tracking stats: {video_data['ball_stats']}")

        if self._next_handler:
            return self._next_handler.post_process(video_data)
        return video_data
