
import numpy as np

class BallTracker:
    def __init__(self, yolo_conf_thresh=0.3, tracknet_conf_thresh=0.5):
        self.yolo_thresh = yolo_conf_thresh
        self.tracknet_thresh = tracknet_conf_thresh
        self.history = []

    def update(self, yolo_det=None, tracknet_det=None):
        result = None

        if yolo_det is not None and yolo_det[2] >= self.yolo_thresh:
            result = {
                "x": float(yolo_det[0]),
                "y": float(yolo_det[1]),
                "source": "yolo",
                "conf": float(yolo_det[2]),
                "is_interpolated": False,
            }

        elif tracknet_det is not None and tracknet_det[2] >= self.tracknet_thresh:
            result = {
                "x": float(tracknet_det[0]),
                "y": float(tracknet_det[1]),
                "source": "tracknet",
                "conf": float(tracknet_det[2]),
                "is_interpolated": False,
            }

        self.history.append(result)
        return result

    def get_raw_trajectory(self):
        return self.history

    def get_interpolated_trajectory(self, max_gap=10, min_real_detections=3):
        n = len(self.history)
        if n == 0:
            return []

        known_frames = []
        known_x = []
        known_y = []
        for i, h in enumerate(self.history):
            if h is not None:
                known_frames.append(i)
                known_x.append(h["x"])
                known_y.append(h["y"])

        if len(known_frames) < 2:
            return list(self.history)

        valid_indices = []
        if len(known_frames) > 2:
            valid_indices.append(0)
            for k in range(1, len(known_frames) - 1):
                prev_x, prev_y = known_x[k-1], known_y[k-1]
                curr_x, curr_y = known_x[k], known_y[k]
                next_x, next_y = known_x[k+1], known_y[k+1]

                dist_prev_next = np.hypot(next_x - prev_x, next_y - prev_y)
                dist_prev_curr = np.hypot(curr_x - prev_x, curr_y - prev_y)
                dist_curr_next = np.hypot(next_x - curr_x, next_y - curr_y)

                if dist_prev_next < 50 and dist_prev_curr > 50 and dist_curr_next > 50:

                    self.history[known_frames[k]] = None
                else:
                    valid_indices.append(k)
            valid_indices.append(len(known_frames) - 1)

            known_frames = [known_frames[i] for i in valid_indices]
            known_x = [known_x[i] for i in valid_indices]
            known_y = [known_y[i] for i in valid_indices]

        interpolatable = set()
        for k in range(len(known_frames) - 1):
            start = known_frames[k]
            end = known_frames[k + 1]
            gap_len = end - start - 1
            if 0 < gap_len <= max_gap:
                for f in range(start + 1, end):
                    interpolatable.add(f)

        interp_x = np.interp(range(n), known_frames, known_x)
        interp_y = np.interp(range(n), known_frames, known_y)

        result = []
        for i in range(n):
            if self.history[i] is not None:
                result.append(dict(self.history[i]))
            elif i in interpolatable:
                result.append({
                    "x": float(interp_x[i]),
                    "y": float(interp_y[i]),
                    "source": "interpolated",
                    "conf": 0.0,
                    "is_interpolated": True,
                })
            else:
                result.append(None)

        def process_segment(seg_indices):
            real_count = sum(1 for idx in seg_indices
                              if result[idx]["source"] in ("yolo", "tracknet", "fused_agree", "dedicated_only", "main_only"))
            if real_count < min_real_detections:
                for idx in seg_indices:
                    result[idx] = None

        seg = []
        for i in range(n):
            if result[i] is not None:
                seg.append(i)
            else:
                if seg:
                    process_segment(seg)
                    seg = []
        if seg:
            process_segment(seg)

        return result

    def get_trajectory_stats(self):
        n = len(self.history)
        if n == 0: return {}
        real_sources = {"yolo", "tracknet", "fused_agree", "dedicated_only", "main_only"}
        n_fused_agree = sum(1 for h in self.history if h and h["source"] == "fused_agree")
        n_dedicated = sum(1 for h in self.history if h and h["source"] == "dedicated_only")
        n_main = sum(1 for h in self.history if h and h["source"] == "main_only")
        n_real = sum(1 for h in self.history if h and h["source"] in real_sources)
        n_lost = sum(1 for h in self.history if h is None)
        interp = self.get_interpolated_trajectory()
        n_interp = sum(1 for h in interp if h and h.get("is_interpolated"))
        n_tracked = sum(1 for h in interp if h is not None)
        return {
            "total_frames": n,
            "real_detections": n_real,
            "fused_agree": n_fused_agree,
            "dedicated_only": n_dedicated,
            "main_only": n_main,
            "lost_frames": n_lost,
            "interpolated_frames": n_interp,
            "total_tracked": n_tracked,
            "tracking_rate": n_tracked / max(n, 1) * 100,
        }
