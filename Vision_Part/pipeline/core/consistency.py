
import numpy as np
from collections import defaultdict, Counter, deque

class ClassLock:

    def __init__(self, lock_after=5):
        self.lock_after = lock_after
        self.votes = defaultdict(Counter)                                            
        self.locked = {}                                                   

    def vote(self, tracker_id, raw_class_id):
        tid = int(tracker_id)

        if tid in self.locked:
            return self.locked[tid]

        self.votes[tid][raw_class_id] += 1
        total = sum(self.votes[tid].values())

        if total >= self.lock_after:
            winner = self.votes[tid].most_common(1)[0][0]
            self.locked[tid] = winner
            return winner

        return self.votes[tid].most_common(1)[0][0]

class SceneCutDetector:

    def __init__(self, drop_ratio=0.6, min_tracks=4):
        self.drop_ratio = drop_ratio
        self.min_tracks = min_tracks
        self.prev_ids = set()

    def check(self, current_ids):
        current_ids = set(int(x) for x in current_ids)

        if len(self.prev_ids) < self.min_tracks:
            self.prev_ids = current_ids
            return False

        survived = len(self.prev_ids & current_ids)
        dropped_ratio = 1.0 - (survived / len(self.prev_ids))

        self.prev_ids = current_ids

        return dropped_ratio >= self.drop_ratio

class BboxSmoother:

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.states = {}                               

    def smooth(self, tracker_id, raw_xyxy):
        tid = int(tracker_id)
        raw = np.asarray(raw_xyxy, dtype=np.float64)

        if tid not in self.states:
            self.states[tid] = raw.copy()
            return raw.copy()

        self.states[tid] = self.alpha * raw + (1 - self.alpha) * self.states[tid]
        return self.states[tid].copy()

    def remove(self, tracker_id):
        self.states.pop(int(tracker_id), None)

class BallVelocityGuard:

    def __init__(self, max_px_per_frame=200, adaptive_factor=3.0, history_len=10):
        self.max_px = max_px_per_frame
        self.adaptive_factor = adaptive_factor
        self.speed_history = deque(maxlen=history_len)
        self.last_pos = None

    def validate(self, x, y):
        if self.last_pos is None:
            self.last_pos = (x, y)
            return True, 0.0

        dist = np.sqrt((x - self.last_pos[0])**2 + (y - self.last_pos[1])**2)

        if dist > self.max_px:
            return False, dist

        if len(self.speed_history) >= 3:
            avg_speed = np.mean(self.speed_history)
            threshold = max(avg_speed * self.adaptive_factor, 50.0)                
            if dist > threshold:
                return False, dist

        self.speed_history.append(dist)
        self.last_pos = (x, y)
        return True, dist

    def force_update(self, x, y):
        self.last_pos = (x, y)
        self.speed_history.clear()

    def reset(self):
        self.last_pos = None
        self.speed_history.clear()

class TrackConsistency:

    def __init__(self,
                 class_lock_after=5,
                 scene_cut_drop_ratio=0.6,
                 bbox_smooth_alpha=0.4,
                 ball_max_px_per_frame=200,
                 ball_adaptive_factor=3.0):

        self.class_lock = ClassLock(lock_after=class_lock_after)
        self.scene_cut = SceneCutDetector(drop_ratio=scene_cut_drop_ratio)
        self.bbox_smoother = BboxSmoother(alpha=bbox_smooth_alpha)
        self.ball_guard = BallVelocityGuard(
            max_px_per_frame=ball_max_px_per_frame,
            adaptive_factor=ball_adaptive_factor,
        )

    def lock_class(self, tracker_id, raw_class_id):
        return self.class_lock.vote(tracker_id, raw_class_id)

    def is_scene_cut(self, tracker_ids):
        if tracker_ids is None or len(tracker_ids) == 0:
            return False
        return self.scene_cut.check(set(tracker_ids))

    def smooth_bbox(self, tracker_id, raw_xyxy):
        return self.bbox_smoother.smooth(tracker_id, raw_xyxy)

    def validate_ball(self, x, y):
        is_ok, dist = self.ball_guard.validate(x, y)
        return is_ok

    def reset_ball(self, x=None, y=None):
        if x is not None and y is not None:
            self.ball_guard.force_update(x, y)
        else:
            self.ball_guard.reset()

    def on_scene_cut(self):
        self.ball_guard.reset()
        self.bbox_smoother.states.clear()

