from .base_handler import BaseHandler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core_logic"))
from player_tracker import PlayerTracker, filter_detections_by_class
from consistency import TrackConsistency

class TrackerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.player_tracker = PlayerTracker()
        self.consistency = TrackConsistency()

    def process_frame(self, frame_data):
        raw_players = frame_data.get('raw_players')

        if raw_players is not None:

            pr_dets = filter_detections_by_class(raw_players, [0, 1])

            tracked_dets = self.player_tracker.update(pr_dets)

            active_ids = tracked_dets.tracker_id if (tracked_dets.tracker_id is not None and len(tracked_dets) > 0) else []
            is_cut = self.consistency.is_scene_cut(active_ids)
            if is_cut:
                self.consistency.on_scene_cut()
                frame_data['scene_cut_detected'] = True

            if tracked_dets.tracker_id is not None and len(tracked_dets) > 0:
                for i in range(len(tracked_dets)):
                    tracked_dets.class_id[i] = self.consistency.lock_class(
                        tracked_dets.tracker_id[i], tracked_dets.class_id[i]
                    )
                    tracked_dets.xyxy[i] = self.consistency.smooth_bbox(
                        tracked_dets.tracker_id[i], tracked_dets.xyxy[i]
                    )

            frame_data['tracked_players'] = tracked_dets

        if self._next_handler: 
            return self._next_handler.process_frame(frame_data)
        return frame_data

    def post_process(self, video_data):
        if self._next_handler: 
            return self._next_handler.post_process(video_data)
        return video_data
