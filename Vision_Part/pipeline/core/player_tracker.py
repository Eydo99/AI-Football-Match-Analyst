
import numpy as np
import supervision as sv

class PlayerTracker:

    def __init__(self,
                 track_activation_threshold=0.25,
                 lost_track_buffer=30,
                 minimum_matching_threshold=0.8,
                 frame_rate=25):
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections

        tracked = self.byte_tracker.update_with_detections(detections)
        return tracked

    def reset(self):
        self.byte_tracker.reset()

def filter_detections_by_class(detections: sv.Detections, class_ids: list):
    if detections.class_id is None or len(detections) == 0:
        return detections

    mask = np.isin(detections.class_id, class_ids)
    return detections[mask]

def get_detection_centers(detections: sv.Detections):
    if len(detections) == 0:
        return np.empty((0, 2))

    x_center = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
    y_bottom = detections.xyxy[:, 3]                         
    return np.column_stack([x_center, y_bottom])
