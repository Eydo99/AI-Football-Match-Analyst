import numpy as np
from .base_handler import BaseHandler
from ultralytics import YOLO
import supervision as sv
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core_logic"))
from ball_fusion import BallFusion
from homography import HomographyEstimator

class YOLOHandler(BaseHandler):

    MAIN_PLAYER_CLASS = 0
    MAIN_REF_CLASS = 1
    MAIN_BALL_CLASS = 2

    def __init__(self, player_model_path, ball_model_path,
                 roboflow_client=None, roboflow_model_id="football-match-segmentation/1",
                 homography_every_n=10, precomputed_homography=None):
        super().__init__()
        self.player_model = YOLO(player_model_path)
        self.ball_model = YOLO(ball_model_path)
        self.ball_fusion = BallFusion()

        from pitch_keypoint_detector import PitchKeypointDetector
        self.keypoint_detector = PitchKeypointDetector()

        self.homography = HomographyEstimator()
        self.roboflow_model = roboflow_client                                
        self.roboflow_model_id = roboflow_model_id

        if homography_every_n != 10:                                      
             pass

        self.calibration_frame_indices = [0, 15, 30, 45, 60]
        self.calibration_candidates = {}                                                

        if precomputed_homography is not None:
            self.homography.H = precomputed_homography
            self._calibrated = True
            print("[YOLOHandler] Initialized with PRECOMPUTED homography. Skipping auto-calibration.")
        else:
            self._calibrated = False
            print("[YOLOHandler] homography_every_n is deprecated. Using fixed 5-frame calibration window instead.")

    def _best_ball_from_detections(self, ball_dets):
        if len(ball_dets) == 0:
            return None
        best_idx = int(np.argmax(ball_dets.confidence))
        xyxy = ball_dets.xyxy[best_idx]
        bx = (xyxy[0] + xyxy[2]) / 2.0
        by = (xyxy[1] + xyxy[3]) / 2.0
        return (float(bx), float(by), float(ball_dets.confidence[best_idx]))

    def _run_roboflow_segmentation(self, frame):
        if self.roboflow_model is None:
            return None, None
        try:
            result = self.roboflow_model.infer(frame)[0]

            print(f"\n[DEBUG] Raw Roboflow result type: {type(result)}")
            print(f"[DEBUG] Raw Roboflow result dir: {dir(result)}")

            field_polygon, circle_polygon = None, None

            predictions = getattr(result, "predictions", [])
            if not predictions and isinstance(result, dict):
                predictions = result.get("predictions", [])

            for pred in predictions:

                if isinstance(pred, dict):
                    cls_name = pred.get("class", "")
                    points = pred.get("points", [])

                    if not points and "x" in pred and "y" in pred and "width" in pred:
                        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                        points = [
                            {"x": x - w/2, "y": y - h/2},
                            {"x": x + w/2, "y": y - h/2},
                            {"x": x + w/2, "y": y + h/2},
                            {"x": x - w/2, "y": y + h/2}
                        ]
                else:
                    cls_name = getattr(pred, "class_name", getattr(pred, "class", ""))
                    points = getattr(pred, "points", [])

                    if not points and hasattr(pred, "x") and hasattr(pred, "y") and hasattr(pred, "width"):
                        x = getattr(pred, "x")
                        y = getattr(pred, "y")
                        w = getattr(pred, "width")
                        h = getattr(pred, "height")
                        points = [
                            {"x": x - w/2, "y": y - h/2},
                            {"x": x + w/2, "y": y - h/2},
                            {"x": x + w/2, "y": y + h/2},
                            {"x": x - w/2, "y": y + h/2}
                        ]

                if not points:
                    continue

                pts = []
                for p in points:
                    if isinstance(p, dict):
                        pts.append([p["x"], p["y"]])
                    else:
                        pts.append([p.x, p.y])
                pts = np.array(pts, dtype=np.float64)

                if "Field" in cls_name and field_polygon is None:
                    field_polygon = pts
                elif "Central Circle" in cls_name and circle_polygon is None:
                    circle_polygon = pts
            return field_polygon, circle_polygon
        except Exception as e:
            print(f"[YOLOHandler] Roboflow segmentation call failed (non-fatal): {e}")
            return None, None

    def _try_calibrate_from_candidates(self):
        best_frame_idx = None
        best_point_count = 0
        best_H = None
        best_method = None

        name_to_real = {v[0]: v[1] for v in self.keypoint_detector.keypoint_mapping.values()}

        for idx, (keypoints_dict, field_poly, circle_poly) in self.calibration_candidates.items():

            temp_kpt = HomographyEstimator(min_points=self.homography.min_points)
            kpt_success = False
            kpt_points = 0
            if keypoints_dict is not None:
                kpt_points = sum(1 for k in keypoints_dict if k in name_to_real)
                if kpt_points >= 4:
                    kpt_success = temp_kpt.update_from_keypoint_model(keypoints_dict, self.keypoint_detector.keypoint_mapping)

            temp_robo = HomographyEstimator(min_points=self.homography.min_points)
            robo_success = temp_robo.update_from_roboflow(field_poly, circle_poly)
            n_field = 4 if field_poly is not None and len(field_poly) >= 4 else 0
            n_circle = 3 if circle_poly is not None and len(circle_poly) >= 4 else 0
            robo_points = n_field + n_circle if robo_success else 0

            print(f"[YOLOHandler] Calibration candidate frame {idx}: Keypoint ({kpt_points} pts) vs Roboflow ({robo_points} pts).")

            frame_best_pts = max(kpt_points if kpt_success else 0, robo_points)
            frame_best_H = temp_kpt.H if (kpt_success and kpt_points >= robo_points) else (temp_robo.H if robo_success else None)
            frame_best_method = "Keypoint" if (kpt_success and kpt_points >= robo_points) else "Roboflow"

            if frame_best_H is not None and frame_best_pts > best_point_count:
                best_point_count = frame_best_pts
                best_frame_idx = idx
                best_H = frame_best_H
                best_method = frame_best_method

        if best_H is not None:
            self.homography.H = best_H
            self.homography.frames_since_update = 0
            self._calibrated = True
            print(f"[YOLOHandler] Homography calibrated from frame {best_frame_idx} "
                  f"({best_point_count} correspondence points via {best_method}). Frozen.")
            print(f"[YOLOHandler] Total API/Model calls made: {len(self.calibration_candidates)}")
        else:
            self._calibrated = True                 
            print("[YOLOHandler] Calibration failed on all 5 candidate frames.")

    def process_frame(self, frame_data):
        frame = frame_data['frame']
        frame_idx = frame_data.get('frame_idx', 0)

        p_res = self.player_model(frame, conf=0.3, verbose=False)[0]
        raw_players = sv.Detections.from_ultralytics(p_res)
        frame_data['raw_players'] = raw_players

        b_res = self.ball_model(frame, conf=0.6, verbose=False)[0]
        dedicated_ball_dets = sv.Detections.from_ultralytics(b_res)
        dedicated_ball = self._best_ball_from_detections(dedicated_ball_dets)

        main_ball_mask = raw_players.class_id == self.MAIN_BALL_CLASS
        main_ball_dets = raw_players[main_ball_mask]
        main_model_ball = self._best_ball_from_detections(main_ball_dets)

        fused_ball = self.ball_fusion.update(dedicated_ball=dedicated_ball,
                                              main_model_ball=main_model_ball)
        frame_data['raw_ball'] = (
            (fused_ball['x'], fused_ball['y'], fused_ball['conf']) if fused_ball else None
        )
        frame_data['ball_source'] = fused_ball['source'] if fused_ball else None

        if not self._calibrated:
            keypoints_dict = None
            valid_kpt_count = 0

            if self.keypoint_detector.available:
                keypoints_dict = self.keypoint_detector.detect(frame, conf_thresh=0.5)
                name_to_real = {v[0]: v[1] for v in self.keypoint_detector.keypoint_mapping.values()}
                valid_kpt_count = sum(1 for k in keypoints_dict if k in name_to_real)

                if valid_kpt_count >= 10:
                    success = self.homography.update_from_keypoint_model(keypoints_dict, self.keypoint_detector.keypoint_mapping)
                    if success:
                        self._calibrated = True
                        print(f"[YOLOHandler] Homography calibrated via Keypoint Model on frame {frame_idx} with {valid_kpt_count} points. Skipping Roboflow.")

            if not self._calibrated and frame_idx in self.calibration_frame_indices:
                field_poly, circle_poly = self._run_roboflow_segmentation(frame)

                self.calibration_candidates[frame_idx] = (keypoints_dict, field_poly, circle_poly)

                if frame_idx == self.calibration_frame_indices[-1]:
                    self._try_calibrate_from_candidates()

        frame_data['homography_available'] = self.homography.is_available()

        if self._next_handler:
            return self._next_handler.process_frame(frame_data)
        return frame_data

    def post_process(self, video_data):

        video_data['homography_estimator'] = self.homography
        if self._next_handler:
            return self._next_handler.post_process(video_data)
        return video_data
