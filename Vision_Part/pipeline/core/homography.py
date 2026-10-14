
import numpy as np
import cv2

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_SPOT = (PITCH_LENGTH / 2, PITCH_WIDTH / 2)
CENTER_CIRCLE_RADIUS = 9.15

REAL_FIELD_CORNERS = np.array([
    [0.0, 0.0],
    [PITCH_LENGTH, 0.0],
    [PITCH_LENGTH, PITCH_WIDTH],
    [0.0, PITCH_WIDTH],
], dtype=np.float64)

class HomographyEstimator:
    def __init__(self, min_points=4):
        self.min_points = min_points
        self.H = None                                                               
        self.frames_since_update = 0
        self.max_stale_frames = 150                                                   

    @staticmethod
    def _mask_corners(polygon_points: np.ndarray):
        if polygon_points is None or len(polygon_points) < 4:
            return None
        s = polygon_points.sum(axis=1)
        diff = polygon_points[:, 0] - polygon_points[:, 1]
        top_left = polygon_points[np.argmin(s)]
        bottom_right = polygon_points[np.argmax(s)]
        top_right = polygon_points[np.argmax(diff)]
        bottom_left = polygon_points[np.argmin(diff)]
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)

    def update_from_roboflow(self, field_polygon=None, circle_polygon=None):
        pixel_pts = []
        real_pts = []

        corners = self._mask_corners(field_polygon)
        if corners is not None:
            pixel_pts.extend(corners.tolist())
            real_pts.extend(REAL_FIELD_CORNERS.tolist())

        if circle_polygon is not None and len(circle_polygon) >= 4:
            cx, cy = circle_polygon.mean(axis=0)

            leftmost = circle_polygon[np.argmin(circle_polygon[:, 0])]
            rightmost = circle_polygon[np.argmax(circle_polygon[:, 0])]
            pixel_pts.append([cx, cy])
            real_pts.append([CENTER_SPOT[0], CENTER_SPOT[1]])
            pixel_pts.append(leftmost.tolist())
            real_pts.append([CENTER_SPOT[0] - CENTER_CIRCLE_RADIUS, CENTER_SPOT[1]])
            pixel_pts.append(rightmost.tolist())
            real_pts.append([CENTER_SPOT[0] + CENTER_CIRCLE_RADIUS, CENTER_SPOT[1]])

        if len(pixel_pts) < self.min_points:
            self.frames_since_update += 1
            return False

        src = np.array(pixel_pts, dtype=np.float64)
        dst = np.array(real_pts, dtype=np.float64)

        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if H is not None:
            self.H = H
            self.frames_since_update = 0
            return True

        self.frames_since_update += 1
        return False

    def is_available(self):
        return self.H is not None and self.frames_since_update <= self.max_stale_frames

    def project(self, x_pixel, y_pixel):
        if not self.is_available():
            return None
        pt = np.array([[[x_pixel, y_pixel]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self.H)
        x_m, y_m = out[0][0]
        return float(x_m), float(y_m)

    def project_batch(self, points_pixel):
        if not self.is_available() or len(points_pixel) == 0:
            return None
        pts = np.array(points_pixel, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def update_from_keypoint_model(self, keypoints_dict, mapping_dict):

        name_to_real = {v[0]: v[1] for v in mapping_dict.values()}

        pixel_pts = []
        real_pts = []
        for name, (px, py, conf) in keypoints_dict.items():
            if name in name_to_real:
                pixel_pts.append([px, py])
                real_pts.append(name_to_real[name])

        if len(pixel_pts) >= max(4, self.min_points):
            H, status = cv2.findHomography(
                np.array(pixel_pts, dtype=np.float32),
                np.array(real_pts, dtype=np.float32),
                cv2.RANSAC,
                5.0
            )
            if H is not None:
                self.H = H
                self.frames_since_update = 0
                return True

        self.frames_since_update += 1
        return False

    def update_from_manual_correspondences(self, pixel_pts, real_pts):
        if len(pixel_pts) < self.min_points or len(pixel_pts) != len(real_pts):
            return False

        src = np.array(pixel_pts, dtype=np.float64)
        dst = np.array(real_pts, dtype=np.float64)

        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if H is not None:
            self.H = H
            self.frames_since_update = 0
            return True

        return False
