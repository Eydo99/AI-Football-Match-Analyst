
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import defaultdict

class TeamClassifier:

    def __init__(self, n_teams=3, method="hsv"):
        self.n_teams = n_teams
        self.method = method
        self.samples = defaultdict(list)                                         
        self.kmeans = None
        self.fitted = False

        self.embedder = None

        self.cluster_to_team = {}

    @staticmethod
    def _extract_jersey_color(frame_bgr, bbox_xyxy):

        x1, y1, x2, y2 = map(int, bbox_xyxy)

        h_frame, w_frame = frame_bgr.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w < 10 or bbox_h < 10:
            return None

        jersey_y1 = y1 + int(bbox_h * 0.35)
        jersey_y2 = y1 + int(bbox_h * 0.65)

        margin_x = int(bbox_w * 0.10)
        jersey_x1 = x1 + margin_x
        jersey_x2 = x2 - margin_x

        if jersey_x2 - jersey_x1 < 5 or jersey_y2 - jersey_y1 < 5:
            return None

        crop = frame_bgr[jersey_y1:jersey_y2, jersey_x1:jersey_x2]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(green_mask)

        non_green_pixels = hsv[non_green_mask > 0]

        if len(non_green_pixels) < 50:
            return None

        median_hsv = np.median(non_green_pixels, axis=0)
        h, s, v = median_hsv

        angle = h * np.pi / 90.0

        s_norm = s / 255.0
        v_norm = v / 255.0

        return np.array([
            np.cos(angle) * s_norm, 
            np.sin(angle) * s_norm, 
            s_norm, 
            v_norm * 0.1                             
        ])

    def collect_samples(self, frame_bgr, detections):
        if detections.tracker_id is None or len(detections) == 0:
            return

        if self.method == "clip":
            embeddings = self.embedder.embed_batch(frame_bgr, detections.xyxy)
            for i in range(len(detections)):
                tid = int(detections.tracker_id[i])
                emb = embeddings[i]
                if emb is not None:
                    self.samples[tid].append(emb)
        else:
            for i in range(len(detections)):
                tid = int(detections.tracker_id[i])
                color = self._extract_jersey_color(frame_bgr, detections.xyxy[i])
                if color is not None:
                    self.samples[tid].append(color)

    def fit(self):

        player_colors = []
        player_ids = []
        for tid, colors in self.samples.items():
            if len(colors) >= 15:                                                    
                avg_color = np.mean(colors, axis=0)
                if self.method == "clip":

                    avg_color = avg_color / np.linalg.norm(avg_color)
                player_colors.append(avg_color)
                player_ids.append(tid)

        if len(player_colors) < self.n_teams:
            print(f"[TeamClassifier] WARNING: only {len(player_colors)} "
                  f"players with enough samples, need >= {self.n_teams}. "
                  f"Using fallback (all team 0).")
            self.fitted = False
            return

        X = np.array(player_colors)
        self.kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10)
        self.kmeans.fit(X)

        labels = self.kmeans.labels_
        for i in range(self.n_teams):
            self.cluster_to_team[i] = i

        self.fitted = True
        n_per_cluster = [int((labels == i).sum()) for i in range(self.n_teams)]
        print(f"[TeamClassifier] Fitted KMeans on {len(player_colors)} players. "
              f"Cluster sizes: {n_per_cluster}")

    def classify(self, frame_bgr, detections):
        n = len(detections)
        if n == 0:
            return np.array([], dtype=int)

        if not self.fitted:
            return np.zeros(n, dtype=int)

        team_ids = np.zeros(n, dtype=int)

        if self.method == "clip":
            embeddings = self.embedder.embed_batch(frame_bgr, detections.xyxy)
            for i in range(n):
                emb = embeddings[i]
                if emb is not None:
                    label = self.kmeans.predict(emb.reshape(1, -1))[0]
                    team_ids[i] = self.cluster_to_team[label]
                else:
                    team_ids[i] = -1
        else:
            for i in range(n):
                color = self._extract_jersey_color(frame_bgr, detections.xyxy[i])
                if color is not None:
                    label = self.kmeans.predict(color.reshape(1, -1))[0]
                    team_ids[i] = self.cluster_to_team[label]
                else:
                    team_ids[i] = -1           

        return team_ids
