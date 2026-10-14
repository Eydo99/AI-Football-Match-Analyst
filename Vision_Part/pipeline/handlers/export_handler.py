import cv2
import pandas as pd
import numpy as np
from .base_handler import BaseHandler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core_logic"))
from player_tracker import filter_detections_by_class, get_detection_centers
from team_classifier import TeamClassifier
import draw_utils as du

class ExportHandler(BaseHandler):
    def __init__(self, output_video_path, output_csv_path, fps, dimensions, output_json_path=None):
        super().__init__()
        self.output_video_path = output_video_path
        self.output_csv_path = output_csv_path
        self.output_json_path = output_json_path or (output_csv_path.rsplit('.', 1)[0] + "_meta.json")
        self.fps = fps
        self.width = dimensions[0]
        self.height = dimensions[1]

        self.csv_rows = []
        self.ball_trail = []

    def process_frame(self, frame_data):
        if self._next_handler:
            return self._next_handler.process_frame(frame_data)
        return frame_data

    def _classify_players(self, frame, p_dets, gmm_classifier, kmeans_classifier):
        result = {}
        if p_dets.tracker_id is None or len(p_dets) == 0:
            return result

        if gmm_classifier is not None:
            if gmm_classifier.method == "clip":
                embeddings = gmm_classifier.embedder.embed_batch(frame, p_dets.xyxy)
                for i in range(len(p_dets)):
                    tid = int(p_dets.tracker_id[i])
                    emb = embeddings[i]
                    team_id, conf, _ = gmm_classifier.classify_soft(tid, emb)
                    result[tid] = (team_id, conf)
            else:
                from team_classifier import TeamClassifier
                for i in range(len(p_dets)):
                    tid = int(p_dets.tracker_id[i])
                    color = TeamClassifier._extract_jersey_color(frame, p_dets.xyxy[i])
                    team_id, conf, _ = gmm_classifier.classify_soft(tid, color)
                    result[tid] = (team_id, conf)
        elif kmeans_classifier is not None and kmeans_classifier.fitted:
            team_ids = kmeans_classifier.classify(frame, p_dets)
            for i in range(len(p_dets)):
                tid = int(p_dets.tracker_id[i])
                result[tid] = (int(team_ids[i]), 1.0)
        else:
            for i in range(len(p_dets)):
                result[int(p_dets.tracker_id[i])] = (-1, 0.0)
        return result

    def post_process(self, video_data):
        def _safe_save(save_fn, primary_path, description):
            try:
                save_fn(primary_path)
                return primary_path
            except PermissionError:
                for i in range(1, 100):
                    backup_path = primary_path.rsplit('.', 1)[0] + f"_backup_{i}." + primary_path.rsplit('.', 1)[1]
                    try:
                        save_fn(backup_path)
                        print(f"[ExportHandler] WARNING: {primary_path} is locked. "
                              f"Saving {description} to {backup_path} instead.")
                        return backup_path
                    except PermissionError:
                        continue
                raise PermissionError(f"Could not save {description} because all backup filenames are also locked. Please close the file in Excel.")

        print("Exporting Final Video, CSV, and Stage-2 handoff metadata...")

        cap = cv2.VideoCapture(video_data['original_video_path'])

        writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                  self.fps, (self.width, self.height))

        if not writer.isOpened():
            backup_path = self.output_video_path.rsplit('.', 1)[0] + "_backup." + self.output_video_path.rsplit('.', 1)[1]
            print(f"[ExportHandler] WARNING: {self.output_video_path} is locked or invalid. Saving video to {backup_path} instead.")
            self.output_video_path = backup_path
            writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      self.fps, (self.width, self.height))

        gmm_classifier = video_data.get('gmm_classifier_fitted')
        kmeans_classifier = video_data.get('team_classifier_fitted')
        homography = video_data.get('homography_estimator')
        interpolated_ball = video_data.get('interpolated_ball', [])
        frames = video_data.get('frames', [])

        centroids_json = {}
        display_id_map = {}
        next_display_id = 1

        for frame_idx, frame_data in enumerate(frames):
            ok, frame = cap.read()
            if not ok:
                break

            if frame_data.get('scene_cut_detected') and gmm_classifier is not None:
                gmm_classifier.reset_all_locks()

            centroids_json[str(frame_idx)] = {"players": []}

            tracked_players = frame_data.get('tracked_players')

            if tracked_players is not None and tracked_players.tracker_id is not None:
                p_dets = filter_detections_by_class(tracked_players, [0])
                r_dets = filter_detections_by_class(tracked_players, [1])

                team_map = self._classify_players(frame, p_dets, gmm_classifier, kmeans_classifier)
                foot_pts = get_detection_centers(p_dets) if len(p_dets) else np.empty((0, 2))
                pitch_pts = homography.project_batch(foot_pts) if (homography and homography.is_available() and len(foot_pts)) else None

                for i in range(len(p_dets)):
                    x1, y1, x2, y2 = map(int, p_dets.xyxy[i])
                    tid = int(p_dets.tracker_id[i])

                    if tid not in display_id_map:
                        display_id_map[tid] = next_display_id
                        next_display_id += 1
                    disp_tid = display_id_map[tid]

                    team_id, team_conf = team_map.get(tid, (-1, 0.0))
                    du.draw_player(frame, p_dets.xyxy[i], disp_tid, team_id, is_ref=False)

                    fx, fy = foot_pts[i]
                    px, py = (pitch_pts[i][0], pitch_pts[i][1]) if pitch_pts is not None else (None, None)
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    centroids_json[str(frame_idx)]["players"].append({
                        "pid": tid,
                        "tid": int(team_id),
                        "position": [float(round(cx, 1)), float(round(cy, 1))]
                    })

                    self.csv_rows.append({
                        "frame": frame_idx, "timestamp_s": frame_idx / self.fps,
                        "object_type": "player", "tracker_id": tid,
                        "team_id": team_id, "team_confidence": round(team_conf, 3),
                        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                        "foot_x": float(fx), "foot_y": float(fy),
                        "pitch_x": px, "pitch_y": py, "ball_source": None,
                    })

                for i in range(len(r_dets)):
                    x1, y1, x2, y2 = map(int, r_dets.xyxy[i])
                    tid = int(r_dets.tracker_id[i])

                    if tid not in display_id_map:
                        display_id_map[tid] = next_display_id
                        next_display_id += 1
                    disp_tid = display_id_map[tid]

                    du.draw_player(frame, r_dets.xyxy[i], disp_tid, -1, is_ref=True)

                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    centroids_json[str(frame_idx)]["players"].append({
                        "pid": tid,
                        "tid": -1,
                        "position": [float(round(cx, 1)), float(round(cy, 1))]
                    })

                    self.csv_rows.append({
                        "frame": frame_idx, "timestamp_s": frame_idx / self.fps,
                        "object_type": "referee", "tracker_id": tid,
                        "team_id": -1, "team_confidence": None,
                        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                        "foot_x": None, "foot_y": None,
                        "pitch_x": None, "pitch_y": None, "ball_source": None,
                    })

            if frame_idx < len(interpolated_ball):
                ball = interpolated_ball[frame_idx]
                if pd.notna(ball.get("x")):
                    bx, by = int(ball["x"]), int(ball["y"])
                    self.ball_trail.append((bx, by))

                    centroids_json[str(frame_idx)]["ball"] = {
                        "position": [float(round(ball["x"], 1)), float(round(ball["y"], 1))]
                    }
                    if len(self.ball_trail) > 12:
                        self.ball_trail.pop(0)
                    du.draw_ball_with_trail(frame, self.ball_trail[:-1], current_pt=(bx, by))

                    ball_pitch = homography.project(bx, by) if (homography and homography.is_available()) else None
                    self.csv_rows.append({
                        "frame": frame_idx, "timestamp_s": frame_idx / self.fps,
                        "object_type": "ball", "tracker_id": -1,
                        "team_id": -1, "team_confidence": None,
                        "bbox_x1": bx - 5, "bbox_y1": by - 5, "bbox_x2": bx + 5, "bbox_y2": by + 5,
                        "foot_x": None, "foot_y": None,
                        "pitch_x": ball_pitch[0] if ball_pitch else None,
                        "pitch_y": ball_pitch[1] if ball_pitch else None,
                        "ball_source": ball.get("source"),
                    })
                else:
                    self.ball_trail.clear()

            du.draw_hud(frame, frame_idx, self.fps,
                        homography.is_available() if homography else False)
            writer.write(frame)

        writer.release()
        cap.release()

        df = pd.DataFrame(self.csv_rows)

        df["pitch_x_normalized"] = df["pitch_x"]
        df["pitch_y_normalized"] = df["pitch_y"]
        df["attack_direction_guess"] = None
        try:
            players = df[(df["object_type"] == "player") & df["pitch_x"].notna()]
            team_avg_x = players.groupby("team_id")["pitch_x"].mean()
            valid_teams = team_avg_x[team_avg_x.index >= 0]                        
            if len(valid_teams) >= 2:
                defending_low_team = valid_teams.idxmin()
                attacking_high_team = valid_teams.idxmax()

                def seen_both_halves(team_id, min_frac=0.15):
                    xs = players.loc[players["team_id"] == team_id, "pitch_x"]
                    if len(xs) == 0:
                        return False
                    return (xs < 52.5).mean() >= min_frac and (xs > 52.5).mean() >= min_frac

                confident = seen_both_halves(defending_low_team) and seen_both_halves(attacking_high_team)
                tag_suffix = "" if confident else " [low_confidence: one-sided clip]"

                flip_mask = (df["team_id"] == attacking_high_team) & df["pitch_x"].notna()
                df.loc[flip_mask, "pitch_x_normalized"] = 105.0 - df.loc[flip_mask, "pitch_x"]
                df.loc[flip_mask, "pitch_y_normalized"] = 68.0 - df.loc[flip_mask, "pitch_y"]
                df.loc[df["team_id"] == defending_low_team, "attack_direction_guess"] = f"x=105 (unflipped){tag_suffix}"
                df.loc[df["team_id"] == attacking_high_team, "attack_direction_guess"] = f"x=105 (flipped from x=0){tag_suffix}"
                print(f"  Attack-direction guess: team {defending_low_team} unflipped, "
                      f"team {attacking_high_team} flipped. Confident: {confident}")

                other_teams = [t for t in valid_teams.index
                               if t not in (defending_low_team, attacking_high_team)]
                for t in other_teams:
                    df.loc[df["team_id"] == t, "attack_direction_guess"] =\
                        "ambiguous (3rd cluster, not assigned a direction)"
            else:
                print("  Attack-direction guess skipped (not enough teams with valid pitch coords).")
        except Exception as e:
            print(f"[ExportHandler] Attack-direction guess failed (non-fatal, raw pitch_x/y unaffected): {e}")

        self.output_csv_path = _safe_save(lambda p: df.to_csv(p, index=False), self.output_csv_path, "tracking CSV")

        meta = {
            "schema": {
                "frame": "int, 0-indexed",
                "timestamp_s": "float seconds from video start",
                "object_type": "player | referee | ball",
                "tracker_id": "persistent ByteTrack id, -1 for ball",
                "team_id": "0/1/2 clustered team, -1 unknown (referees/ball)",
                "team_confidence": "GMM posterior prob or KMeans=1.0, null if unknown",
                "bbox_x1/y1/x2/y2": "pixel bounding box",
                "foot_x/foot_y": "pixel bottom-center point (players only)",
                "pitch_x/pitch_y": "meters on 105x68 pitch via homography, null if unavailable that frame",
                "pitch_x_normalized/pitch_y_normalized": "best-effort GUESS: pitch_x/y with one team's coords flipped so both teams attack toward x=105. Based on WHOLE-CLIP average team x-position (not early-frame) — no ground-truth half/side info, so this cannot distinguish 'defends this end all match' from 'attacking this end for this whole highlight clip'. Equals raw pitch_x/y when guess was skipped (fewer than 2 teams with valid coords) or left unassigned for a 3rd cluster beyond the two extreme teams (see attack_direction_guess).",
                "attack_direction_guess": "'x=105 (unflipped)' | 'x=105 (flipped from x=0)', optionally suffixed ' [low_confidence: one-sided clip]' when a team never appeared on both halves of the pitch (typical for short highlight clips -- can't distinguish 'defends this end' from 'attacking this end right now'). null if guess was skipped entirely. NOT ground truth either way.",
                "ball_source": "fused_agree | dedicated_only | main_only | interpolated | lost, ball rows only",
            },
            "pitch_dimensions_m": {"length": 105.0, "width": 68.0},
            "fps": self.fps,
            "video_dimensions_px": {"width": self.width, "height": self.height},
            "ball_stats": video_data.get("ball_stats", {}),
            "note": "This table is the Stage 1 -> Stage 2 handoff. It matches "
                    "the shape a downloaded tracking dataset would provide "
                    "(Timestamp, Player_ID, Team_ID, X, Y) once pitch_x/pitch_y "
                    "is populated, so Stage 2 can treat video-derived and "
                    "downloaded data identically per the project design.",
        }
        import json

        def write_json(data, path):
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        self.output_json_path = _safe_save(lambda p: write_json(meta, p), self.output_json_path, "handoff metadata")

        output_json_centroid_path = self.output_csv_path.rsplit('.', 1)[0] + "_centroids.json"
        output_json_centroid_path = _safe_save(lambda p: write_json(centroids_json, p), output_json_centroid_path, "centroids metadata")

        print(f"Saved Final Video: {self.output_video_path}")
        print(f"Saved CSV: {self.output_csv_path}")
        print(f"Saved handoff metadata: {self.output_json_path}")
        print(f"Saved centroids metadata: {output_json_centroid_path}")

        if self._next_handler:
            return self._next_handler.post_process(video_data)
        return video_data
