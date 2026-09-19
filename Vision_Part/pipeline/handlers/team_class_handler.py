from .base_handler import BaseHandler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core_logic"))
from team_classifier import TeamClassifier
from gmm_team_classifier import GMMTeamClassifier
from player_tracker import filter_detections_by_class

class TeamClassifierHandler(BaseHandler):
    def __init__(self, n_teams=2, method="hsv"):
        super().__init__()
        self.method = method

        self.embedder = None
        if self.method == "clip":
            from jersey_embedder import JerseyEmbedder
            self.embedder = JerseyEmbedder()

        self.team_classifier = TeamClassifier(n_teams=n_teams, method=method)
        self.gmm_classifier = GMMTeamClassifier(n_teams=n_teams, method=method)

        if self.method == "clip":
            self.team_classifier.embedder = self.embedder
            self.gmm_classifier.embedder = self.embedder

    def process_frame(self, frame_data):
        tracked_players = frame_data.get('tracked_players')
        frame = frame_data.get('frame')

        if tracked_players is not None and tracked_players.tracker_id is not None:

            p_dets = filter_detections_by_class(tracked_players, [0])

            valid_players = sum(1 for colors in self.team_classifier.samples.values() if len(colors) >= 15)
            if valid_players < 15:
                self.team_classifier.collect_samples(frame, p_dets)

        if self._next_handler:
            return self._next_handler.process_frame(frame_data)
        return frame_data

    def post_process(self, video_data):
        print("Fitting KMeans (fallback) + GMM (primary) team clustering...")
        self.team_classifier.fit()
        self.gmm_classifier.fit_from_samples(self.team_classifier.samples)

        video_data['team_classifier_fitted'] = self.team_classifier
        video_data['gmm_classifier_fitted'] = self.gmm_classifier if self.gmm_classifier.fitted else None

        if self._next_handler:
            return self._next_handler.post_process(video_data)
        return video_data
