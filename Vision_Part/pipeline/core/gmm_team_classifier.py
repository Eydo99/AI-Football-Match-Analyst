
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import defaultdict, Counter

class GMMTeamClassifier:
    def __init__(self, n_teams=2, uncertainty_thresh=0.50, method="hsv"):
        self.n_teams = n_teams
        self.uncertainty_thresh = uncertainty_thresh
        self.method = method
        self.embedder = None
        self.gmm = None
        self.fitted = False
        self.cluster_to_team = {}

        self.history_votes = defaultdict(Counter)

        self._locked_team = {}
        self.consecutive_confident_frames = defaultdict(int)
        self.last_confident_team = {}

    def set_lock(self, tracker_id, team_id):
        tid = int(tracker_id)
        if tid in self._locked_team:
            return
        self._locked_team[tid] = team_id

    def fit_from_samples(self, samples: dict):
        player_colors = []
        for tid, colors in samples.items():
            if len(colors) >= 15:
                avg_color = np.mean(colors, axis=0)
                if self.method == "clip":
                    avg_color = avg_color / np.linalg.norm(avg_color)
                player_colors.append(avg_color)

        if len(player_colors) < self.n_teams:
            print(f"[GMMTeamClassifier] WARNING: only {len(player_colors)} "
                  f"players with enough samples, need >= {self.n_teams}. Skipping GMM.")
            self.fitted = False
            return

        X = np.array(player_colors)
        self.gmm = GaussianMixture(n_components=self.n_teams, random_state=42, n_init=5)
        self.gmm.fit(X)
        for i in range(self.n_teams):
            self.cluster_to_team[i] = i

        means = self.gmm.means_
        print("\n[GMMTeamClassifier] Post-fit cluster means in HSV space:")
        for i, m in enumerate(means):
            print(f"  Cluster {i}: {m}")

        print("[GMMTeamClassifier] Pairwise distances between cluster means:")
        import itertools
        pairs = list(itertools.combinations(range(self.n_teams), 2))

        min_cluster_separation = 0.5 if self.method == "hsv" else 0.1

        for i, j in pairs:
            dist = np.linalg.norm(means[i] - means[j])
            print(f"  Dist({i}, {j}) = {dist:.2f}")
            if dist < min_cluster_separation:
                target_team = self.cluster_to_team[i]
                self.cluster_to_team[j] = target_team
                print(f"  -> Merging cluster {j} into team {target_team} (distance < {min_cluster_separation})")

        distinct_teams = set(self.cluster_to_team.values())
        if len(distinct_teams) < 2:
            print("[GMMTeamClassifier] WARNING: Fewer than 2 distinct team clusters remain post-merge. Color separation failed entirely!")

        self.fitted = True
        print(f"[GMMTeamClassifier] Fitted GMM on {len(player_colors)} players. Distinct teams: {len(distinct_teams)}")

    def classify_soft(self, tracker_id, color_vec):
        tid = int(tracker_id)

        if tid in self._locked_team:
            return self._locked_team[tid], 1.0, None

        if not self.fitted or color_vec is None:
            return -1, 0.0, None

        probs = self.gmm.predict_proba(color_vec.reshape(1, -1))[0]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        team_id = self.cluster_to_team[top_idx]

        if top_conf >= self.uncertainty_thresh:
            self.history_votes[tid][team_id] += 1

            if self.last_confident_team.get(tid) == team_id:
                self.consecutive_confident_frames[tid] += 1
            else:
                self.last_confident_team[tid] = team_id
                self.consecutive_confident_frames[tid] = 1

            if self.consecutive_confident_frames[tid] >= 10:
                self.set_lock(tid, team_id)
                return team_id, top_conf, probs

            return -1, 0.0, probs

        return -1, 0.0, probs

    def reset_lock(self, tracker_id):
        tid = int(tracker_id)
        if tid in self._locked_team:
            del self._locked_team[tid]
        self.consecutive_confident_frames[tid] = 0
        self.last_confident_team[tid] = None
        self.history_votes[tid].clear()

    def reset_all_locks(self):
        print("[GMMTeamClassifier] Scene cut detected: resetting all temporal team locks.")
        self._locked_team.clear()
        self.consecutive_confident_frames.clear()
        self.last_confident_team.clear()
        self.history_votes.clear()

    def classify_batch(self, tracker_ids, color_vecs):
        team_ids, confs = [], []
        for tid, cv in zip(tracker_ids, color_vecs):
            t, c, _ = self.classify_soft(tid, cv)
            team_ids.append(t)
            confs.append(c)
        return team_ids, confs
