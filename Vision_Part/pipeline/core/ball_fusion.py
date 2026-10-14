
import numpy as np

class BallFusion:
    def __init__(self,
                 agreement_px=40,                                                     
                 dedicated_min_conf=0.15,
                 main_model_min_conf=0.25,
                 max_jump_px=180):                                                
        self.agreement_px = agreement_px
        self.dedicated_min_conf = dedicated_min_conf
        self.main_model_min_conf = main_model_min_conf
        self.max_jump_px = max_jump_px
        self.last_accepted = None          

    def _dist(self, a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _plausible(self, x, y):
        if self.last_accepted is None:
            return True
        return self._dist((x, y), self.last_accepted) <= self.max_jump_px

    def update(self, dedicated_ball=None, main_model_ball=None):
        d_ok = dedicated_ball is not None and dedicated_ball[2] >= self.dedicated_min_conf
        m_ok = main_model_ball is not None and main_model_ball[2] >= self.main_model_min_conf

        candidate = None

        if d_ok and m_ok:
            dist = self._dist(dedicated_ball[:2], main_model_ball[:2])
            if dist <= self.agreement_px:

                wd, wm = dedicated_ball[2], main_model_ball[2]
                x = (dedicated_ball[0] * wd + main_model_ball[0] * wm) / (wd + wm)
                y = (dedicated_ball[1] * wd + main_model_ball[1] * wm) / (wd + wm)
                candidate = (x, y, max(wd, wm), "fused_agree")
            else:

                d_plaus = self._plausible(*dedicated_ball[:2])
                m_plaus = self._plausible(*main_model_ball[:2])
                if d_plaus and not m_plaus:
                    candidate = (*dedicated_ball[:2], dedicated_ball[2], "dedicated_only")
                elif m_plaus and not d_plaus:
                    candidate = (*main_model_ball[:2], main_model_ball[2], "main_only")
                else:
                    best = dedicated_ball if dedicated_ball[2] >= main_model_ball[2] else main_model_ball
                    tag = "dedicated_only" if best is dedicated_ball else "main_only"
                    candidate = (*best[:2], best[2], tag)
        elif d_ok:
            candidate = (*dedicated_ball[:2], dedicated_ball[2], "dedicated_only")
        elif m_ok:
            candidate = (*main_model_ball[:2], main_model_ball[2], "main_only")
        else:
            return None

        x, y, conf, source = candidate
        if not self._plausible(x, y):

            return None

        self.last_accepted = (x, y)
        return {"x": float(x), "y": float(y), "conf": float(conf),
                "source": source, "is_interpolated": False}
