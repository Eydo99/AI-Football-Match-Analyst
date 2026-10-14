
import cv2
import numpy as np

TEAM_COLORS = {
    0: (60, 60, 220),                
    1: (220, 150, 60),                
    2: (60, 200, 220),                                          
    -1: (140, 140, 140),                
}
REF_COLOR = (230, 230, 230)
BALL_COLOR = (0, 165, 255)

def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner_len_ratio=0.18):
    w, h = x2 - x1, y2 - y1
    cl = max(6, int(min(w, h) * corner_len_ratio))

    corners = [
        ((x1, y1), (x1 + cl, y1), (x1, y1 + cl)),
        ((x2, y1), (x2 - cl, y1), (x2, y1 + cl)),
        ((x1, y2), (x1 + cl, y2), (x1, y2 - cl)),
        ((x2, y2), (x2 - cl, y2), (x2, y2 - cl)),
    ]
    for corner, h_end, v_end in corners:
        cv2.line(frame, corner, h_end, color, thickness, cv2.LINE_AA)
        cv2.line(frame, corner, v_end, color, thickness, cv2.LINE_AA)

def draw_label(frame, x1, y1, text, color, text_color=(15, 15, 15)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.42, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 3
    y_top = max(0, y1 - th - 2 * pad - 2)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 2 * pad, y_top + th + 2 * pad), color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x1 + pad, y_top + th + pad - 1), font, scale, text_color, thick, cv2.LINE_AA)

def draw_player(frame, xyxy, tracker_id, team_id, is_ref=False):
    x1, y1, x2, y2 = map(int, xyxy)
    color = REF_COLOR if is_ref else TEAM_COLORS.get(team_id, TEAM_COLORS[-1])
    draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2)
    label = f"REF {tracker_id}" if is_ref else f"#{tracker_id}"
    draw_label(frame, x1, y1, label, color)

def draw_ball_with_trail(frame, trail_points, current_pt=None):
    n = len(trail_points)
    for i, pt in enumerate(trail_points):
        alpha = (i + 1) / max(n, 1)                              
        radius = 1 + int(2 * alpha)
        overlay_color = tuple(int(c * alpha + 255 * 0.15 * (1 - alpha)) for c in BALL_COLOR)
        cv2.circle(frame, pt, radius, overlay_color, -1, cv2.LINE_AA)

    if current_pt is not None:
        cv2.circle(frame, current_pt, 5, BALL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, current_pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

def draw_hud(frame, frame_idx, fps, homography_available):
    h, w = frame.shape[:2]
    text = f"f{frame_idx}  t={frame_idx / fps:0.1f}s  homography:{'ok' if homography_available else 'stale'}"
    cv2.putText(frame, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
