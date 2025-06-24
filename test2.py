#!/usr/bin/env python3
# flex_wrist_mcp_tip.py — per-finger Wrist→MCP & MCP→TIP vectors + 0-180° angle

import cv2, numpy as np, mediapipe as mp, math, argparse
ALPHA, ARROW = 0.25, 0.55
unit = lambda v: v / (np.linalg.norm(v) + 1e-9)

FINGERS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]  # (MCP , TIP)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 2, 1, 0.7, 0.7)
draw = mp.solutions.drawing_utils

def palm_normal(pts, right):
    w, i, p = pts[0], pts[5], pts[17]
    n = np.cross(i - w, p - w) if right else np.cross(p - w, i - w)
    return unit(n)

def angle(v1, v2):
    cos = np.clip(np.dot(unit(v1), unit(v2)), -1.0, 1.0)
    return math.degrees(math.acos(cos))          # 0–180°

def arrow(img, start, vec, col):
    end = (int(start[0] + vec[0] * ARROW),
           int(start[1] + vec[1] * ARROW))
    cv2.arrowedLine(img, start, end, col, 2, tipLength=0.25)

def run(cam=0):
    cap, smooth = cv2.VideoCapture(cam), {}
    while cap.isOpened():
        ok, frame = cap.read();  h, w = frame.shape[:2]
        if not ok: break
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            for idx, (hl, hd) in enumerate(zip(res.multi_hand_landmarks,
                                               res.multi_handedness)):
                pts = np.array([[l.x, l.y, l.z] for l in hl.landmark])
                right = hd.classification[0].label == 'Right'

                # smooth + optional palm normal (kept for reference, blue)
                n_now = palm_normal(pts, right)
                if idx not in smooth: smooth[idx] = n_now
                smooth[idx] = unit(ALPHA*n_now + (1-ALPHA)*smooth[idx])
                n = smooth[idx]
                cen = (pts[0] + pts[5] + pts[17]) / 3.0
                arrow(frame, (int(cen[0]*w), int(cen[1]*h)),
                      np.array([n[0]*w, n[1]*h]), (255,0,0))

                # per-finger vectors + angle
                for m, t in FINGERS:
                    v_ref  = pts[m] - pts[0]      # Wrist → MCP
                    v_flex = pts[t] - pts[m]      # MCP → TIP
                    deg    = angle(v_ref, v_flex)

                    m_px  = (int(pts[m][0]*w), int(pts[m][1]*h))
                    arrow(frame, (int(pts[0][0]*w), int(pts[0][1]*h)),
                          np.array([v_ref[0]*w, v_ref[1]*h]), (255,255,0))
                    arrow(frame, m_px,
                          np.array([v_flex[0]*w, v_flex[1]*h]), (0,255,0))

                    tip_px = (int(pts[t][0]*w), int(pts[t][1]*h))
                    cv2.putText(frame, f"{deg:5.1f}",
                                (tip_px[0]-18, tip_px[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255,255,255), 2, cv2.LINE_AA)

                draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Wrist→MCP & MCP→TIP vectors (0-180°)", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cam", type=int, default=0)
    run(ap.parse_args().cam)
