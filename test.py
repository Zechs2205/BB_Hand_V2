#!/usr/bin/env python3
# hand_abduction_overlay_offset90.py — palm-relative ab/ad, +90 deg offset

import cv2
import mediapipe as mp
import numpy as np

OFFSET_DEG = 90.0                        # constant offset for all fingers

def unit(v, eps=1e-6):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def project(v, n):
    n = unit(n)
    return v - np.sum(v * n, axis=-1, keepdims=True) * n

def signed_angle(v1, v2, n):
    v1u, v2u = unit(v1), unit(v2)
    dot = np.clip(np.sum(v1u * v2u, axis=-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    sign = np.sign(np.sum(np.cross(v1u, v2u) * n, axis=-1))
    return ang * sign

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

FINGER_NAMES = ['Index', 'Middle', 'Ring', 'Pinky']
FINGER_MCP   = [5, 9, 13, 17]
FINGER_PIP   = [6, 10, 14, 18]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand.landmark])

            wrist     = pts[0]
            idx_mcp   = pts[5]
            pky_mcp   = pts[17]
            palm_norm = unit(np.cross(idx_mcp - wrist, pky_mcp - wrist))
            ref_vec   = idx_mcp - pky_mcp          # across-palm baseline
            ref_proj  = project(ref_vec, palm_norm)

            for name, mcp_id, pip_id in zip(FINGER_NAMES, FINGER_MCP, FINGER_PIP):
                finger_vec  = pts[pip_id] - pts[mcp_id]
                finger_proj = project(finger_vec, palm_norm)
                ang = signed_angle(finger_proj, ref_proj, palm_norm)
                if np.isnan(ang):
                    ang = 0.0
                ang += OFFSET_DEG                    # <<< 90-degree offset applied

                cx, cy = int(pts[mcp_id][0]), int(pts[mcp_id][1])
                cv2.putText(frame, f'{name}: {ang:+.1f}°',
                            (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2, cv2.LINE_AA)

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Palm-Relative Ab/Ad (+90°)', frame)
    if cv2.waitKey(1) & 0xFF == 27:          # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
