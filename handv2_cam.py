#!/usr/bin/env python3
# hand_control_serial_webcam.py
#
# Webcam-based Mediapipe hand-tracking → serial control packet
#
# Packet format:
#   A  idxAB  midAB  ringAB  pinkyAB  thumbFLX  idxFLX  midFLX  ringFLX  pinkyFLX  thumbAB
#   Example:
#   A -12  8  15  0  35  42  60  75  80  -5
#
# Press ESC to quit; keys 1-4 flip ab/ad sign for Index-Pinky

import cv2, mediapipe as mp, numpy as np, math, time, serial, sys

# ───────── USER CONFIG ─────────
SERIAL_PORT = "COM11"        # Windows "COMx", Linux "/dev/ttyUSB0"; set None to disable
BAUD_RATE   = 1_000_000      # 1 Mbps default

FLEX_MIN  = [0,   0,   0,   0,   0]
FLEX_MAX  = [270, 270, 270, 270, 270]
FLEX_TLO  = [20,  15,  15,  15,  15]
FLEX_THI  = [50, 100, 100, 100,  90]
FLEX_REV  = [False, False, False, False, False]

AB_MIN   = [-30]*4
AB_MAX   = [ 30]*4
AB_REV   = [True]*4           # start with reversed ab/ad for Index-Pinky

OFFSET_DEG     = 90.0         # +90° so “straight” = 90
AB_DISABLE_TOL = 2.0          # flex ≈90° ±2 → ab/ad forced 0
# ───────────────────────────────

unit  = lambda v,eps=1e-6: v / (np.linalg.norm(v,axis=-1,keepdims=True)+eps)
clamp = lambda x,lo,hi: lo if x<lo else hi if x>hi else x

def safe_angle(v1, v2):
    v1u, v2u = unit(v1).flatten(), unit(v2).flatten()
    cos = max(-1.0, min(1.0, np.dot(v1u, v2u)))
    return math.degrees(math.acos(cos))

def project(v, n):
    n = unit(n);  return v - np.sum(v*n,axis=-1,keepdims=True)*n

def signed_angle(v1, v2, n):
    v1u, v2u = unit(v1), unit(v2)
    dot  = np.clip(np.sum(v1u*v2u,axis=-1), -1.0, 1.0)
    ang  = np.degrees(np.arccos(dot))
    sign = np.sign(np.sum(np.cross(v1u, v2u)*n, axis=-1))
    return ang*sign

FINGER_MCP   = [5, 9, 13, 17]                  # Index-Pinky MCP joints
FINGER_PIP   = [6,10,14,18]
FLEX_FINGERS = [(1,4), (5,8), (9,12), (13,16), (17,20)]  # Thumb-Pinky tip pairs

def abduction_angles(pts):
    wrist, idx, pky = pts[0], pts[5], pts[17]
    n   = unit(np.cross(idx-wrist, pky-wrist))
    ref = project(idx - pky, n)
    ab  = []
    for k,(mcp,pip) in enumerate(zip(FINGER_MCP, FINGER_PIP)):
        vec = project(pts[pip]-pts[mcp], n)
        ang = signed_angle(vec, ref, n) + OFFSET_DEG
        if AB_REV[k]: ang = -ang
        ab.append(clamp(ang, AB_MIN[k], AB_MAX[k]))
    thumb_vec = project(pts[3]-pts[2], n)
    thumb_ang = signed_angle(thumb_vec, ref, n) + OFFSET_DEG
    thumb_ang = clamp(thumb_ang, -30, 30)
    return ab, thumb_ang

def flexion_angles(pts):
    wrist = pts[0]
    vals  = []
    for k,(mcp,tip) in enumerate(FLEX_FINGERS):
        ang = safe_angle(pts[mcp]-wrist, pts[tip]-pts[mcp])
        if FLEX_REV[k]: ang = -ang
        ang = clamp(ang, FLEX_MIN[k], FLEX_MAX[k])
        if ang <= FLEX_TLO[k]: ang = FLEX_MIN[k]
        if ang >= FLEX_THI[k]: ang = FLEX_MAX[k]
        vals.append(ang)
    return vals

ser = None
try:
    if SERIAL_PORT:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Serial open → {SERIAL_PORT} @ {BAUD_RATE}")
except serial.SerialException:
    print(f"[WARN] Serial port {SERIAL_PORT} unavailable → packets disabled.", file=sys.stderr)

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(False, 1, 1, 0.7, 0.7)
draw     = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)      # default webcam
if not cap.isOpened():
    sys.exit("[ERROR] Cannot open webcam")

frame_cnt, t0 = 0, time.time()

def window_title(flags):
    tag = '|'.join(f'{n}{"-" if r else "+"}' for n,r in zip("IMRP", flags))
    return f'Flex / AbAd  [{tag}]  –  ESC quit, 1-4 flip sign'

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Camera frame read failed")
        break
    h, w = frame.shape[:2]

    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        for hl in res.multi_hand_landmarks:
            pts = np.array([[lm.x*w, lm.y*h, lm.z*w] for lm in hl.landmark])
            abd, thumb_ab = abduction_angles(pts)
            flex          = flexion_angles(pts)

            for i, flex_deg in enumerate(flex[1:]):  # Index-Pinky
                if abs(flex_deg-90) <= AB_DISABLE_TOL:
                    abd[i] = 0.0

            for (mcp,val) in zip(FINGER_MCP, abd):
                x,y = int(pts[mcp][0]), int(pts[mcp][1])
                cv2.putText(frame, f'{val:+.1f}', (x-22,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
            for ((_,tip),val) in zip(FLEX_FINGERS, flex):
                x,y = int(pts[tip][0]), int(pts[tip][1])
                cv2.putText(frame, f'{val:5.1f}', (x-18,y-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),2)

            draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            packet = [
                int(round(abd[0])), int(round(abd[1])), int(round(abd[2])), int(round(abd[3])),
                int(round(flex[0])), int(round(flex[1])), int(round(flex[2])), int(round(flex[3])), int(round(flex[4])),
                int(round(thumb_ab))
            ]
            if ser: ser.write(("A "+" ".join(map(str,packet))+"\n").encode())

    frame_cnt += 1
    fps = frame_cnt / (time.time()-t0)
    cv2.putText(frame,f'Frames:{frame_cnt}  FPS:{fps:4.1f}',(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)

    cv2.setWindowTitle('main', window_title(AB_REV))
    cv2.imshow('main', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    if key in (49,50,51,52): AB_REV[key-49] = not AB_REV[key-49]

cap.release()
if ser: ser.close()
cv2.destroyAllWindows()
#!/usr/bin/env python3
# hand_control_serial_webcam.py
#
# Webcam-based Mediapipe hand-tracking → serial control packet
#
# Packet format:
#   A  idxAB  midAB  ringAB  pinkyAB  thumbFLX  idxFLX  midFLX  ringFLX  pinkyFLX  thumbAB
#   Example:
#   A -12  8  15  0  35  42  60  75  80  -5
#
# Press ESC to quit; keys 1-4 flip ab/ad sign for Index-Pinky

import cv2, mediapipe as mp, numpy as np, math, time, serial, sys

# ───────── USER CONFIG ─────────
SERIAL_PORT = "COM11"        # Windows "COMx", Linux "/dev/ttyUSB0"; set None to disable
BAUD_RATE   = 1_000_000      # 1 Mbps default

FLEX_MIN  = [0,   0,   0,   0,   0]
FLEX_MAX  = [270, 270, 270, 270, 270]
FLEX_TLO  = [20,  15,  15,  15,  15]
FLEX_THI  = [50, 100, 100, 100,  90]
FLEX_REV  = [False]*5

AB_MIN   = [-30]*4
AB_MAX   = [ 30]*4
AB_REV   = [True]*4           # start with reversed ab/ad for Index-Pinky

OFFSET_DEG     = 90.0         # +90° so “straight” = 90
AB_DISABLE_TOL = 2.0          # flex ≈90° ±2 → ab/ad forced 0
# ───────────────────────────────

unit  = lambda v,eps=1e-6: v / (np.linalg.norm(v,axis=-1,keepdims=True)+eps)
clamp = lambda x,lo,hi: lo if x<lo else hi if x>hi else x

def safe_angle(v1, v2):
    v1u, v2u = unit(v1).flatten(), unit(v2).flatten()
    cos = max(-1.0, min(1.0, np.dot(v1u, v2u)))
    return math.degrees(math.acos(cos))

def project(v, n):
    n = unit(n);  return v - np.sum(v*n,axis=-1,keepdims=True)*n

def signed_angle(v1, v2, n):
    v1u, v2u = unit(v1), unit(v2)
    dot  = np.clip(np.sum(v1u*v2u,axis=-1), -1.0, 1.0)
    ang  = np.degrees(np.arccos(dot))
    sign = np.sign(np.sum(np.cross(v1u, v2u)*n, axis=-1))
    return ang*sign

FINGER_MCP   = [5, 9, 13, 17]                  # Index-Pinky MCP joints
FINGER_PIP   = [6,10,14,18]
FLEX_FINGERS = [(1,4), (5,8), (9,12), (13,16), (17,20)]  # Thumb-Pinky tip pairs

def abduction_angles(pts):
    wrist, idx, pky = pts[0], pts[5], pts[17]
    n   = unit(np.cross(idx-wrist, pky-wrist))
    ref = project(idx - pky, n)
    ab  = []
    for k,(mcp,pip) in enumerate(zip(FINGER_MCP, FINGER_PIP)):
        vec = project(pts[pip]-pts[mcp], n)
        ang = signed_angle(vec, ref, n) + OFFSET_DEG
        if AB_REV[k]: ang = -ang
        ab.append(clamp(ang, AB_MIN[k], AB_MAX[k]))
    thumb_vec = project(pts[3]-pts[2], n)
    thumb_ang = signed_angle(thumb_vec, ref, n) + OFFSET_DEG
    thumb_ang = clamp(thumb_ang, -30, 30)
    return ab, thumb_ang

def flexion_angles(pts):
    wrist = pts[0]
    vals  = []
    for k,(mcp,tip) in enumerate(FLEX_FINGERS):
        ang = safe_angle(pts[mcp]-wrist, pts[tip]-pts[mcp])
        if FLEX_REV[k]: ang = -ang
        ang = clamp(ang, FLEX_MIN[k], FLEX_MAX[k])
        if ang <= FLEX_TLO[k]: ang = FLEX_MIN[k]
        if ang >= FLEX_THI[k]: ang = FLEX_MAX[k]
        vals.append(ang)
    return vals

ser = None
try:
    if SERIAL_PORT:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Serial open → {SERIAL_PORT} @ {BAUD_RATE}")
except serial.SerialException:
    print(f"[WARN] Serial port {SERIAL_PORT} unavailable → packets disabled.", file=sys.stderr)

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(False, 1, 1, 0.7, 0.7)
draw     = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)      # default webcam
if not cap.isOpened():
    sys.exit("[ERROR] Cannot open webcam")

frame_cnt, t0 = 0, time.time()

def window_title(flags):
    tag = '|'.join(f'{n}{"-" if r else "+"}' for n,r in zip("IMRP", flags))
    return f'Flex / AbAd  [{tag}]  –  ESC quit, 1-4 flip sign'

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Camera frame read failed")
        break
    h, w = frame.shape[:2]

    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        for hl in res.multi_hand_landmarks:
            pts = np.array([[lm.x*w, lm.y*h, lm.z*w] for lm in hl.landmark])
            abd, thumb_ab = abduction_angles(pts)
            flex          = flexion_angles(pts)

            for i, flex_deg in enumerate(flex[1:]):  # Index-Pinky
                if abs(flex_deg-90) <= AB_DISABLE_TOL:
                    abd[i] = 0.0

            for (mcp,val) in zip(FINGER_MCP, abd):
                x,y = int(pts[mcp][0]), int(pts[mcp][1])
                cv2.putText(frame, f'{val:+.1f}', (x-22,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
            for ((_,tip),val) in zip(FLEX_FINGERS, flex):
                x,y = int(pts[tip][0]), int(pts[tip][1])
                cv2.putText(frame, f'{val:5.1f}', (x-18,y-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),2)

            draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            packet = [
                int(round(abd[0])), int(round(abd[1])), int(round(abd[2])), int(round(abd[3])),
                int(round(flex[0])), int(round(flex[1])), int(round(flex[2])), int(round(flex[3])), int(round(flex[4])),
                int(round(thumb_ab))
            ]
            if ser: ser.write(("A "+" ".join(map(str,packet))+"\n").encode())

    frame_cnt += 1
    fps = frame_cnt / (time.time()-t0)
    cv2.putText(frame,f'Frames:{frame_cnt}  FPS:{fps:4.1f}',(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)

    cv2.setWindowTitle('main', window_title(AB_REV))
    cv2.imshow('main', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    if key in (49,50,51,52): AB_REV[key-49] = not AB_REV[key-49]

cap.release()
if ser: ser.close()
cv2.destroyAllWindows()
