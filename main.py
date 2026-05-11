import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound
from ultralytics import YOLO

# ---------------------- CONFIG ----------------------
CALIBRATE_SECONDS = 6.0
EMA_ALPHA = 0.35

RES_W = 1280
RES_H = 720

PIXEL_YAWN_THR = 28

# HOLD TIMES
EYE_HOLD_TIME = 2.0
YAWN_HOLD_TIME = 2.0
HEAD_HOLD_TIME = 3.0
HEAD_DOWN_HOLD_TIME = 3.0
FACE_LOST_TIME = 2.5

# YOLO
YOLO_CONF = 0.45
YOLO_MODEL = "yolov8n.pt"

# ---------------------- BEEP CONTROL ----------------------
beeping = {
    "sleepy": False,
    "yawn": False,
    "head_left": False,
    "head_right": False,
    "head_down": False,
    "face_lost": False,
    "phone": False,
    "bottle": False,
    "distract": False
}

def pulse_beep(alert_key):
    while beeping[alert_key]:
        winsound.Beep(2000, 400)
        time.sleep(0.4)

# ------------------------------------------------------------------
# Initialize models
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

yolo_model = YOLO(YOLO_MODEL)

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def eye_ear(lm, idx, w, h):
    pts = [(lm[i].x*w, lm[i].y*h) for i in idx]
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A+B)/(2*C+1e-6)

def outer_mar(lm, w, h):
    top = (lm[13].x*w, lm[13].y*h)
    bottom = (lm[14].x*w, lm[14].y*h)
    left = (lm[78].x*w, lm[78].y*h)
    right = (lm[308].x*w, lm[308].y*h)
    return dist(top,bottom)/(dist(left,right)+1e-6)

def inner_mar(lm, w, h):
    top = (lm[13].x*w, lm[13].y*h)
    bottom = (lm[14].x*w, lm[14].y*h)
    left = (lm[82].x*w, lm[82].y*h)
    right = (lm[312].x*w, lm[312].y*h)
    return dist(top,bottom)/(dist(left,right)+1e-6)

# ---------------------- SKIN DETECTION ----------------------
def skin_present(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0,30,60])
    upper = np.array([20,150,255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) > 3000

# ---------------------- CAMERA ----------------------
cap = cv2.VideoCapture(0)
cap.set(3, RES_W)
cap.set(4, RES_H)

calib_start = time.time()
calib_ear_vals = []
calib_mar_vals = []
calib_nose_vals = []
calib_facecy_vals = []

ear_ema = None
mar_ema = None

baseline_nose_y = None
baseline_face_cy = None

prev_nose_y = None
nose_vel_ema = 0.0
NOSE_VEL_ALPHA = 0.35

eye_closed_since = None
yawn_since = None
head_left_since = None
head_right_since = None
head_down_since = None
face_lost_since = None

head_dev_ema = 0

DISPLAY_TIME = 2.0
display_eye_until = 0
display_yawn_until = 0
display_head_left_until = 0
display_head_right_until = 0
display_head_down_until = 0
display_face_lost_until = 0

display_phone_until = 0
display_bottle_until = 0
display_distract_until = 0

print("Calibration started... Keep your head straight for {} seconds.".format(CALIBRATE_SECONDS))

def expand_bbox(x1,y1,x2,y2, w, h, pad=0.20):
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw/2.0
    cy = y1 + bh/2.0
    bwp = bw * (1 + pad)
    bhp = bh * (1 + pad)
    nx1 = max(0, int(cx - bwp/2.0))
    ny1 = max(0, int(cy - bhp/2.0))
    nx2 = min(w-1, int(cx + bwp/2.0))
    ny2 = min(h-1, int(cy + bhp/2.0))
    return nx1, ny1, nx2, ny2

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(1, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    areaB = max(1, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    iou = interArea / float(areaA + areaB - interArea + 1e-6)
    return iou, interArea

# ========================================================================
#                           MAIN LOOP
# ========================================================================
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.time()

        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        yolo_results = yolo_model(frame, conf=YOLO_CONF, verbose=False)[0]

        # --------------------------------------------------------
        # FACE NOT DETECTED
        # --------------------------------------------------------
        if not face_results.multi_face_landmarks:
            if skin_present(frame):
                face_lost_since = None
            else:
                if face_lost_since is None:
                    face_lost_since = now
                else:
                    if now - face_lost_since > FACE_LOST_TIME:
                        if not beeping["face_lost"]:
                            beeping["face_lost"] = True
                            threading.Thread(target=pulse_beep, args=("face_lost",), daemon=True).start()
                        display_face_lost_until = now + DISPLAY_TIME

            if now < display_face_lost_until:
                cv2.putText(frame,"FACE NOT DETECTED",(100,100),
                            cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

            cv2.imshow("Drowsiness", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        else:
            if beeping["face_lost"]:
                beeping["face_lost"] = False
            face_lost_since = None

        lm = face_results.multi_face_landmarks[0].landmark

        # FACE width
        face_width = abs(lm[234].x - lm[454].x)

        # EAR
        ear = (eye_ear(lm, LEFT_EYE, w, h) + eye_ear(lm, RIGHT_EYE, w, h)) / 2

        # MAR
        mar_out = outer_mar(lm, w, h)
        mar_in = inner_mar(lm, w, h)
        mar = (mar_out + mar_in) / 2

        pixel_gap = dist((lm[13].x*w, lm[13].y*h),(lm[14].x*w, lm[14].y*h))
        jaw_drop = dist((lm[1].x*w, lm[1].y*h),(lm[152].x*w, lm[152].y*h))

        # --------------------------------------------------------
        # CALIBRATION
        # --------------------------------------------------------
        if ear_ema is None:
            if now - calib_start <= CALIBRATE_SECONDS:
                calib_ear_vals.append(ear)
                calib_mar_vals.append(mar)

                nose_y_c = lm[1].y
                face_cy_c = (lm[1].y + lm[159].y + lm[386].y) / 3.0
                calib_nose_vals.append(nose_y_c)
                calib_facecy_vals.append(face_cy_c)

                cv2.putText(frame,"CALIBRATING... Keep head straight",(20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),2)
                cv2.imshow("Drowsiness", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            else:
                baseline_ear = np.median(calib_ear_vals)
                baseline_mar = np.median(calib_mar_vals)

                EAR_THR = baseline_ear * 0.55
                MAR_OUT_THR = baseline_mar * 1.6
                MAR_IN_THR  = baseline_mar * 1.75

                baseline_nose_y = float(np.median(calib_nose_vals))
                baseline_face_cy = float(np.median(calib_facecy_vals))

                NOSE_DROP_RATIO = 0.045
                FACE_CY_DROP_RATIO = 0.035
                NOSE_VEL_THRESHOLD = 0.004

                ear_ema = baseline_ear
                mar_ema = baseline_mar
                continue

        # smoothing
        ear_ema = EMA_ALPHA * ear + (1 - EMA_ALPHA) * ear_ema
        mar_ema = EMA_ALPHA * mar + (1 - EMA_ALPHA) * mar_ema

        # --------------------------------------------------------
        # SLEEPY
        # --------------------------------------------------------
        if ear_ema < EAR_THR:
            if eye_closed_since is None:
                eye_closed_since = now
            else:
                if now - eye_closed_since >= EYE_HOLD_TIME:
                    if not beeping["sleepy"]:
                        beeping["sleepy"] = True
                        threading.Thread(target=pulse_beep, args=("sleepy",), daemon=True).start()
                    display_eye_until = now + DISPLAY_TIME
        else:
            eye_closed_since = None
            beeping["sleepy"] = False

        # --------------------------------------------------------
        # YAWNING
        # --------------------------------------------------------
        yawn_condition = (
            pixel_gap >= PIXEL_YAWN_THR and
            mar_out >= MAR_OUT_THR and
            mar_in >= MAR_IN_THR and
            jaw_drop >= face_width * 0.15
        )

        if yawn_condition:
            if yawn_since is None:
                yawn_since = now
            else:
                if now - yawn_since >= YAWN_HOLD_TIME:
                    if not beeping["yawn"]:
                        beeping["yawn"] = True
                        threading.Thread(target=pulse_beep, args=("yawn",), daemon=True).start()
                    display_yawn_until = now + DISPLAY_TIME
        else:
            yawn_since = None
            beeping["yawn"] = False

        # --------------------------------------------------------
        # HEAD LEFT / RIGHT
        # --------------------------------------------------------
        nose_x  = lm[1].x
        left_x  = lm[234].x
        right_x = lm[454].x

        face_center = (left_x + right_x) / 2
        dev = (nose_x - face_center) / (abs(right_x - left_x) + 1e-6)

        head_dev_ema = 0.20 * dev + 0.80 * head_dev_ema

        if head_dev_ema < -0.18:
            if head_left_since is None:
                head_left_since = now
            else:
                if now - head_left_since >= HEAD_HOLD_TIME:
                    if not beeping["head_left"]:
                        beeping["head_left"] = True
                        threading.Thread(target=pulse_beep, args=("head_left",), daemon=True).start()
                    display_head_left_until = now + DISPLAY_TIME
        else:
            head_left_since = None
            beeping["head_left"] = False

        if head_dev_ema > 0.18:
            if head_right_since is None:
                head_right_since = now
            else:
                if now - head_right_since >= HEAD_HOLD_TIME:
                    if not beeping["head_right"]:
                        beeping["head_right"] = True
                        threading.Thread(target=pulse_beep, args=("head_right",), daemon=True).start()
                    display_head_right_until = now + DISPLAY_TIME
        else:
            head_right_since = None
            beeping["head_right"] = False

        # --------------------------------------------------------
        # HEAD DOWN
        # --------------------------------------------------------
        forehead_y = lm[10].y
        eye_y      = (lm[159].y + lm[386].y) / 2
        chin_y     = lm[152].y
        nose_y     = lm[1].y

        face_height_norm = abs(chin_y - forehead_y) + 1e-6

        # Cue1
        cue1 = False
        if baseline_nose_y is not None:
            nose_drop_norm = (nose_y - baseline_nose_y) / face_height_norm
            cue1 = nose_drop_norm > NOSE_DROP_RATIO

        # Cue2
        face_cy = (lm[1].y + lm[159].y + lm[386].y) / 3.0
        cue2 = False
        if baseline_face_cy is not None:
            facecy_drop_norm = (face_cy - baseline_face_cy) / face_height_norm
            cue2 = facecy_drop_norm > FACE_CY_DROP_RATIO

        # Cue3
        if prev_nose_y is None:
            prev_nose_y = nose_y
        inst_vel = (nose_y - prev_nose_y) / face_height_norm
        nose_vel_ema = NOSE_VEL_ALPHA * inst_vel + (1 - NOSE_VEL_ALPHA) * nose_vel_ema
        prev_nose_y = nose_y
        cue3 = nose_vel_ema > NOSE_VEL_THRESHOLD

        cue_sum = int(cue1) + int(cue2) + int(cue3)
        head_down_detected = cue_sum >= 2

        if head_down_detected:
            if head_down_since is None:
                head_down_since = now
            else:
                if now - head_down_since >= HEAD_HOLD_TIME:
                    if not beeping["head_down"]:
                        beeping["head_down"] = True
                        threading.Thread(target=pulse_beep, args=("head_down",), daemon=True).start()
                    display_head_down_until = now + DISPLAY_TIME
        else:
            head_down_since = None
            beeping["head_down"] = False

        # --------------------------------------------------------
        # HAND ZONES
        # --------------------------------------------------------
        hand_boxes = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x1 = max(0, min(xs))
                y1 = max(0, min(ys))
                x2 = min(w-1, max(xs))
                y2 = min(h-1, max(ys))
                hx1, hy1, hx2, hy2 = expand_bbox(x1,y1,x2,y2, w, h, pad=0.5)
                hand_boxes.append((hx1,hy1,hx2,hy2))
                cv2.rectangle(frame, (hx1,hy1), (hx2,hy2), (0,200,0), 2)

        else:
            hand_boxes = []

        # --------------------------------------------------------
        # YOLO + HAND OVERLAP LOGIC
        # --------------------------------------------------------
        detected_phone = False
        detected_bottle = False
        detected_other = False

        yolo_names = yolo_model.names

        for box in yolo_results.boxes:

            try:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_idx = int(box.cls[0].cpu().numpy())
            except:
                vals = box.xyxy[0]
                xyxy = [int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])]
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])

            if conf < YOLO_CONF:
                continue

            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cls_name = yolo_names.get(cls_idx, str(cls_idx))

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            if len(hand_boxes) == 0:
                continue

            obj_box = (x1,y1,x2,y2)
            overlapped = False
            for hb in hand_boxes:
                iou, interArea = bbox_iou(obj_box, hb)
                if iou > 0.03 or interArea > 2000:
                    overlapped = True
                    break
            if not overlapped:
                continue

            nm = cls_name.lower()

            # ===================================================
            #          ★★★ PHONE FALSE-POSITIVE FIX ★★★
            # ===================================================
            obj_w = (x2 - x1)
            obj_h = (y2 - y1)
            if obj_h > 0 and obj_w > 0:
                aspect_ratio = obj_h / obj_w
            else:
                aspect_ratio = 0

            is_phone_shape = 1.2 < aspect_ratio < 3.0
            # ===================================================

            # PHONE
            if (("cell phone" in nm or "phone" in nm or "mobile" in nm)) and is_phone_shape:
                detected_phone = True

            # BOTTLE
            elif "bottle" in nm:
                detected_bottle = True

            # OTHER DISTRACTION
            else:
                if nm not in ["person", "car", "truck"]:
                    detected_other = True

        # --------------------------------------------------------
        # ALERT LOGIC
        # --------------------------------------------------------
        if detected_phone:
            if not beeping["phone"]:
                beeping["phone"] = True
                threading.Thread(target=pulse_beep, args=("phone",), daemon=True).start()
            display_phone_until = now + DISPLAY_TIME
        else:
            beeping["phone"] = False

        if detected_bottle:
            if not beeping["bottle"]:
                beeping["bottle"] = True
                threading.Thread(target=pulse_beep, args=("bottle",), daemon=True).start()
            display_bottle_until = now + DISPLAY_TIME
        else:
            beeping["bottle"] = False

        if detected_other:
            if not beeping["distract"]:
                beeping["distract"] = True
                threading.Thread(target=pulse_beep, args=("distract",), daemon=True).start()
            display_distract_until = now + DISPLAY_TIME
        else:
            beeping["distract"] = False

        # --------------------------------------------------------
        # DISPLAY ALERT TEXTS
        # --------------------------------------------------------
        if now < display_eye_until:
            cv2.putText(frame,"SLEEPY!",(250,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

        if now < display_yawn_until:
            cv2.putText(frame,"YAWNING!",(250,160),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

        if now < display_head_left_until:
            cv2.putText(frame,"HEAD LEFT!",(200,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

        if now < display_head_right_until:
            cv2.putText(frame,"HEAD RIGHT!",(200,240),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

        if now < display_head_down_until:
            cv2.putText(frame,"HEAD DOWN!",(200,280),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

        if now < display_face_lost_until:
            cv2.putText(frame,"FACE NOT DETECTED",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

        if now < display_phone_until:
            cv2.putText(frame,"PHONE DETECTED",(200,330),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        if now < display_bottle_until:
            cv2.putText(frame,"BOTTLE DETECTED",(200,370),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        if now < display_distract_until:
            cv2.putText(frame,"DISTRACTION!",(200,410),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        cv2.imshow("Drowsiness", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    for k in beeping:
        beeping[k] = False
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()
