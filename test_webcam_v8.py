import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os

# ============================================================
# PATH
# ============================================================
MODEL_PATH = 'D:/TEST1/alphabet_mlp_v8.pth'

CLASSES        = [str(i) for i in range(10)] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FEATURE_SIZE   = 180
CONF_THRESHOLD = 0.45

# ============================================================
# LANDMARK INDICES
# ============================================================
FINGERTIPS   = [4, 8, 12, 16, 20]
BONE_TRIPLES = [
    (1, 2, 3),    (2, 3, 4),
    (5, 6, 7),    (6, 7, 8),
    (9, 10, 11),  (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
]

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def normalize_hand(hand_63):
    if np.all(hand_63 == 0):
        return hand_63
    hand  = hand_63.reshape(21, 3)
    wrist = hand[0].copy()
    hand  = hand - wrist
    scale = np.linalg.norm(hand[9])
    if scale > 0:
        hand = hand / scale
    return hand.flatten()

def compute_derived_features(norm_xyz_63):
    is_empty = np.all(norm_xyz_63 == 0)
    pts = norm_xyz_63.reshape(21, 3)

    angles = np.zeros(10)
    if not is_empty:
        for i, (a, b, c) in enumerate(BONE_TRIPLES):
            v1 = pts[a] - pts[b]
            v2 = pts[c] - pts[b]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles[i] = np.arccos(cos_a)

    tip_dists = np.zeros(10)
    if not is_empty:
        idx = 0
        for i in range(5):
            for j in range(i+1, 5):
                tip_dists[idx] = np.linalg.norm(
                    pts[FINGERTIPS[i]] - pts[FINGERTIPS[j]])
                idx += 1

    thumb_dists = np.zeros(4)
    if not is_empty:
        for i, tip in enumerate(FINGERTIPS[1:]):
            thumb_dists[i] = np.linalg.norm(pts[4] - pts[tip])

    palm_normal = np.zeros(3)
    if not is_empty:
        v1 = pts[5]  - pts[0]
        v2 = pts[17] - pts[0]
        cross  = np.cross(v1, v2)
        norm_c = np.linalg.norm(cross)
        if norm_c > 1e-6:
            palm_normal = cross / norm_c

    return np.concatenate([angles, tip_dists, thumb_dists, palm_normal])

def build_full_feature(right_raw_63, left_raw_63):
    right_norm = normalize_hand(right_raw_63)
    left_norm  = normalize_hand(left_raw_63)
    right_der  = compute_derived_features(right_norm)
    left_der   = compute_derived_features(left_norm)
    return np.concatenate([right_norm, right_der,
                           left_norm,  left_der]).astype(np.float32)

# ============================================================
# MODEL
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_size=180, num_classes=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 512),        nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# LOAD MODEL
# ============================================================
DEVICE = torch.device('cpu')
model  = MLP(FEATURE_SIZE, 36).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded!")
    print(f"Running on CPU")
else:
    print(f"Model file not found at: {MODEL_PATH}")
    print("Make sure alphabet_mlp_v8.pth is in D:/TEST1/")
    input("Press Enter to exit...")
    exit()

# ============================================================
# TTA
# ============================================================
TTA_RUNS  = 7
TTA_NOISE = 0.003

def predict_with_tta(feats):
    all_probs = []
    tensor = torch.tensor(feats, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(tensor.unsqueeze(0))
        all_probs.append(torch.softmax(out, dim=1).squeeze())
        for _ in range(TTA_RUNS - 1):
            noisy = tensor + torch.randn_like(tensor) * TTA_NOISE
            out   = model(noisy.unsqueeze(0))
            all_probs.append(torch.softmax(out, dim=1).squeeze())
    avg      = torch.stack(all_probs).mean(dim=0)
    conf, idx = torch.max(avg, dim=0)
    return CLASSES[idx.item()], conf.item(), avg.cpu().numpy()

# ============================================================
# WEIGHTED VOTER
# ============================================================
class WeightedVoter:
    def __init__(self, maxlen=20):
        self.queue = deque(maxlen=maxlen)

    def add(self, char, conf):
        self.queue.append((char, conf))

    def result(self):
        if not self.queue:
            return None, 0.0
        scores = {}
        for i, (char, conf) in enumerate(self.queue):
            w = (i + 1) * conf
            scores[char] = scores.get(char, 0) + w
        best  = max(scores, key=scores.get)
        total = sum(scores.values())
        return best, scores[best] / total if total > 0 else 0

voter = WeightedVoter(maxlen=20)

# ============================================================
# MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ============================================================
# WEBCAM LOOP
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Camera started. Press Q to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame   = cv2.flip(frame, 1)
    h, w    = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res     = detector.process(img_rgb)

    display_text = "Show your hand..."
    conf_text    = ""
    top3_text    = ""
    color        = (0, 0, 200)
    bg_color     = (30, 30, 30)

    if res.multi_hand_landmarks:
        lh          = np.zeros(63)
        rh          = np.zeros(63)
        num_hands   = len(res.multi_hand_landmarks)
        skeleton_ok = True

        for i, landmarks in enumerate(res.multi_hand_landmarks):
            side     = res.multi_handedness[i].classification[0].label
            det_conf = res.multi_handedness[i].classification[0].score
            if det_conf < 0.6:
                skeleton_ok = False

            pts = np.array([[lm.x, lm.y, lm.z]
                            for lm in landmarks.landmark]).flatten()
            if num_hands == 1:
                rh = pts
            elif side == 'Left':
                rh = pts
            else:
                lh = pts

            draw_color = (0, 255, 0) if skeleton_ok else (0, 165, 255)
            mp_draw.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=draw_color, thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

        if skeleton_ok:
            feats = build_full_feature(rh, lh)
            char, conf, probs = predict_with_tta(feats)

            voter.add(char, conf)
            stable_char, stable_score = voter.result()

            top3_idx   = np.argsort(probs)[::-1][:3]
            top3_parts = [f"{CLASSES[j]}:{probs[j]*100:.0f}%" for j in top3_idx]
            top3_text  = "   ".join(top3_parts)

            if conf >= CONF_THRESHOLD:
                display_text = f"SIGN: {stable_char}"
                conf_text    = f"Conf: {conf*100:.1f}%   Stable: {stable_score*100:.0f}%"
                color        = (0, 255, 0)
                bg_color     = (0, 55, 0)
            else:
                display_text = f"Unsure: {char}"
                conf_text    = f"Conf: {conf*100:.1f}%  (need {CONF_THRESHOLD*100:.0f}%+)"
                color        = (0, 165, 255)
                bg_color     = (0, 40, 60)
        else:
            display_text = "Bad skeleton - adjust hand angle"
            color        = (0, 0, 200)

    cv2.rectangle(frame, (20, 20), (w - 20, 130), bg_color, -1)
    cv2.rectangle(frame, (20, 20), (w - 20, 130), color, 2)
    cv2.putText(frame, display_text, (35, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
    if conf_text:
        cv2.putText(frame, conf_text, (35, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    if top3_text:
        cv2.rectangle(frame, (20, 138), (w - 20, 172), (40, 40, 40), -1)
        cv2.putText(frame, f"Top 3:  {top3_text}", (35, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 1)

    cv2.putText(frame, "Project ATHENA  |  ISL Alphabet Recognition",
                (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    cv2.imshow("Project ATHENA", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()