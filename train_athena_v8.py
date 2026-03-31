import cv2
import mediapipe as mp
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from multiprocessing import Pool, cpu_count

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR          = '/mnt/d/power/projects/ATHENA'
ISLRTC_PATH       = os.path.join(BASE_DIR, 'data/islrtc')
PREKSHAPALVA_PATH = os.path.join(BASE_DIR, 'data/prekshapalva')
MODEL_PATH        = os.path.join(BASE_DIR, 'models/alphabet_mlp_v8.pth')
CACHE_X           = os.path.join(BASE_DIR, 'data/cache_X_v7.npy')  # reuse v7 cache, it's correct
CACHE_Y           = os.path.join(BASE_DIR, 'data/cache_Y_v7.npy')

# Features: 63 xyz + 10 angles + 10 tip dists + 4 thumb dists + 3 palm = 90 per hand x2 = 180
FEATURE_SIZE = 180

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES      = [str(i) for i in range(10)] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

print(f"Using device: {DEVICE}")
print(f"Total classes: {len(CLASSES)}")
print(f"Feature size: {FEATURE_SIZE}")

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
# WORKER INIT
# ============================================================
worker_hands = None

def init_worker():
    global worker_hands
    worker_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

# ============================================================
# NORMALIZATION
# ============================================================
def normalize_hand(hand_63):
    """Normalize hand: subtract wrist, scale by middle finger MCP distance."""
    if np.all(hand_63 == 0):
        return hand_63
    hand  = hand_63.reshape(21, 3)
    wrist = hand[0].copy()
    hand  = hand - wrist
    scale = np.linalg.norm(hand[9])
    if scale > 0:
        hand = hand / scale
    return hand.flatten()

# ============================================================
# COMPUTE DERIVED FEATURES FROM ALREADY-NORMALIZED xyz
# KEY FIX: This function takes ALREADY-NORMALIZED xyz (no re-normalization)
# ============================================================
def compute_derived_features(norm_xyz_63):
    """
    Compute angles, distances, palm normal from already-normalized xyz.
    Input:  63 floats (already normalized, 21 points x 3)
    Output: 27 floats (10 angles + 10 tip dists + 4 thumb dists + 3 palm normal)
    """
    is_empty = np.all(norm_xyz_63 == 0)
    pts = norm_xyz_63.reshape(21, 3)

    # Bone angles (10)
    angles = np.zeros(10)
    if not is_empty:
        for i, (a, b, c) in enumerate(BONE_TRIPLES):
            v1 = pts[a] - pts[b]
            v2 = pts[c] - pts[b]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles[i] = np.arccos(cos_a)

    # Fingertip pairwise distances (10)
    tip_dists = np.zeros(10)
    if not is_empty:
        idx = 0
        for i in range(5):
            for j in range(i+1, 5):
                tip_dists[idx] = np.linalg.norm(
                    pts[FINGERTIPS[i]] - pts[FINGERTIPS[j]])
                idx += 1

    # Thumb to each fingertip (4)
    thumb_dists = np.zeros(4)
    if not is_empty:
        for i, tip in enumerate(FINGERTIPS[1:]):
            thumb_dists[i] = np.linalg.norm(pts[4] - pts[tip])

    # Palm normal (3)
    palm_normal = np.zeros(3)
    if not is_empty:
        v1 = pts[5]  - pts[0]
        v2 = pts[17] - pts[0]
        cross  = np.cross(v1, v2)
        norm_c = np.linalg.norm(cross)
        if norm_c > 1e-6:
            palm_normal = cross / norm_c

    return np.concatenate([angles, tip_dists, thumb_dists, palm_normal])  # 27

def build_full_feature(right_raw_63, left_raw_63):
    """
    Build 180 features from raw (un-normalized) xyz.
    Used only during image processing / cache building.
    """
    right_norm = normalize_hand(right_raw_63)   # 63
    left_norm  = normalize_hand(left_raw_63)    # 63
    right_der  = compute_derived_features(right_norm)  # 27
    left_der   = compute_derived_features(left_norm)   # 27
    return np.concatenate([right_norm, right_der,
                           left_norm,  left_der]).astype(np.float32)  # 180

# ============================================================
# PROCESS SINGLE IMAGE
# ============================================================
def process_image(args):
    path, replace_bg, label = args
    global worker_hands
    try:
        img = cv2.imread(path)
        if img is None:
            return None

        if replace_bg:
            mask      = np.all(img < 30, axis=2)
            bg_color  = [random.randint(80, 220)] * 3
            img[mask] = bg_color

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result  = worker_hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            return None

        left_raw  = np.zeros(63)
        right_raw = np.zeros(63)
        num_hands = len(result.multi_hand_landmarks)

        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            if i >= len(result.multi_handedness):
                break
            side = result.multi_handedness[i].classification[0].label
            pts  = np.array([[lm.x, lm.y, lm.z]
                              for lm in hand_landmarks.landmark]).flatten()
            if num_hands == 1:
                right_raw = pts
            elif side == 'Left':
                left_raw = pts
            else:
                right_raw = pts

        feats = build_full_feature(right_raw, left_raw)

        if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
            return None

        return (feats, label)
    except:
        return None

# ============================================================
# LOAD DATASET
# ============================================================
def load_dataset():
    if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
        print("Loading from cache...")
        X = np.load(CACHE_X)
        y = np.load(CACHE_Y)
        print(f"Loaded {len(X)} samples | Feature size: {X.shape[1]}")
        if X.shape[1] != FEATURE_SIZE:
            print(f"⚠️  Cache has {X.shape[1]} features but expected {FEATURE_SIZE}.")
            print("    Delete cache_X_v7.npy and cache_Y_v7.npy and rerun.")
            exit()
        return X, y

    DATA_SOURCES = []
    for path, replace_bg, name in [
        (ISLRTC_PATH,       False, 'islrtc'),
        (PREKSHAPALVA_PATH, False, 'prekshapalva'),
    ]:
        if os.path.exists(path):
            DATA_SOURCES.append((path, replace_bg))
            print(f"✅ Found: {name}")
        else:
            print(f"⚠️  Skipped (not found): {name}")

    tasks = []
    for base_path, replace_bg in DATA_SOURCES:
        for cls in CLASSES:
            folder = os.path.join(base_path, cls)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        tasks.append((
                            os.path.join(folder, f),
                            replace_bg,
                            CLASS_TO_IDX[cls]
                        ))

    num_workers = 4 if os.path.exists('/proc/version') else cpu_count()
    print(f"\nTotal images: {len(tasks)}")
    print(f"Using {num_workers} workers...")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_image, tasks, chunksize=50),
            total=len(tasks),
            desc="Extracting keypoints"
        ))

    valid  = [r for r in results if r is not None]
    failed = len(results) - len(valid)
    X = np.array([r[0] for r in valid], dtype=np.float32)
    y = np.array([r[1] for r in valid], dtype=np.int64)

    print(f"Success: {len(valid)} | Failed: {failed}")
    print(f"Success rate: {len(valid)/len(results)*100:.1f}%")

    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    print("Cache saved!")

    return X, y

# ============================================================
# AUGMENTATION — v8 FIXED
# ✅ No double normalization
# Cache stores: [right_norm(63) | right_der(27) | left_norm(63) | left_der(27)]
# Augment only the norm_xyz parts, then recompute derived features fresh
# ============================================================
def augment_keypoints(feat):
    feat = feat.copy()

    # Extract the 4 segments from the 180-feature vector
    right_norm = feat[0:63].copy()      # normalized xyz, right
    # feat[63:90] = right derived — will recompute
    left_norm  = feat[90:153].copy()    # normalized xyz, left
    # feat[153:180] = left derived — will recompute

    # ── 1. Rotation in xy plane ──────────────────────────
    angle = random.uniform(-20, 20)
    rad   = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rot   = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    for xyz in [right_norm, left_norm]:
        hand = xyz.reshape(21, 3)
        if not np.all(hand == 0):
            hand[:, :2] = (rot @ hand[:, :2].T).T
            xyz[:] = hand.flatten()

    # ── 2. Scale ─────────────────────────────────────────
    scale = random.uniform(0.80, 1.20)
    for xyz in [right_norm, left_norm]:
        if not np.all(xyz == 0):
            xyz *= scale

    # ── 3. Wrist jitter (small translation) ──────────────
    for xyz in [right_norm, left_norm]:
        hand = xyz.reshape(21, 3)
        if not np.all(hand == 0):
            hand += np.random.normal(0, 0.02, 3)
            xyz[:] = hand.flatten()

    # ── 4. Mirror flip ────────────────────────────────────
    if random.random() < 0.3:
        for xyz in [right_norm, left_norm]:
            hand = xyz.reshape(21, 3)
            if not np.all(hand == 0):
                hand[:, 0] = -hand[:, 0]
                xyz[:] = hand.flatten()

    # ── 5. Finger dropout ─────────────────────────────────
    if random.random() < 0.2:
        finger_indices = [
            [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]
        ]
        drop_finger = random.choice(finger_indices)
        for xyz in [right_norm, left_norm]:
            hand = xyz.reshape(21, 3)
            if not np.all(hand == 0):
                for fi in drop_finger:
                    hand[fi] = 0
                xyz[:] = hand.flatten()

    # ── Recompute derived features from augmented xyz ─────
    # ✅ KEY FIX: compute_derived_features takes already-normalized xyz
    #    so there is NO second normalize_hand call here
    right_der = compute_derived_features(right_norm)   # 27
    left_der  = compute_derived_features(left_norm)    # 27

    new_feat = np.concatenate([right_norm, right_der, left_norm, left_der])

    # ── 6. Tiny noise on final features ───────────────────
    new_feat += np.random.normal(0, 0.005, new_feat.shape)

    return new_feat.astype(np.float32)

# ============================================================
# DATASET
# ============================================================
class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        if self.augment:
            x = augment_keypoints(x)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

# ============================================================
# MODEL
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_size=180, num_classes=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# CLASS WEIGHTS
# ============================================================
def compute_class_weights(y, num_classes=36):
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# ============================================================
# TRAINING
# ============================================================
def train():
    X, y = load_dataset()
    print(f"\nData shape: {X.shape}")

    print("\nClass distribution:")
    for i, cls in enumerate(CLASSES):
        count = np.sum(y == i)
        bar   = "█" * (count // 200)
        print(f"  {cls}: {count} {bar}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(
        SignDataset(X_train, y_train, augment=True),
        batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        SignDataset(X_val, y_val, augment=False),
        batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        SignDataset(X_test, y_test, augment=False),
        batch_size=256, shuffle=False, num_workers=2)

    model     = MLP(FEATURE_SIZE, 36).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    class_weights = compute_class_weights(y_train)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-5)

    best_val_loss    = float('inf')
    patience_counter = 0
    PATIENCE         = 20

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Training on {DEVICE}...")
    print("=" * 60)

    for epoch in range(200):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb   = xb.to(DEVICE), yb.to(DEVICE)
                out       = model(xb)
                val_loss += criterion(out, yb).item()
                correct  += (out.argmax(1) == yb).sum().item()
                total    += yb.size(0)
        val_loss /= len(val_loader)
        val_acc   = correct / total * 100

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"Acc: {val_acc:.1f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"           ✅ Saved! Best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    correct = total = 0
    class_correct = np.zeros(36)
    class_total   = np.zeros(36)

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb  = xb.to(DEVICE), yb.to(DEVICE)
            out      = model(xb)
            pred     = out.argmax(1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
            for i in range(len(yb)):
                class_correct[yb[i]] += (pred[i] == yb[i]).item()
                class_total[yb[i]]   += 1

    print(f"\nOverall Accuracy: {correct/total*100:.2f}%")
    print("\nPer class accuracy:")
    for i, cls in enumerate(CLASSES):
        acc = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
        bar = "█" * int(acc // 5)
        print(f"  {cls}: {acc:.1f}% {bar}")

if __name__ == '__main__':
    train()