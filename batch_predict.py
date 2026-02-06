import os
import cv2
import json
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.interpolate import interp1d

# ===================== CONFIG =====================
VIDEO_DIR = r"Dataset/Videos"
MODEL_PATH = "Models/checkpoints/final_model.keras"
LABEL_MAP_PATH = "Logs/label_map.json"
OUTPUT_TXT = "predict_results.txt"

SEQUENCE_LENGTH = 60
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21

mp_holistic = mp.solutions.holistic

# ===================== LOAD MODEL & LABEL MAP =====================
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

NUM_CLASSES = len(inv_label_map)
print(f"✅ Model loaded | {NUM_CLASSES} classes")

# ===================== MEDIAPIPE =====================
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def extract_keypoints(results):
    pose = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    lh = np.zeros((N_HAND_LANDMARKS, 3))
    rh = np.zeros((N_HAND_LANDMARKS, 3))

    if results.pose_landmarks:
        for i in range(min(N_UPPER_BODY_POSE_LANDMARKS, len(results.pose_landmarks.landmark))):
            lm = results.pose_landmarks.landmark[i]
            pose[i] = [lm.x, lm.y, lm.z]

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    return np.concatenate([pose, lh, rh]).flatten()

# ===================== VIDEO → SEQUENCE =====================
def sequence_frames(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // 100)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue

        try:
            results = mediapipe_detection(frame, holistic)
            frames.append(extract_keypoints(results))
        except:
            continue

    cap.release()
    return frames

def interpolate_keypoints(sequence, target_len=60):
    if not sequence:
        return None

    t_ori = np.linspace(0, 1, len(sequence))
    t_tar = np.linspace(0, 1, target_len)

    feat = sequence[0].shape[0]
    out = np.zeros((target_len, feat))

    for i in range(feat):
        vals = [f[i] for f in sequence]
        f = interp1d(t_ori, vals, kind="cubic", bounds_error=False, fill_value="extrapolate")
        out[:, i] = f(t_tar)

    return out

# ===================== MAIN =====================
videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi"))]
print(f"🎬 Found {len(videos)} videos")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic, open(OUTPUT_TXT, "w", encoding="utf-8") as out:

    for idx, video in enumerate(videos, start=1):
        video_path = os.path.join(VIDEO_DIR, video)
        print(f"[{idx}/{len(videos)}] Processing {video}...")

        frames = sequence_frames(video_path, holistic)
        kp = interpolate_keypoints(frames, SEQUENCE_LENGTH)

        if kp is None or np.isnan(kp).any():
            out.write(f"{video} | ERROR | 0%\n")
            continue

        preds = model.predict(np.expand_dims(kp, axis=0), verbose=0)[0]
        best_idx = int(np.argmax(preds))
        best_label = inv_label_map[best_idx]
        best_prob = preds[best_idx] * 100

        out.write(f"{video} | {best_label} | {best_prob:.2f}%\n")

print("✅ DONE! Results saved to predict_results.txt")
