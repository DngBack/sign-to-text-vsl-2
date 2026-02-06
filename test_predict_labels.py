import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import pandas as pd
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

# ================= CONFIG =================
MODEL_PATH = "Models/checkpoints/final_model.keras"
LABEL_MAP_PATH = "Logs/label_map.json"
LABEL_CSV_PATH = "Dataset/Text/label.csv"
VIDEO_FOLDER = "Dataset/Videos"

SEQUENCE_LEN = 60
TOP_K = 5
LOG_DIR = "Logs"
LOG_FILE = os.path.join(LOG_DIR, "wrong_predictions.txt")
# =========================================

os.makedirs(LOG_DIR, exist_ok=True)

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21

# ================= LOAD =================
print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

df = pd.read_csv(LABEL_CSV_PATH)
video_to_label = dict(zip(df["VIDEO"], df["LABEL"]))


# ================= FUNCTIONS =================
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    pose = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    lh = np.zeros((N_HAND_LANDMARKS, 3))
    rh = np.zeros((N_HAND_LANDMARKS, 3))

    if results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            lm = results.pose_landmarks.landmark[i]
            pose[i] = [lm.x, lm.y, lm.z]

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    return np.concatenate([pose, lh, rh]).flatten()


def interpolate_keypoints(seq, target_len=SEQUENCE_LEN):
    if len(seq) == 0:
        return None

    t_src = np.linspace(0, 1, len(seq))
    t_dst = np.linspace(0, 1, target_len)

    seq = np.array(seq)
    out = np.zeros((target_len, seq.shape[1]))

    for i in range(seq.shape[1]):
        f = interp1d(t_src, seq[:, i], kind="cubic", fill_value="extrapolate")
        out[:, i] = f(t_dst)

    return out


def extract_sequence_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            _, results = mediapipe_detection(frame, holistic)
            frames.append(extract_keypoints(results))

    cap.release()
    return frames


# ================= TEST LOOP =================
videos = [
    f for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
]

total = 0
correct = 0
wrong_logs = []

print(f"\n🎞️ Found {len(videos)} videos\n")

for video in tqdm(videos, desc="Testing videos"):
    if video not in video_to_label:
        continue

    gt_label = video_to_label[video]
    video_path = os.path.join(VIDEO_FOLDER, video)

    raw_seq = extract_sequence_from_video(video_path)
    seq = interpolate_keypoints(raw_seq)

    if seq is None or np.isnan(seq).any():
        continue

    preds = model.predict(seq[None, ...], verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = inv_label_map[pred_idx]

    total += 1

    if pred_label == gt_label:
        correct += 1
    else:
        topk = np.argsort(preds)[::-1][:TOP_K]
        wrong_logs.append(
            f"VIDEO: {video}\n"
            f"  TRUE : {gt_label}\n"
            f"  PRED : {pred_label}\n"
            f"  CONF : {preds[pred_idx]:.4f}\n"
            f"  TOP{TOP_K}: {', '.join([inv_label_map[i] for i in topk])}\n"
            f"{'-'*40}\n"
        )

# ================= RESULT =================
accuracy = correct / total * 100 if total > 0 else 0

print("\n" + "=" * 60)
print(f"✅ TOTAL TESTED : {total}")
print(f"🎯 CORRECT      : {correct}")
print(f"❌ WRONG        : {total - correct}")
print(f"📊 ACCURACY     : {accuracy:.2f}%")
print("=" * 60)

# ================= SAVE TXT LOG =================
if wrong_logs:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(wrong_logs)
    print(f"\n📄 Wrong predictions saved to: {LOG_FILE}")
else:
    print("\n🎉 No wrong predictions!")
