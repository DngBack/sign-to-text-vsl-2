import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
import json
import sys
import time

# ====================
# Config
# ====================
MODEL_PATH = 'Models/checkpoints/final_model.keras'
LABEL_MAP_PATH = 'Logs/label_map.json'
SEQUENCE_LEN = 60

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21

# ====================
# Load model & labels
# ====================
print('🔄 Loading model...')
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

print(f'✅ Model loaded | {len(inv_label_map)} classes')

# ====================
# Helper functions
# ====================
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

    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_len)
    seq = np.array(seq)

    out = np.zeros((target_len, seq.shape[1]))
    for i in range(seq.shape[1]):
        f = interp1d(x_old, seq[:, i], kind='cubic', fill_value='extrapolate')
        out[:, i] = f(x_new)
    return out


def extract_sequence_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = mediapipe_detection(frame, holistic)
            kps = extract_keypoints(results)
            frames.append(kps)

    cap.release()
    return frames

# ====================
# Main
# ====================
if len(sys.argv) < 2:
    print('❌ Usage: python main_cli.py <video_path>')
    sys.exit(1)

video_path = sys.argv[1]
print(f'🎞️ Processing video: {video_path}')

seq = extract_sequence_from_video(video_path)
seq = interpolate_keypoints(seq)

if seq is None:
    print('❌ Failed to extract keypoints')
    sys.exit(1)

preds = model.predict(np.expand_dims(seq, axis=0))[0]

TOP_K = 5
idxs = np.argsort(preds)[::-1][:TOP_K]

print('\n🎯 Prediction result:')
for rank, idx in enumerate(idxs, 1):
    print(f'{rank}. {inv_label_map[idx]} : {preds[idx]*100:.2f}%')
