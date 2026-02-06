import cv2
import numpy as np
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from scipy.interpolate import interp1d
import random
import re
import unicodedata

from augment_function import (
    inter_hand_distance,
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence,
    solve_2_link_ik_2d_v2
)

# =========================
# WINDOWS SAFE NAME
# =========================
def sanitize_windows_name(name: str, max_len=100):
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = re.sub(r'[\x00-\x1f]', '', name)
    name = name.strip().strip('.')
    if len(name) > max_len:
        name = name[:max_len]
    if not name:
        name = "unknown"
    return name


# =========================
# MEDIAPIPE SETUP
# =========================
mp_holistic = mp.solutions.holistic

N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = (
    N_UPPER_BODY_POSE_LANDMARKS
    + N_HAND_LANDMARKS
    + N_HAND_LANDMARKS
)

ALL_POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
UPPER_BODY_POSE_CONNECTIONS = [
    c for c in ALL_POSE_CONNECTIONS
    if c[0] < N_UPPER_BODY_POSE_LANDMARKS and c[1] < N_UPPER_BODY_POSE_LANDMARKS
]


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))

    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]

    if results and results.left_hand_landmarks:
        left_hand_kps = np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        )

    if results and results.right_hand_landmarks:
        right_hand_kps = np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        )

    keypoints = np.concatenate([pose_kps, left_hand_kps, right_hand_kps])
    return keypoints.flatten()


def interpolate_keypoints(keypoints_sequence, target_len=60):
    if len(keypoints_sequence) == 0:
        return None

    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)

    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))

    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]
        interpolator = interp1d(
            original_times,
            feature_values,
            kind='cubic',
            bounds_error=False,
            fill_value="extrapolate"
        )
        interpolated_sequence[:, feature_idx] = interpolator(target_times)

    return interpolated_sequence


def sequence_frames(video_path, holistic):
    sequence_frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 100)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue

        try:
            _, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence_frames.append(keypoints)
        except Exception:
            continue

    cap.release()
    return sequence_frames


def create_action_folder(data_path, action):
    action_path = os.path.join(data_path, action)
    os.makedirs(action_path, exist_ok=True)
    return action_path


class GetTime:
    def __init__(self):
        self.starttime = datetime.now()

    def get_time(self):
        return datetime.now() - self.starttime


augmentations = [
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence,
    inter_hand_distance
]


def generate_augmented_samples(
    original_sequence,
    augmentation_functions,
    num_samples_to_generate,
    max_augs_per_sample=3
):
    generated_samples = []
    if not original_sequence:
        return generated_samples

    for _ in range(num_samples_to_generate):
        current_sequence = [kp.copy() for kp in original_sequence]

        num_augs = random.randint(
            1,
            min(max_augs_per_sample, len(augmentation_functions))
        )

        selected_augs = random.sample(augmentation_functions, num_augs)
        random.shuffle(selected_augs)

        for aug_func in selected_augs:
            current_sequence = aug_func(current_sequence)
            if not current_sequence:
                break

        if current_sequence:
            generated_samples.append(current_sequence)

    return generated_samples


# =========================
# PATH SETUP
# =========================
DATA_PATH = 'Data'
DATASET_PATH = 'Dataset'
LOG_PATH = 'Logs'

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

label_file = os.path.join(DATASET_PATH, 'Text', 'label.csv')
video_folder = os.path.join(DATASET_PATH, 'Videos')

df = pd.read_csv(label_file)

selected_actions = sorted(df['LABEL'].unique())
label_map = {action: idx for idx, action in enumerate(selected_actions)}

with open(os.path.join(LOG_PATH, 'label_map.json'), 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

# mapping label gốc -> tên folder sạch
action_name_map = {
    action: sanitize_windows_name(action)
    for action in selected_actions
}

with open(os.path.join(LOG_PATH, 'action_name_map.json'), 'w', encoding='utf-8') as f:
    json.dump(action_name_map, f, ensure_ascii=False, indent=4)

print(f"Selected {len(selected_actions)} actions")

time = GetTime()
print(f"{datetime.now()} Start processing data...")

# =========================
# MAIN PROCESS
# =========================
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Process actions'):
        action_raw = row['LABEL']
        action = action_name_map[action_raw]
        video_file = row['VIDEO']
        label = label_map[action_raw]

        action_path = create_action_folder(DATA_PATH, action)

        existing_files = [f for f in os.listdir(action_path) if f.endswith('.npz')]
        idx = len(existing_files)

        video_path = os.path.join(video_folder, video_file)
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        frame_lists = sequence_frames(video_path, holistic)

        augmenteds = generate_augmented_samples(
            frame_lists,
            augmentations,
            num_samples_to_generate=50,
            max_augs_per_sample=2
        )

        augmenteds.append(frame_lists)

        for aug in augmenteds:
            seq = interpolate_keypoints(aug)
            if seq is None or np.isnan(seq).any():
                continue

            np.savez(
                os.path.join(action_path, f'{idx}.npz'),
                sequence=seq.astype(np.float32),
                label=label
            )
            idx += 1

        print(f"Action: {action_raw} → {action} | Time: {time.get_time()}")

print("-" * 50)
print("DATA PROCESSING COMPLETED.")
