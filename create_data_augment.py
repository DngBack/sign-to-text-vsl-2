import cv2
import numpy as np
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from scipy.interpolate import interp1d
import random
from augment_function import inter_hand_distance, scale_keypoints_sequence,rotate_keypoints_sequence,translate_keypoints_sequence,time_stretch_keypoints_sequence,solve_2_link_ik_2d_v2

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

ALL_POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
UPPER_BODY_POSE_CONNECTIONS = []
for connection in ALL_POSE_CONNECTIONS:
    if connection[0] < N_UPPER_BODY_POSE_LANDMARKS and connection[1] < N_UPPER_BODY_POSE_LANDMARKS:
        UPPER_BODY_POSE_CONNECTIONS.append(connection)

def mediapipe_detection(image, model):
    # MediaPipe uses RGB, OpenCV uses BGR
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
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    keypoints = np.concatenate([pose_kps,left_hand_kps, right_hand_kps])
    return keypoints.flatten()


def interpolate_keypoints(keypoints_sequence, target_len=60):
    """Interpolate keypoints sequence to 60 frames."""
    if len(keypoints_sequence) == 0:
        return None

    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)

    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))

    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]

        interpolator = interp1d(
            original_times, feature_values,
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
          image, results = mediapipe_detection(frame, holistic)
          keypoints = extract_keypoints(results)

          if keypoints is not None:
              sequence_frames.append(keypoints)

      except Exception as e:
          continue

  cap.release()
  return sequence_frames

def create_action_folder(data_path, action):
    action_path = os.path.join(data_path, action)
    os.makedirs(action_path, exist_ok=True)
    return action_path

class GetTime():
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
    num_samples_to_generate: int,
    max_augs_per_sample: int = 3,
):
    """
    Generate augmented samples by randomly combining augmentation functions.

    Args:
        original_sequence: Original keypoints sequence.
        augmentation_functions: List of augmentation functions to choose from.
        num_samples_to_generate: Number of augmented samples to generate.
        max_augs_per_sample: Max number of augmentations to apply per sample.

    Returns:
        List of augmented keypoints sequences.
    """
    generated_samples = []
    if not original_sequence or not augmentation_functions:
        return generated_samples

    num_available_augs = len(augmentation_functions)

    for i in range(num_samples_to_generate):
        current_sequence = [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in original_sequence]

        num_augs_to_apply = random.randint(1, min(max_augs_per_sample, num_available_augs))
        selected_aug_funcs_indices = random.sample(range(num_available_augs), num_augs_to_apply)
        selected_aug_funcs = [augmentation_functions[idx] for idx in selected_aug_funcs_indices]
        random.shuffle(selected_aug_funcs)

        for aug_func in selected_aug_funcs:
            current_sequence = aug_func(current_sequence)
            if not current_sequence or all(frame is None for frame in current_sequence):
                break

        if not current_sequence or all(frame is None for frame in current_sequence):
            continue

        generated_samples.append(current_sequence)

    return generated_samples


DATA_PATH = os.path.join('Data')
DATASET_PATH = os.path.join('Dataset')
LOG_PATH = os.path.join('Logs')

sequence_length = 60

os.makedirs(LOG_PATH, exist_ok=True)
label_file = os.path.join(DATASET_PATH, 'Text', 'label.csv')
video_folder = os.path.join(DATASET_PATH, 'Videos')
df = pd.read_csv(label_file)

selected_actions = []

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)



selected_actions = sorted(df['LABEL'].unique())
label_map = { action: idx for idx, action in enumerate(selected_actions) }

label_map_path = os.path.join(LOG_PATH, 'label_map.json')
with open(label_map_path, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)


print(f"\nSelected {len(df['LABEL'].unique())} actions.")



time = GetTime()
print(f"{datetime.now()} Start processing data...")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    action_position = {action: idx + 1 for idx, action in enumerate(pd.unique(df['LABEL']))}

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Process actions'):
        action = row['LABEL']
        video_file = row['VIDEO']
        label      = label_map[action]

        print()
        #action_ascii = convert_to_ascii(action)
        action_path = create_action_folder(DATA_PATH, action)

        existing_files = [
            f for f in os.listdir(action_path)
            if f.endswith('.npz')
        ]
        idx = len(existing_files)

        #sequence_folder = os.path.join(action_path, str(idx))
        #os.makedirs(sequence_folder, exist_ok=True)

        video_path = os.path.join(video_folder, video_file)

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        frame_lists = sequence_frames(video_path, holistic)

        augmenteds = generate_augmented_samples(frame_lists, augmentations, 50, 2)

        augmenteds.append(frame_lists)

        for aug in augmenteds:
            seq = interpolate_keypoints(aug)

            if seq is None or np.isnan(seq).any():
                continue

            file_path = os.path.join(action_path, f'{idx}.npz')
            np.savez(
                file_path,
                sequence=seq.astype(np.float32),
                label=label
            )
            idx += 1

        #current_state['progress'].update({action: idx + 1})
        #save_progress_state(current_state, LOG_PATH)

        print(f"Action {action_position[action]}/{len(df['LABEL'].unique())} : {action} - Time: {time.get_time()}")


print(f"{'-'*50}\n")
print("DATA PROCESSING COMPLETED.")