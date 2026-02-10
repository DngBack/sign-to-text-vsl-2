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
import re  # Thư viện để xử lý ký tự đặc biệt

# Import các hàm augmentation từ file của bạn
from augment_function import (
    inter_hand_distance, scale_keypoints_sequence, 
    rotate_keypoints_sequence, translate_keypoints_sequence, 
    time_stretch_keypoints_sequence
)

# --- KHỞI TẠO CẤU HÌNH MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

# --- HÀM XỬ LÝ KÝ TỰ ĐẶC BIỆT CHO WINDOWS ---
def sanitize_folder_name(name):
    """
    Loại bỏ các ký tự cấm trên Windows: \ / : * ? " < > |
    """
    # Thay thế các ký tự cấm bằng dấu gạch dưới
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', name)
    # Loại bỏ khoảng trắng thừa hoặc dấu chấm ở cuối
    return sanitized.strip().strip('.')

# --- CÁC HÀM XỬ LÝ NHẬN DIỆN ---
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
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        
    return np.concatenate([pose_kps, left_hand_kps, right_hand_kps]).flatten()

def interpolate_keypoints(keypoints_sequence, target_len=60):
    if not keypoints_sequence or len(keypoints_sequence) < 2:
        return None

    keypoints_sequence = np.array(keypoints_sequence)
    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)
    
    num_features = keypoints_sequence.shape[1]
    interpolated_sequence = np.zeros((target_len, num_features))

    for feature_idx in range(num_features):
        feature_values = keypoints_sequence[:, feature_idx]
        interpolator = interp1d(original_times, feature_values, kind='cubic', 
                                bounds_error=False, fill_value="extrapolate")
        interpolated_sequence[:, feature_idx] = interpolator(target_times)

    return interpolated_sequence

def sequence_frames(video_path, holistic):
    sequence_frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Giới hạn lấy tối đa 100 frames để xử lý cho nhanh nhưng vẫn đủ thông tin
    step = max(1, total_frames // 100) 

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            try:
                _, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                sequence_frames.append(keypoints)
            except Exception:
                pass
        frame_idx += 1

    cap.release()
    return sequence_frames

def create_action_folder(data_path, action):
    # Sử dụng hàm sanitize để bảo vệ hệ thống file Windows
    clean_action = sanitize_folder_name(action)
    action_path = os.path.join(data_path, clean_action)
    os.makedirs(action_path, exist_ok=True)
    return action_path

class GetTime():
    def __init__(self):
        self.starttime = datetime.now()
    def get_time(self):
        return datetime.now() - self.starttime

# --- CẤU HÌNH DATA AUGMENTATION ---
augmentations = [
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence,
    inter_hand_distance
]

def generate_augmented_samples(original_sequence, augmentation_functions, num_samples, max_augs=2):
    generated_samples = []
    if not original_sequence:
        return []

    for _ in range(num_samples):
        # Copy dữ liệu gốc để tránh thay đổi sequence chính
        current_sequence = [kp.copy() for kp in original_sequence]
        
        num_augs_to_apply = random.randint(1, max_augs)
        selected_funcs = random.sample(augmentation_functions, num_augs_to_apply)

        for aug_func in selected_funcs:
            current_sequence = aug_func(current_sequence)
            if not current_sequence: break

        if current_sequence:
            generated_samples.append(current_sequence)
    return generated_samples

# --- MAIN PROCESS ---
DATA_PATH = 'Data'
DATASET_PATH = 'Dataset'
LOG_PATH = 'Logs'

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

label_file = os.path.join(DATASET_PATH, 'Text', 'label.csv')
video_folder = os.path.join(DATASET_PATH, 'Videos')
df = pd.read_csv(label_file)

# Tạo label map
selected_actions = sorted(df['LABEL'].unique())
label_map = {action: idx for idx, action in enumerate(selected_actions)}

with open(os.path.join(LOG_PATH, 'label_map.json'), 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print(f"Selected {len(selected_actions)} actions.")
timer = GetTime()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing'):
        action = row['LABEL']
        video_file = row['VIDEO']
        label = label_map[action]

        action_path = create_action_folder(DATA_PATH, action)
        
        # Đếm số file hiện có để đặt tên file tiếp theo
        existing_idx = len([f for f in os.listdir(action_path) if f.endswith('.npz')])

        video_path = os.path.join(video_folder, video_file)
        if not os.path.exists(video_path):
            continue

        # 1. Trích xuất keypoints gốc
        raw_frames = sequence_frames(video_path, holistic)
        if len(raw_frames) < 5: continue # Bỏ qua video quá ngắn/lỗi

        # 2. Tạo mẫu tăng cường (50 mẫu)
        augmenteds = generate_augmented_samples(raw_frames, augmentations, 50, 2)
        augmenteds.append(raw_frames) # Thêm cả mẫu gốc

        # 3. Nội suy và Lưu trữ
        for seq_data in augmenteds:
            interpolated = interpolate_keypoints(seq_data, target_len=60)
            
            if interpolated is not None and not np.isnan(interpolated).any():
                file_name = f"{existing_idx}.npz"
                np.savez(
                    os.path.join(action_path, file_name),
                    sequence=interpolated.astype(np.float32),
                    label=label
                )
                existing_idx += 1

print(f"\nCOMPLETED. Total time: {timer.get_time()}")