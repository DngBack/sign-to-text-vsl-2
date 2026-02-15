import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
import os
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
import time
st.set_page_config(page_title="VSL Prediction", layout="centered")
st.title("SIGN LANGUAGE PREDICTION")

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

ALL_POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
UPPER_BODY_POSE_CONNECTIONS = []
# ====================
# Load model and label_map
# ====================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Models/checkpoints/final_model.keras')

@st.cache_data
def load_label_map():
    import json
    with open('Logs/label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

model = load_model()
label_map, inv_label_map = load_label_map()
# ====================
# Video processing functions
# ====================
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
    # Interpolate keypoints sequence to 60 frames
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

  step = max(1, total_frames // 100)  # step to sample frames

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

def process_webcam_to_sequence():
    cap = cv2.VideoCapture(0)
    st.write("⏳ Preparing... Starting in 1.5 seconds...")
    time.sleep(1.5)
    st.write("🎥 Recording for 4 seconds...")
    sequence = []
    start_time = time.time()

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam")
            break
        elapsed_time = time.time() - start_time
        if elapsed_time > 4:
            break
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        if keypoints is not None:
            sequence.append(keypoints)
        stframe.image(image, channels="BGR", caption="Webcam feed", use_container_width=True)

    cap.release()
    
    return sequence

# Streamlit App

input_mode = st.radio("Select input source:", ["🎞️ Video file", "📷 Webcam"])

sequence = None
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
if input_mode == "🎞️ Video file":
    uploaded_file = st.file_uploader("Upload video (.mp4, .avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.video(tmp_path)
        if st.button("🔍 Predict from video"):
            sequence = sequence_frames(tmp_path, holistic)

elif input_mode == "📷 Webcam":
    st.warning("Click the button below to start recording from webcam.")
    if st.button("📸 Record and predict"):
        sequence = process_webcam_to_sequence()

# Prediction
if sequence is not None:
    kp = interpolate_keypoints(sequence)
    preds = model.predict(np.expand_dims(kp, axis=0))[0]
    top_k = 10
    top_indices = np.argsort(preds)[::-1][:top_k]
    st.success(f"✅ Top prediction: **{inv_label_map[top_indices[0]]}**")
    st.markdown("### 🔝 Top 10 most similar labels:")
    for rank, idx in enumerate(top_indices, start=1):
        label = inv_label_map[idx]
        prob = preds[idx] * 100
        st.write(f"{rank}. **{label}** — {prob:.2f}%")
