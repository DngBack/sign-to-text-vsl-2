import os
import glob
import numpy as np
import json
from collections import Counter

DATA_PATH = "Data"               # thư mục chứa các folder .npz
LABEL_MAP_PATH = "Logs/label_map.json"

# Load label_map
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    label_map = json.load(f)

num_classes = len(label_map)
print("NUM_CLASSES =", num_classes)

# Đếm label từ file npz
label_counter = Counter()

npz_files = glob.glob(os.path.join(DATA_PATH, "**", "*.npz"), recursive=True)
print("Total npz files:", len(npz_files))

for fpath in npz_files:
    data = np.load(fpath)
    lbl = int(data["label"])
    label_counter[lbl] += 1

# In kết quả
print("\n=== THỐNG KÊ THEO CLASS ===")
for word, idx in label_map.items():
    count = label_counter.get(idx, 0)
    print(f"{idx:3d} | {count:4d} | {word}")
