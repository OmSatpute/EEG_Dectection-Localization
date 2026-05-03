import os
import sys

# ✅ Fix import path
root_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(root_dir, 'config.py')) and root_dir != os.path.dirname(root_dir):
    root_dir = os.path.dirname(root_dir)
sys.path.insert(0, root_dir)

from config import DATA_DIR
import numpy as np

def load_single_subject(folder):
    subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    if subdirs:
        folder = os.path.join(folder, subdirs[0])
    
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    return folder, len(files)

stroke_folder = os.path.join(DATA_DIR, "topomap_single", "stroke")
actual_path, count = load_single_subject(stroke_folder)
print(f"Path: {actual_path}")
print(f"Count: {count}")
