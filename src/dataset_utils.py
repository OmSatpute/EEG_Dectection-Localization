import os
import random
import re

import numpy as np

from src.features import segment_signal
from src.preprocessing import preprocess_file


def numeric_key(path):
    name = os.path.basename(path)
    numbers = re.findall(r"\d+", name)
    return int(numbers[-1]) if numbers else 0


def split_subjects(data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    if train_ratio + val_ratio + test_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be <= 1.0")

    random.seed(seed)

    stroke_path = os.path.join(data_dir, "raw", "stroke")
    nonstroke_path = os.path.join(data_dir, "raw", "non_stroke")

    stroke_files = sorted(os.listdir(stroke_path))
    nonstroke_files = sorted(os.listdir(nonstroke_path))

    random.shuffle(stroke_files)
    random.shuffle(nonstroke_files)

    stroke_train_end = int(len(stroke_files) * train_ratio)
    stroke_val_end = stroke_train_end + int(len(stroke_files) * val_ratio)

    nonstroke_train_end = int(len(nonstroke_files) * train_ratio)
    nonstroke_val_end = nonstroke_train_end + int(len(nonstroke_files) * val_ratio)

    return {
        "train": {
            "stroke": stroke_files[:stroke_train_end],
            "non_stroke": nonstroke_files[:nonstroke_train_end],
        },
        "val": {
            "stroke": stroke_files[stroke_train_end:stroke_val_end],
            "non_stroke": nonstroke_files[nonstroke_train_end:nonstroke_val_end],
        },
        "test": {
            "stroke": stroke_files[stroke_val_end:],
            "non_stroke": nonstroke_files[nonstroke_val_end:],
        },
    }


def limit_segments(segments, max_segments=60):
    if len(segments) > max_segments:
        idx = np.random.choice(len(segments), max_segments, replace=False)
        return segments[idx]
    return segments


def process_subjects(file_list, data_dir, label, ordered_channels):
    all_segments = []
    all_labels = []

    for file_name in file_list:
        file_path = os.path.join(data_dir, "raw", label, file_name)

        raw = preprocess_file(file_path, ordered_channels)
        segments = segment_signal(raw)
        segments = limit_segments(segments, 60)

        all_segments.append(segments)

        label_value = 1 if label == "stroke" else 0
        all_labels.extend([label_value] * len(segments))

    all_segments = np.concatenate(all_segments, axis=0)
    return all_segments, np.array(all_labels)
