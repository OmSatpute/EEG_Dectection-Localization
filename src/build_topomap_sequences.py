import argparse
import os
import random
import shutil

import numpy as np

from config import DATA_DIR, ORDERED_CHANNELS
from src.features import extract_features, segment_signal
from src.preprocessing import preprocess_file
from src.topomap import generate_topomap_array


SFREQ = 160


def split_file_list(files, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)

    val_count = max(1, int(len(files) * val_ratio))
    test_count = max(1, int(len(files) * test_ratio))
    val_files = set(files[:val_count])
    test_files = set(files[val_count:val_count + test_count])

    return {
        "train": [file for file in files if file not in val_files and file not in test_files],
        "val": [file for file in files if file in val_files],
        "test": [file for file in files if file in test_files],
    }


def clear_output(output_dir):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def process_subject(file_path, save_dir, max_segments):
    subject_name = os.path.splitext(os.path.basename(file_path))[0]

    if os.path.isdir(save_dir) and any(name.endswith(".npy") for name in os.listdir(save_dir)):
        print(f"Skipping existing subject: {subject_name}")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing subject: {subject_name}")

    raw = preprocess_file(file_path, ORDERED_CHANNELS)
    segments = segment_signal(raw)

    if max_segments is not None:
        segments = segments[:max_segments]

    if len(segments) == 0:
        print(f"Warning: no segments created for {subject_name}")
        return

    features = extract_features(segments, SFREQ)
    topomaps = generate_topomap_array(features, raw.info)

    for index, topomap in enumerate(topomaps):
        np.save(os.path.join(save_dir, f"seg_{index}.npy"), topomap)


def build_sequences(args):
    output_dir = os.path.join(DATA_DIR, "topomap_sequence")

    if args.clean:
        clear_output(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        for label in ["stroke", "non_stroke"]:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    for label in ["stroke", "non_stroke"]:
        raw_dir = os.path.join(DATA_DIR, "raw", label)
        files = [file for file in os.listdir(raw_dir) if file.lower().endswith(".edf")]
        split_map = split_file_list(files, args.val_ratio, args.test_ratio, args.seed)

        for split, split_files in split_map.items():
            print(f"\n{label} {split}: {len(split_files)} subject(s)")

            for file in split_files:
                file_path = os.path.join(raw_dir, file)
                subject_name = os.path.splitext(file)[0]
                save_dir = os.path.join(output_dir, split, label, subject_name)
                process_subject(file_path, save_dir, args.max_segments)

    print(f"\nDone. Sequence dataset saved at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build subject-wise topomap sequences for CNN-LSTM training."
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-segments", type=int, default=60)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing data/topomap_sequence before rebuilding.",
    )
    args = parser.parse_args()

    build_sequences(args)


if __name__ == "__main__":
    main()
