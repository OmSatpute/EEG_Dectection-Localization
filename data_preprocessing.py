import os
import shutil
import numpy as np

from src.dataset_utils import split_subjects, process_subjects
from src.features import extract_features
from src.topomap import generate_topomap_dataset
from src.preprocessing import preprocess_file
from config import DATA_DIR, ORDERED_CHANNELS

sfreq = 160


# SHUFFLE FUNCTION
def shuffle_data(X, y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def clear_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def build_split_data(splits, split_name):
    X_stroke, y_stroke = process_subjects(
        splits[split_name]["stroke"], DATA_DIR, "stroke", ORDERED_CHANNELS
    )

    X_nonstroke, y_nonstroke = process_subjects(
        splits[split_name]["non_stroke"], DATA_DIR, "non_stroke", ORDERED_CHANNELS
    )

    X = np.concatenate([X_stroke, X_nonstroke], axis=0)
    y = np.concatenate([y_stroke, y_nonstroke], axis=0)
    X, y = shuffle_data(X, y)

    return X, y, X_stroke, X_nonstroke


def save_topomap_split(split_name, X_stroke, X_nonstroke, info):
    stroke_dir = os.path.join(DATA_DIR, "topomap", split_name, "stroke")
    nonstroke_dir = os.path.join(DATA_DIR, "topomap", split_name, "non_stroke")

    clear_directory(stroke_dir)
    clear_directory(nonstroke_dir)

    generate_topomap_dataset(
        extract_features(X_stroke, sfreq),
        info,
        stroke_dir
    )

    generate_topomap_dataset(
        extract_features(X_nonstroke, sfreq),
        info,
        nonstroke_dir
    )


def main():

    # SPLIT SUBJECTS
    splits = split_subjects(DATA_DIR)

    X_train, y_train, X_train_stroke, X_train_nonstroke = build_split_data(splits, "train")
    X_val, y_val, X_val_stroke, X_val_nonstroke = build_split_data(splits, "val")
    X_test, y_test, X_test_stroke, X_test_nonstroke = build_split_data(splits, "test")

    # FEATURE EXTRACTION
    X_train_features = extract_features(X_train, sfreq)
    X_val_features = extract_features(X_val, sfreq)
    X_test_features = extract_features(X_test, sfreq)

    print("Train Features:", X_train_features.shape)
    print("Validation Features:", X_val_features.shape)
    print("Test Features:", X_test_features.shape)

    # GET CHANNEL INFO
    sample_file = splits["train"]["stroke"][0]
    sample_path = os.path.join(DATA_DIR, "raw", "stroke", sample_file)

    raw_sample = preprocess_file(sample_path, ORDERED_CHANNELS)
    info = raw_sample.info

    # TOPOMAP GENERATION
    save_topomap_split("train", X_train_stroke, X_train_nonstroke, info)
    save_topomap_split("val", X_val_stroke, X_val_nonstroke, info)
    save_topomap_split("test", X_test_stroke, X_test_nonstroke, info)

    print("Topomap generation complete!")

    # SAVE FOR TEAM
    np.save(os.path.join(DATA_DIR, "X_train_features.npy"), X_train_features)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)

    np.save(os.path.join(DATA_DIR, "X_val_features.npy"), X_val_features)
    np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val)

    np.save(os.path.join(DATA_DIR, "X_test_features.npy"), X_test_features)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)


# REQUIRED FOR WINDOWS MULTIPROCESSING
if __name__ == "__main__":
    main()
