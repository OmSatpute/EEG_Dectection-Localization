import mne
import numpy as np
import csv
mne.set_log_level('ERROR')

from config import ELECTRODE_PATH


def load_custom_montage(file_path):
    """
    Load electrode coordinates from a TSV file and create an MNE montage.
    The TSV should have columns: name, X, Y, Z.
    """
    pos_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["name"]
            try:
                x = float(row["X"])
                y = float(row["Y"])
                z = float(row["Z"])
                pos_dict[name] = np.array([x, y, z])
            except (ValueError, KeyError):
                continue

    return mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame="head")


def load_edf(filepath):
    """
    Load EDF file and clean channel names
    """
    raw = mne.io.read_raw_edf(filepath, preload=True)

    # Remove 'time' channel if present
    if 'time' in raw.ch_names:
        raw.drop_channels(['time'])

    # Clean channel names: remove dots + uppercase
    cleaned_names = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
    raw.rename_channels(cleaned_names)

    return raw


def select_channels(raw, ordered_channels):
    """
    Select only required EEG channels (robust version)
    """
    available_channels = raw.ch_names

    # Keep only channels that exist in the file
    valid_channels = [ch for ch in ordered_channels if ch in available_channels]

    if len(valid_channels) == 0:
        raise ValueError("No matching EEG channels found!")

    # Use modern API
    raw.pick(valid_channels)

    return raw


def apply_filter(raw, l_freq=0.5, h_freq=40):
    """
    Apply band-pass filter
    """
    raw.filter(l_freq, h_freq)
    return raw


def apply_montage(raw):
    """
    Apply custom electrode positions for topomap
    """
    montage = load_custom_montage(ELECTRODE_PATH)

    # Apply montage safely
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    return raw


def preprocess_file(filepath, ordered_channels, target_sfreq=160):
    """
    Full preprocessing pipeline
    """

    # Step 1: Load
    raw = load_edf(filepath)

    # Step 2: Select channels
    raw = select_channels(raw, ordered_channels)

    # Step 3: Apply montage (AFTER channel selection)
    raw = apply_montage(raw)

    # Step 4: Resample
    raw.resample(target_sfreq)

    # Step 5: Filter
    raw = apply_filter(raw)

    return raw
