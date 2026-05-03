import numpy as np
from scipy.signal import welch


def segment_signal(raw, window_size=2, step_size=1):
    """
    Segment EEG signal into fixed-size windows.

    raw: MNE Raw object
    window_size: window length in seconds
    step_size: seconds between consecutive windows
    returns: (num_segments, channels, samples)
    """
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    samples_per_window = int(window_size * sfreq)
    samples_per_step = int(step_size * sfreq)

    segments = []

    for start in range(0, data.shape[1] - samples_per_window + 1, samples_per_step):
        segment = data[:, start:start + samples_per_window]
        segments.append(segment)

    return np.array(segments)


def compute_band_power(segment, sfreq):
    """
    Compute Delta, Theta, Alpha, and Beta band power for one segment.

    segment: (channels, samples)
    returns: (channels * 4)
    """
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    features = []

    for channel_data in segment:
        freqs, psd = welch(channel_data, fs=sfreq)

        for band in bands.values():
            idx = (freqs >= band[0]) & (freqs <= band[1])
            features.append(np.mean(psd[idx]))

    return np.array(features)


def extract_features(segments, sfreq):
    """
    Extract frequency features for all segments.

    segments: (num_segments, channels, samples)
    returns: (num_segments, channels * 4)
    """
    return np.array([compute_band_power(segment, sfreq) for segment in segments])
