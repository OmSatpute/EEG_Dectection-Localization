import os

# Get the directory where config.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORDERED_CHANNELS = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8',
    'FCZ', 'FC3', 'FC4', 'FT7', 'FT8',
    'CZ', 'C3', 'C4',
    'CPZ', 'CP3', 'CP4',
    'TP7', 'TP8',
    'PZ', 'P3', 'P4',
    'OZ', 'O1', 'O2'
]

# Path to custom electrode locations (absolute path)
ELECTRODE_PATH = os.path.join(BASE_DIR, "electrodes.tsv")

# Path to data (absolute path)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data"))