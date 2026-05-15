import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "topomap_sequence"
OUTPUT_DIR = ROOT_DIR / "outputs" / "sample_topomaps"

def plot_topomap(npy_path, output_path, title):
    # Load the (4, 64, 64) numpy array
    data = np.load(npy_path)
    
    bands = ["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", "Beta (13-30 Hz)"]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i in range(4):
        # We use the 'jet' colormap which is standard for EEG heatmaps
        im = axes[i].imshow(data[i], cmap='jet', interpolation='bilinear')
        axes[i].set_title(bands[i], fontsize=12)
        axes[i].axis('off')
        
        # Add a colorbar to each subplot
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for title
    plt.savefig(output_path, dpi=200, bbox_inches='tight', transparent=False)
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Grab a few specific samples from the generated test sequence data
    samples_to_plot = [
        (DATA_DIR / "test" / "stroke" / "stroke_04" / "seg_1.npy", "Stroke Subject 04 - Segment 1"),
        (DATA_DIR / "test" / "stroke" / "stroke_10" / "seg_1.npy", "Stroke Subject 10 - Segment 1"),
        (DATA_DIR / "test" / "non_stroke" / "non_stroke_04" / "seg_1.npy", "Non-Stroke Subject 04 - Segment 1"),
        (DATA_DIR / "test" / "non_stroke" / "non_stroke_10" / "seg_1.npy", "Non-Stroke Subject 10 - Segment 1")
    ]
    
    generated_count = 0
    for path, title in samples_to_plot:
        if path.exists():
            out_name = f"{title.replace(' ', '_').replace('-', '')}.png"
            plot_topomap(path, OUTPUT_DIR / out_name, title)
            print(f"Generated: {out_name}")
            generated_count += 1
        else:
            print(f"Could not find {path}")
            
    if generated_count == 0:
        print("No samples were found.")
    else:
        print(f"\nSuccessfully saved {generated_count} topomap images to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
