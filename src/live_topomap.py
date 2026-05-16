import sys
import os

# =======================
# FIX IMPORT PATH
# =======================
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =======================
# LOAD DATA
# =======================
def load_all(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    data = []

    for f in files:
        arr = np.load(os.path.join(folder, f))
        data.append(arr)

    return np.array(data)  # (N, 4, 64, 64)


# =======================
# MASK
# =======================
def apply_mask(img):
    h, w = img.shape
    Y, X = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    radius = h // 2 - 2

    mask = (X - center[1])**2 + (Y - center[0])**2 <= radius**2
    img[~mask] = np.nan
    return img


# =======================
# BAND SELECTION
# =======================
def select_best_band(stroke_maps, nonstroke_maps):
    band_names = ["Delta", "Theta", "Alpha", "Beta"]
    scores = []

    for i in range(4):
        var = np.var(stroke_maps[:, i]) + np.var(nonstroke_maps[:, i])
        scores.append(var)

    best_idx = np.argmax(scores)
    print(f"Selected Band: {band_names[best_idx]}")

    return best_idx, band_names[best_idx]


# =======================
# MAIN
# =======================
def run_live_topomap():

    stroke_folder = "data/topomap_sequence/train/stroke/stroke_01"
    nonstroke_folder = "data/topomap_sequence/train/non_stroke/non_stroke_01"

    stroke_maps = load_all(stroke_folder)
    nonstroke_maps = load_all(nonstroke_folder)

    # =======================
    # BAND
    # =======================
    band_idx, band_name = select_best_band(stroke_maps, nonstroke_maps)

    stroke_band = stroke_maps[:, band_idx]

    # Fixed reference (mean non-stroke)
    nonstroke_mean = np.mean(nonstroke_maps[:, band_idx], axis=0)
    n = apply_mask(nonstroke_mean.copy())

    T = len(stroke_band)

    # =======================
    # PLOT
    # =======================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    titles = [
        f"Stroke Activity ({band_name})",
        "Abnormal Activity Map"
    ]

    for ax in axes:
        ax.axis("off")

    # =======================
    # UPDATE
    # =======================
    def update(t):

        s = apply_mask(stroke_band[t].copy())

        # Difference per frame
        d = s - n

        data_list = [s, d]

        for i in range(2):
            axes[i].clear()
            data = data_list[i]

            if i == 0:
                vmin = np.nanpercentile(data, 5)
                vmax = np.nanpercentile(data, 95)
                cmap = "jet"
            else:
                vmax = np.nanpercentile(np.abs(d), 99)
                vmin = -vmax
                cmap = "seismic"

            axes[i].imshow(
                data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear"
            )

            axes[i].contour(
                data,
                levels=6,
                colors='black',
                linewidths=0.4
            )

            # highlight abnormal
            if i == 1:
                import scipy.ndimage as ndimage
                mask = np.abs(d) > np.nanpercentile(np.abs(d), 95)
                # Remove small noise artifacts using morphological opening
                mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
                
                # Keep only the largest connected component to eliminate isolated noise
                labeled_mask, num_features = ndimage.label(mask)
                if num_features > 0:
                    sizes = np.bincount(labeled_mask.flat)[1:] # Sizes of each component
                    max_label = np.argmax(sizes) + 1
                    mask = (labeled_mask == max_label)
                    
                axes[i].contour(mask, colors='yellow', linewidths=1.5)

            circle = plt.Circle((32, 32), 31, color='black', fill=False)
            axes[i].add_patch(circle)

            axes[i].axis("off")
            axes[i].set_title(titles[i])

        return []

    # =======================
    # ANIMATION
    # =======================
    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=200,
        blit=False,
        cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()


# =======================
# RUN
# =======================
if __name__ == "__main__":
    run_live_topomap()
