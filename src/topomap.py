import os
import numpy as np
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count


# GET ELECTRODE POSITIONS
def get_2d_pos(info):
    return np.array([ch['loc'][:2] for ch in info['chs']])


# GENERATE FOR ALL SAMPLES (IN-MEMORY)
def generate_topomap_array(features, info, img_size=64):
    pos = get_2d_pos(info)

    xi = np.linspace(pos[:, 0].min(), pos[:, 0].max(), img_size)
    yi = np.linspace(pos[:, 1].min(), pos[:, 1].max(), img_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    all_maps = []

    print(f"Generating topomap array for {len(features)} samples...")

    for i, feature in enumerate(features):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(features)}")
            
        bands = feature.reshape(26, 4).T
        band_maps = []

        for j in range(4):
            grid_z = griddata(
                pos,
                bands[j],
                (grid_x, grid_y),
                method='cubic',
                fill_value=0
            )
            band_maps.append(grid_z)

        all_maps.append(band_maps)

    return np.array(all_maps)  # (T, 4, 64, 64)


# WORKER FUNCTION   
def process_single_sample(args):
    feature, pos, grid_x, grid_y, idx, save_dir = args

    if idx % 500 == 0:
        print(f"Processing sample {idx}", flush=True)

    bands = feature.reshape(26, 4).T
    topomap_stack = []

    for j in range(4):
        grid_z = griddata(
            pos,
            bands[j],
            (grid_x, grid_y),
            method='cubic',
            fill_value=0
        )
        topomap_stack.append(grid_z)

    topomap_stack = np.array(topomap_stack)

    np.save(os.path.join(save_dir, f"{idx}.npy"), topomap_stack)


# MAIN FUNCTION
def generate_topomap_dataset(features, info, save_dir, img_size=64):

    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating topomaps in: {save_dir}")

    pos = get_2d_pos(info)

    xi = np.linspace(pos[:, 0].min(), pos[:, 0].max(), img_size)
    yi = np.linspace(pos[:, 1].min(), pos[:, 1].max(), img_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    args_list = [
        (features[i], pos, grid_x, grid_y, i, save_dir)
        for i in range(len(features))
    ]

    num_workers = min(4, cpu_count())

    with Pool(num_workers) as pool:
        pool.map(process_single_sample, args_list)

    print(f"Done: {save_dir}")
