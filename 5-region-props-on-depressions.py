# %% 1.0 Libraries and filepaths

import pandas as pd
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt

from skimage import measure, morphology

from joblib import Parallel, delayed
from multiprocessing import shared_memory
import time
from tqdm import tqdm # Optional, for progress bars

detrend_path = f'./out_data/detrended_dem_all_basins.tif'

# %% 2.0 Get summary stats for elevation and make a histogram

# with rio.open(detrend_path) as src:
#     data = src.read(1, masked=True)
#     data = data * 0.3048 # Convert feet to meters
#     flat = data.compressed()  

#     total_pix = len(flat)
#     below_neg_3 = (np.sum(flat < -3) / total_pix * 100)
#     above_3 = (np.sum(flat > 3) / total_pix * 100)
#     below_neg_1 = (np.sum(flat < -1) / total_pix * 100)
#     above_1 = (np.sum(flat > 1) / total_pix * 100)
#     below_0 = (np.sum(flat < 0) / total_pix * 100)
#     print(f'{below_neg_3:.2f} % of pixels below -3 meters')
#     print(f'{above_3:.2f} % of pixels above 3 meters')
#     print(f'{below_neg_1:.2f} % of pixels below -1 meters')
#     print(f'{above_1:.2f} % of pixels above 1 meters')
#     print(f'{below_0:.2f} % of pixels below 0 meters')

#     # Trim the data to only include elevations between -3 and 3
#     trimmed = flat[(flat >= -3) & (flat <= 3)]

#     # Create the histogram using the trimmed data
#     plt.hist(trimmed, bins=100, color='blue', edgecolor='black')
#     plt.title("Histogram of Detrended DEM Elevations")
#     plt.xlabel("Elevation (meters)")
#     plt.xlim(-3, 3)
#     plt
#     plt.ylabel("Frequency")
#     plt.show()

# %% 3.0 Prep the DEM in shared memory for multiprocessing

with rio.open(detrend_path) as src:
    dem_local = src.read(1, masked=True) * 0.3048  # Convert feet to meters

shm = shared_memory.SharedMemory(create=True, size=dem_local.nbytes)
dem = np.ndarray(dem_local.shape, dtype=dem_local.dtype, buffer=shm.buf)
dem[:] = dem_local[:]

# %% 4.0 Worker function to process each elevation threshold

def analyze_threshold(t):
    """
    Analyze the DEM for depressions below a given elevation threshold.
    """
    binary = np.logical_and(dem < t, dem > -9000)
    binary = morphology.remove_small_objects(binary, min_size=300, connectivity=2)  # Used more relaxed connectivity
    labels = measure.label(binary, connectivity=2)
    props_table = measure.regionprops_table(labels, properties=['num_pixels', 'perimeter', 'perimeter_crofton'])

    pixels = props_table['num_pixels'] # NOTE: using num_pixels and scaling to meters outside of this function
    perimeter = props_table['perimeter']
    perimeter_crofton = props_table['perimeter_crofton']

    return dict(
        threshold=t,
        n_ponds=len(pixels),
        total_inundated_pix=np.sum(pixels),
        total_perimeter=np.sum(perimeter),
        total_perimeter_crofton=np.sum(perimeter_crofton),
        mean_feature_pix=np.mean(pixels),
        std_feature_pix=np.std(pixels),
    )

# %% 5.0 Run the analysis in parallel

thresholds = np.arange(-1.5, 0.5, 0.1)
batch_size = 4

results = []
start_time = time.time()

for i in range(0, len(thresholds), batch_size):
    batch = thresholds[i:i + batch_size]
    batch_start = time.time()
    r = Parallel(
        n_jobs=batch_size, 
        backend='loky', # Bypasses the GIL for CPU-bound tasks
        prefer='processes')( # Processes forks the subprocesses, becuse workload is CPU heavy. Would use threads for IO heavy tasks (e.g., file reads)
            delayed(analyze_threshold)(t) for t in batch)
    
    results.extend(r)
    batch_elapsed = time.time() - batch_start
    print(f'Processed batch {i // batch_size + 1} in {batch_elapsed:.2f} seconds')

total_elapsed = time.time() - start_time
print(f'Total elapsed time: {total_elapsed:.2f} seconds')

# clean up shared memory
shm.close()
shm.unlink()


# %% 6.0 Create a DataFrame with the results

total_pix = 170773718 # NOTE: this number was taken from code chunk 2.0
out_df = pd.DataFrame(results)
out_df['inundated_frac'] = out_df['total_inundated_pix'] / total_pix * 100
out_df['inundated_area_m2'] = out_df['total_inundated_pix'] * (2.5 * .3048)**2 
out_df['mean_feature_area_m2'] = out_df['mean_feature_pix'] * (2.5 * 0.3048)**2  
out_df['std_feature_area_m2'] = out_df['std_feature_pix'] * (2.5 * 0.3048)**2  
out_df['total_perimeter_m'] = out_df['total_perimeter'] * 2.5 * 0.3048  # Convert pixels to feet to meters
out_df['total_perimeter_crofton_m'] = out_df['total_perimeter_crofton'] * 2.5 * 0.3048  # Convert pixels to feet to meters

out_df.drop(columns=
            ['total_inundated_pix', 'mean_feature_pix', 'std_feature_pix',
             'total_perimeter', 'total_perimeter_crofton'], 
        inplace=True
)

out_df.to_csv('./out_data/region_props_on_depressions.csv', index=False)

# %% Write binary rasters for a few thresholds
def write_binary_inundation_raster(
    bool_mask: np.array,
    out_path: str, 
    src_profile: dict
):
    """
    Write a binary raster mask to disk.
    """
    prof = src_profile.copy()
    prof.update({
        'dtype': 'uint8', 
        'count': 1,
        'nodata': 0,
    })

    with rio.open(out_path, 'w', **prof) as dst:
        dst.write(bool_mask.astype(np.uint8), 1)

# %%
write_thresholds = [-1, -0.75, -0.5, -0.25, -0.1, 0.25]
out_dir = './out_data/modeled_inundations/'

with rio.open(detrend_path) as src:
    dem = src.read(1, masked=True) * 0.3048  # Convert feet to meters
    profile = src.profile

for t in write_thresholds:
    mask = dem < t
    out_path = f'{out_dir}inundation_mask_{t:.2f}m.tif'
    write_binary_inundation_raster(mask, out_path, profile)
    print(f'Wrote {out_path}')

# %%
