# %% 1.0 Libraries and filepaths
import os
import pandas as pd
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt

from skimage import measure, morphology

from joblib import Parallel, delayed
from multiprocessing import shared_memory
import time

os.chdir('D:/depressional_lidar/data/')

site_name = 'bradford'
basin = 'all_basins'
smoothing_window = 1000

# So far resampling resolutions are 'native', 2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50
resampling_resolution = 50

thresholds = np.arange(-1.5, 1.5, 0.02)
batch_size = 1
min_feature_size = 300

if resampling_resolution != 'native':
    min_feature_size = max(1, int(round(300 / resampling_resolution**2)))  # ensure at least 1 pixel minimum size

print(min_feature_size)

if resampling_resolution != 'native':
    detrend_path = f'./{site_name}/in_data/resampled_DEMs/detrended_dem_{basin}_resampled{resampling_resolution}_size{smoothing_window}.tif'
else:
    detrend_path = f'./{site_name}/in_data/detrended_dem_{basin}_size{smoothing_window}.tif'


# %% 2.0 Get summary stats for elevation and make a histogram

with rio.open(detrend_path) as src:
    print(src.meta)
    data = src.read(1, masked=True)
    data = data * 0.3048 # Convert feet to meters in elevation data
    flat = data.compressed()  

    total_pix = len(flat)
    below_neg_3 = (np.sum(flat < -3) / total_pix * 100)
    above_3 = (np.sum(flat > 3) / total_pix * 100)
    below_neg_1 = (np.sum(flat < -1) / total_pix * 100)
    above_1 = (np.sum(flat > 1) / total_pix * 100)
    below_0 = (np.sum(flat < 0) / total_pix * 100)
    print(f'{below_neg_3:.2f} % of pixels below -3 meters')
    print(f'{above_3:.2f} % of pixels above 3 meters')
    print(f'{below_neg_1:.2f} % of pixels below -1 meters')
    print(f'{above_1:.2f} % of pixels above 1 meters')
    print(f'{below_0:.2f} % of pixels below 0 meters')

    # Trim the data to only include elevations between -3 and 3
    trimmed = flat[(flat >= -3) & (flat <= 3)]

    # Create the histogram using the trimmed data
    plt.hist(trimmed, bins=100, color='blue', edgecolor='black')
    plt.title("Histogram of Detrended DEM Elevations")
    plt.xlabel("Elevation (meters)")
    plt.xlim(-3, 3)
    plt
    plt.ylabel("Frequency")
    plt.show()

    # Write a csv with da/dz for the DEM

# %% 3.0 Prep the DEM in shared memory for multiprocessing

with rio.open(detrend_path) as src:
    dem_local = src.read(1, masked=True) * 0.3048  # Convert feet to meters

shm = shared_memory.SharedMemory(create=True, size=dem_local.nbytes)
dem = np.ndarray(dem_local.shape, dtype=dem_local.dtype, buffer=shm.buf)
dem[:] = dem_local[:]

# %% 4.0 Worker function to process each elevation threshold

def analyze_threshold(t, min_feature_size):
    """
    Analyze the DEM for depressions below a given elevation threshold.
    """
    binary = np.logical_and(dem < t, dem > -9000)
    binary = morphology.remove_small_objects(binary, min_size=min_feature_size, connectivity=1)  # Used more strict connectivity for less funky shapes
    labels = measure.label(binary, connectivity=1) # NOTE: changed from 2 -> 1??
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
        median_feature_pix=np.median(pixels),
        std_feature_pix=np.std(pixels),
    )

# %% 5.0 Run the analysis in parallel

results = []
start_time = time.time()

for i in range(0, len(thresholds), batch_size):
    batch = thresholds[i:i + batch_size]
    batch_start = time.time()
    r = Parallel(
        n_jobs=batch_size, 
        backend='loky', # Bypasses the GIL for CPU-bound tasks
        prefer='processes')( # Processes forks the subprocesses, becuse workload is CPU heavy. Would use threads for IO heavy tasks (e.g., file reads)
            delayed(analyze_threshold)(t, min_feature_size) for t in batch)
    
    results.extend(r)
    batch_elapsed = time.time() - batch_start
    print(f'Processed batch {i // batch_size + 1} in {batch_elapsed:.2f} seconds')

total_elapsed = time.time() - start_time
print(f'Total elapsed time: {total_elapsed:.2f} seconds')

# clean up shared memory
shm.close()
shm.unlink()


# %% 6.0 Create a DataFrame with the results

US_SURVEY_FOOT_TO_METER = 0.304800609601219
pixel_size_m = 2.5 * US_SURVEY_FOOT_TO_METER * resampling_resolution
pixel_area_m2 = pixel_size_m**2

out_df = pd.DataFrame(results)
out_df['inundated_frac'] = out_df['total_inundated_pix'] / total_pix * 100
out_df['inundated_area_m2'] = out_df['total_inundated_pix'] * pixel_area_m2
out_df['mean_feature_area_m2'] = out_df['mean_feature_pix'] * pixel_area_m2
out_df['median_feature_area_m2'] = out_df['median_feature_pix'] * pixel_area_m2
out_df['std_feature_area_m2'] = out_df['std_feature_pix'] * pixel_area_m2
out_df['total_perimeter_m'] = out_df['total_perimeter'] * pixel_size_m
out_df['total_perimeter_crofton_m'] = out_df['total_perimeter_crofton'] * pixel_size_m

out_df.drop(columns=
            ['total_inundated_pix', 'mean_feature_pix', 'std_feature_pix',
             'total_perimeter', 'total_perimeter_crofton'], 
        inplace=True
)

# TODO: change this to dynamically accomodate different smoothing windows???
out_df.to_csv(f'./{site_name}/out_data/{site_name}_region_props_on_depressions_{resampling_resolution}.csv', index=False)

# %%
