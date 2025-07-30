import os
import numpy as np
import rasterio as rio

os.chdir('D:/depressional_lidar/data/')

site_name = 'bradford'
basin = 'all_basins'
smoothing_window = 1_000
resampling_resolution = 2 #

if resampling_resolution != 'native':
    detrend_path = f'./{site_name}/in_data/resampled_DEMs/detrended_dem_{basin}_resampled{resampling_resolution}_size{smoothing_window}.tif'
else:
    detrend_path = f'./{site_name}/in_data/detrended_dem_{basin}_size{smoothing_window}.tif'

# %% Function to write the binary raster at defined thresholds

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
out_dir = f'./{site_name}/out_data/modeled_inundations/'

with rio.open(detrend_path) as src:
    dem = src.read(1, masked=True) * 0.3048  # Convert feet to meters
    profile = src.profile

for t in write_thresholds:
    mask = dem < t
    out_path = f'{out_dir}inundation_mask_smoothed{smoothing_window}_resampled{resampling_resolution}_{t:.2f}m.tif'
    write_binary_inundation_raster(mask, out_path, profile)
    print(f'Wrote {out_path}')
