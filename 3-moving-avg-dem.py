# %% 1.0 Libraries and file paths

"""
Experiementing with options to calculate the smoothed landscape relief
Will be important for tracking depressions across water levels
"""

import rasterio as rio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
import os
from scipy.ndimage import generic_filter 
from scipy.ndimage import zoom

avg_window = 5000
mosaic_dem_path = './temp/dem_mosaic_basin_all_basins.tif'
basin_shapes = gpd.read_file('./in_data/Final_Basins/Final_Basins.shp')
bradford_shape = gpd.GeoDataFrame(geometry=[basin_shapes.union_all()], crs=basin_shapes.crs)

# %% Spatially average the DEM and crop it to the geometry for all basins

with rio.open(mosaic_dem_path) as src:
    
    dem = src.read(1, masked=True)
    target_crs = src.crs
    profile = src.profile
    
    # Removes nan pixels when calculating the rolling average
    def nan_mean_filter(window):
        valid = window[~np.isnan(window)]
        if len(valid) > 0: 
            return np.mean(valid)
        else:
            return np.nan

    # Downsample the DEM, to make computation reasonably fast
    scale_factor = 0.05 
    # NOTE: Downsampled DEM for averaging, because computation was so slow.
    # order = 1 means bilinear interpolation
    dem_upsampled = zoom(dem.filled(np.nan), scale_factor, order=1, mode='nearest') 
    small_window = max(int(avg_window * scale_factor), 3)

    # Take the spatial average of the down-sampled pixels
    averaged_upsampled = generic_filter(
        dem_upsampled, 
        nan_mean_filter,
        size=small_window,
        mode='nearest'
    )

    # Upsample the averaged DEM again back to its original resolution
    averaged_dem = zoom(averaged_upsampled, 1/scale_factor, order=1, mode='nearest')
    
    # Update profile and save
    profile.update(dtype='float32', nodata=profile.get('nodata', None))
    
    temp_path = f'temp/dem_averaged_{avg_window}.tif'
    with rio.open(temp_path, 'w', **profile) as dst:
        dst.write(averaged_dem.astype('float32'), 1)

    """
    Re-read the averaged DEM and crop it to the basin shapes. 
    """

    crop_shape = bradford_shape.to_crs(target_crs).geometry

    with rio.open(temp_path) as src2:
        masked_averaged, masked_trans = mask(src2, crop_shape, crop=True)
        out_meta = src2.meta.copy()
        out_meta.update({
            'height': masked_averaged.shape[1],
            'width': masked_averaged.shape[2],
            'transform': masked_trans
        })

        output_path = f'./out_data/dem_averaged_{avg_window}.tif'
        with rio.open(output_path, 'w', **out_meta) as dst:
            dst.write(masked_averaged)

    # Clean up the disk
    os.remove(temp_path)
# %%
