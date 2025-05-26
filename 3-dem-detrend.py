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

detrend_window = 250
mosaic_dem_path = './temp/dem_mosaic_basin_all_basins.tif'
basin_shapes = gpd.read_file('./in_data/Final_Basins/Final_Basins.shp')
bradford_shape = gpd.GeoDataFrame(geometry=[basin_shapes.union_all()], crs=basin_shapes.crs)

# %% 

with rio.open(mosaic_dem_path) as src:
    
    dem = src.read(1, masked=True)
    target_crs = dem.crs
    profile = src.profile
    
    def nan_mean_filter(window):
        valid = window[~np.isnan(window)]
        if len(valid) > 0: 
            return np.mean(valid)
        else:
            return np.nan
        
    dem_with_nans = dem.filled(np.nan)

    averaged_dem = generic_filter(
        dem_with_nans, 
        nan_mean_filter,
        size=detrend_window,
        mode='nearest'
    )

    # Reapply the original mask
    averaged_dem = np.ma.array(averaged_dem, mask=dem.mask)
    
    # Update profile and save
    profile.update(dtype='float32', nodata=profile.get('nodata', None))
    
    temp_path = f'temp/dem_trend.tif'
    with rio.open(temp_path, 'w', **profile) as dst:
        dst.write(averaged_dem.astype('float32'), 1)

    """
    Re-read the averaged DEM and crop it to the basin shapes. 
    """

    crop_shape = bradford_shape.to_crs(target_crs).geometry

    with rio.open(temp_path) as src2:
        masked_detrend, masked_trans = mask(src2, crop_shape, crop=True)
        out_meta = src2.meta.copy()
        out_meta.update({
            'height': masked_detrend.shape[1],
            'width': masked_detrend.shape[2],
            'transform': masked_trans
        })

        output_path = f'./out_data/dem_detrended.tif'
        with rio.open(output_path, 'w', **out_meta) as dst:
            dst.write(masked_detrend)
# %%
