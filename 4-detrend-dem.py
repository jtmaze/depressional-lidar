# %% 1.0 Libraries and packages

import rasterio as rio
import numpy as np

site_name = 'bradford'
basin = 'all_basins'

# NOTE: Use 1000 LXW grid cells for the smoothed DEM
dem_moving_avg = f'./{site_name}/out_data/dem_averaged_1000.tif'
dem_smoothed = f'./{site_name}/out_data/smoothed_dems/dem_smoothed_all_basins.tif'

# %% 2.0 De-trend the original DEM by subtracting by the moving average

with rio.open(dem_smoothed) as src1, rio.open(dem_moving_avg) as src2:

    dem = src1.read(1, masked=True)
    avg_topo = src2.read(1, masked=True)
    detrended_dem = dem - avg_topo

    nodata_val = src1.meta.get('nodata')
    detrended_dem_filled = detrended_dem.filled(nodata_val)
    
    out_path = f'./{site_name}/out_data/detrended_dem_{basin}.tif'
    out_meta = src1.meta.copy()
    out_meta.update(dtype='float32', nodata=nodata_val)

    with rio.open(out_path, 'w', **out_meta) as dst:
        dst.write(detrended_dem_filled.astype('float32'), 1)


# %%
