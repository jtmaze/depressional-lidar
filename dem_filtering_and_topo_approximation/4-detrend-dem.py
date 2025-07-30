# %% 1.0 Libraries and packages
import os
import rasterio as rio

site = 'osbs'
basin = 'all_basins'
smoothing_window = 2000 # NOTE: Use 1000 LXW grid cells for the smoothed DEM in main workflow

os.chdir('D:/depressional_lidar/data/')

dem_moving_avg = f'./{site}/in_data/dem_averaged_{smoothing_window}.tif'
dem_no_veg = f'./{site}/in_data/{site}_DEM_cleaned_veg.tif'

# %% 2.0 De-trend the original DEM by subtracting by the moving average

with rio.open(dem_no_veg) as src1, rio.open(dem_moving_avg) as src2:

    dem = src1.read(1, masked=True)
    avg_topo = src2.read(1, masked=True)
    detrended_dem = dem - avg_topo

    nodata_val = src1.meta.get('nodata')
    detrended_dem_filled = detrended_dem.filled(nodata_val)
    
    out_path = f'./{site}/in_data/detrended_dem_{basin}_size{smoothing_window}.tif'
    out_meta = src1.meta.copy()
    out_meta.update(dtype='float32', nodata=nodata_val)

    with rio.open(out_path, 'w', **out_meta) as dst:
        dst.write(detrended_dem_filled.astype('float32'), 1)


# %%
