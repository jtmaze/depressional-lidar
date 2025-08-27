# %% 1.0 Libraries and file paths

import os
import rasterio as rio
from rasterio.enums import Resampling
import numpy

os.chdir('D:/depressional_lidar/data/')

site = 'osbs'
basin = 'all_basins'
smoothing_window = 1000
# Used so far: 2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50
resampling_factors = [2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50]

# Path to the detrended DEM
dem_path = f'./{site}/in_data/detrended_dems/detrended_dem_{basin}_size{smoothing_window}.tif'


# %% 2.0 Resample the data based on the scale factor

for i in resampling_factors:
    out_path = f'./{site}/in_data/resampled_detrended_dems/detrended_dem_{basin}_resampled{i}_size{smoothing_window}.tif'
    print(f'Downsampling to {i}x resolution: {out_path}')
    with rio.open(dem_path) as src:

        profile = src.profile.copy()
        scale = 1 / i

        out_height = int(src.height * scale)
        out_width = int(src.width * scale)

        data = src.read(
            out_shape=(src.count, out_height, out_width), 
            resampling=Resampling.bilinear
        )

        out_trans = src.transform * src.transform.scale(
            (src.width / data.shape[-1]), 
            (src.height / data.shape[-2])
        )

        profile.update({
            'height': out_height,
            'width': out_width,
            'transform': out_trans
        })

        with rio.open(out_path, 'w', **profile) as dst:
            dst.write(data)

# %%
