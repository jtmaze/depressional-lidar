# %% 1.0 Libraries and file paths

import os
import rasterio as rio
from rasterio.enums import Resampling
import numpy

os.chdir('D:/depressional_lidar/data/')

site = 'bradford'
basin = 'all_basins'
smoothing_window = 1000
# Used so far: 2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50
resampling_factor = 50

# Path to the 
dem_path = f'./{site}/in_data/detrended_dem_{basin}_size{smoothing_window}.tif'
out_path = f'./{site}/in_data/resampled_dems/detrended_dem_{basin}_resampled{resampling_factor}_size{smoothing_window}.tif'

# %% 2.0 Resample the data based on the scale factor

with rio.open(dem_path) as src:

    profile = src.profile.copy()
    scale = 1 / resampling_factor

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
