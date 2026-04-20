# %% 1.0 Libraries and file paths

import rasterio as rio
from scipy.ndimage import gaussian_filter
import numpy as np

src_dem = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_USGS.tif'
out_dem = 'D:/depressional_lidar/data/bradford/in_data/bradford_dem_smoothed_7.tif'

# %% 2.0 Run gaussian smoothing filter.

sigma = 7

with rio.open(src_dem) as src:
    profile = src.profile.copy()
    profile.update(dtype='float32')
    data = src.read(1).astype('float32')
    nodata = src.nodata

mask = (data == nodata) if nodata is not None else np.zeros(data.shape, dtype=bool)
data[mask] = np.nan

smoothed = gaussian_filter(np.where(np.isnan(data), 0, data), sigma=sigma)
weight = gaussian_filter(np.where(np.isnan(data), 0, 1.0), sigma=sigma)
smoothed = np.where(weight > 0, smoothed / weight, np.nan)

if nodata is not None:
    smoothed[mask] = nodata

with rio.open(out_dem, 'w', **profile) as dst:
    dst.write(smoothed.astype('float32'), 1)

print(f"Saved smoothed DEM to {out_dem}")

# %%
