# %% 1.0 Libraries and file paths

import rasterio as rio
from scipy.ndimage import uniform_filter, zoom
from skimage import measure
from skimage.filters import gaussian
import numpy as np
import geopandas as gpd

from shapely.geometry import shape
from rasterio import features


site = 'osbs'
dataset = 'neon_apr2023'
data_dir = f'D:/depressional_lidar/data/{site}/'

src_path = f'{data_dir}/in_data/{site}_dem_cleaned_{dataset}.tif'
nwi_path = f'{data_dir}/in_data/original_basins/osbs_nwi_polygons.shp'
temp_path = f'{data_dir}/temp/{site}_{dataset}_temp1.tif'
temp_gdf_path = f'{data_dir}/temp/gdf_test_data.shp'

# params
size = 5
sd_thresh = 0.01
slope_thresh = 0.005
min_area = 10

# %% 2.0 Read the DEM and apply a light gaussian filter

with rio.open(src_path) as src:
    # Read as a masked array to capture the true nodata values
    dem_masked = src.read(1, masked=True)
    profile = src.profile
    nodata_val = src.nodata
    transform = src.transform

# Mask the DEM to NWI data
nwi_data = gpd.read_file(nwi_path)
nwi_data = nwi_data.to_crs(profile['crs'])

# Rasterize NWI polygons to a boolean mask
nwi_mask = features.rasterize(
    shapes=nwi_data.geometry,
    out_shape=dem_masked.shape,
    transform=transform,
    fill=0,
    default_value=1,
    dtype='uint8'
).astype(bool)

dem = dem_masked.filled(np.nan)

dem_filled_for_smooth = np.where(np.isfinite(dem), dem, 0)
dem_smooth = gaussian(dem_filled_for_smooth, sigma=1, preserve_range=True)
dem_smooth[~nwi_mask] = np.nan

# %% 3.0 Compute local standard deviation and slope as thresholds

# %% 3.1 COmpute the local standard deviation
valid = np.isfinite(dem_smooth)
dem_filled = np.where(valid, dem_smooth, 0)

frac_valid = uniform_filter(valid.astype(np.float32), size=size)
mean = uniform_filter(dem_filled, size=size) / frac_valid
mean_sq = uniform_filter(dem_filled**2, size=size) / frac_valid

var = mean_sq - mean**2
var[var < 0] = 0

local_sd = np.sqrt(var)
local_sd[frac_valid < (3 / size**2)] = np.nan

# %% 3.2 Compute the slope

# Downsample by factor of 3 (e.g., 1m → 3m pixels)
factor = 1/3
dem_coarse = zoom(dem_filled, factor, order=1)

xres = transform.a
yres = abs(transform.e)

dzdx = np.gradient(dem_coarse, axis=1) / (xres / factor)
dzdy = np.gradient(dem_coarse, axis=0) / (yres / factor)
slope_coarse = np.sqrt(dzdx**2 + dzdy**2)

# Upsample back to original shape
original_shape = dem_filled.shape
slope = zoom(slope_coarse, zoom=[original_shape[0]/slope_coarse.shape[0], 
                                  original_shape[1]/slope_coarse.shape[1]], order=1)

# %% 3.3 Apply thresholds to determine flooded area

flat_mask = (local_sd < sd_thresh) & (slope < slope_thresh) & nwi_mask

# --- save masks ---
mask_profile = profile.copy()
mask_profile.update(dtype=rio.uint8, count=1, nodata=None)

with rio.open(f'{data_dir}/temp/sd_mask_test.tif', 'w', **mask_profile) as dst:
    dst.write((local_sd < sd_thresh).astype(rio.uint8), 1)

with rio.open(f'{data_dir}/temp/slope_mask_test.tif', 'w', **mask_profile) as dst:
    dst.write((slope < slope_thresh).astype(rio.uint8), 1)


# %% 3.4 Use region props to find

labels = measure.label(flat_mask, connectivity=1)
props = measure.regionprops(labels, intensity_image=dem)


# %%
records = []

for prop in props:
    if prop.area < min_area:  
        continue

    est_wse = np.percentile(prop.intensity_image[prop.image], 25)

    min_row, min_col, max_row, max_col = prop.bbox
    window = rio.windows.Window(col_off=min_col, row_off=min_row, width=max_col-min_col, height=max_row-min_row)
    window_transform = rio.windows.transform(window, transform)
    
    mask = prop.image.astype('uint8')
    shapes = list(features.shapes(mask, mask=mask, transform=window_transform))
    
    if shapes:
        geom, val = shapes[0]
        poly = shape(geom)

        records.append({
            'idx': prop.label,
            'est_wse': est_wse,
            'area': prop.area,
            'geometry': poly
        })

gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=profile['crs'])

test_path = f'{data_dir}/temp/wse_test_data.shp'
gdf.to_file(test_path, index=False)
# %%
print(len(gdf))

# %%
