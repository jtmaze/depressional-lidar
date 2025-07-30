# %% 1.0 Libraries and paths

import os
import shutil
import whitebox_workflows as wbw

import geopandas as gpd
import rasterio as rio
from rasterio.windows import Window
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.mask import mask

wbe = wbw.WbEnvironment()

site = 'osbs'
lidar_data = 'neon_sep2016.tif' # NOTE: 2016 DEM at OSBS has lowest water levels. Still lots of flooding. 
os.chdir(f'D:/depressional_lidar/data/{site}/')

if site == 'bradford':
    src_path = './in_data/dem_mosaic_basin_all_basins.tif'
elif site == 'osbs': 
    src_path = f'./in_data/dem_mosaic_basin_all_basins_{lidar_data}'

processing_dir = r'C:\Users\jtmaz\Documents\temp'

wbe.working_directory = processing_dir

# %% 2.0 Copy DEM to local drive for faster processing. I/O speeds bad on external hard-drive

print(f"Copying DEM to local drive at {processing_dir}")
shutil.copy2(src_path, processing_dir)
print(f"Copy completed")
base_name = os.path.basename(src_path)
temp_file_path = os.path.join(processing_dir, f'./{base_name}')

wbt_off_terrain_path = os.path.join(processing_dir, f'{site}_DEM_wbt_off_terrain.tif')
final_out_path = os.path.join(processing_dir, f'{site}_DEM_cleaned_veg.tif')

# %% 3.0 Crop the DEM to immediate watershed areas instead of buffered. Imporves processing speeds
# Get boundary
if site == 'bradford':
    boundary_path = f'D:/depressional_lidar/data/{site}/in_data/original_basins/watershed_delineations.shp'
elif site == 'osbs':
    boundary_path = f'D:/depressional_lidar/data/{site}/in_data/OSBS_boundary.shp'

boundary = gpd.read_file(boundary_path)
boundary_union = gpd.GeoDataFrame(geometry=[boundary.union_all()], crs=boundary.crs)

# Read the DEM to get its CRS
with rio.open(temp_file_path) as dem_src:
    dem_crs = dem_src.crs
    
    # Reproject boundary to match DEM CRS
    boundary_union = boundary_union.to_crs(dem_crs)
    
    # Crop/clip the DEM to the boundary
    cropped_file_path = os.path.join(processing_dir, f'{site}_DEM_cropped.tif')
    
    # Build the transform arguments
    out_image, out_transform = mask(dem_src, boundary_union.geometry, crop=True)
    
    # Update metadata for the new raster
    out_meta = dem_src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    # Write the cropped raster
    with rio.open(cropped_file_path, "w", **out_meta) as dest:
        dest.write(out_image)
    
    # Update temp_file_path to use the cropped version
    temp_file_path = cropped_file_path

# %% 4.0 Set-up parameters for vegitation filtering
"""
Parameters
"""
# WBT remove_off_terrain_objects
off_terrian_filter = 7 # number of cells
off_terrian_slope = 0.55 # max slope to smooth

# Custom minima in window based on percentiles
minima_window_size = 3 # NxN cells
mininma_pct_thresh = 10 # percentile for vegitation noise filtering
max_drop = None 
pad = minima_window_size // 2
tile_h, tile_w = 512, 512

# %% 6.0 Run the WhiteBoxTools off terrian objects tool before minima filter. 

# Read the DEM file using the wbe environment instance
dem_raster = wbe.read_raster(temp_file_path)

# Apply the terrain object removal
smoothed_dem = wbe.remove_off_terrain_objects(
    dem=dem_raster,
    filter_size=off_terrian_filter,
    slope_threshold=off_terrian_slope
)

# Save the result to the output path
wbe.write_raster(smoothed_dem, wbt_off_terrain_path)

# %% 7.0 Define functions for focal percentile and chunking the raster (incase its too large for memory)

def chunk_windows(width, height, tile_w, tile_h):
    for row_off in range(0, height, tile_h):
        for col_off in range(0, width, tile_w):
            yield Window(col_off, row_off,
                         min(tile_w,  width  - col_off),
                         min(tile_h,  height - row_off))

def mask_arr(arr: np.ndarray, no_data: float):
    return arr == no_data if no_data is not None else np.isnan(arr)

def focal_percentile(arr, nodata, win_size, pct_thresh):
    """
    Compute pct_thresh-th percentile within a win_size×win_size window.
    Uses edge padding so output has same shape as 'arr'.
    """
    # Mask -> nan
    m = mask_arr(arr, nodata)
    work = np.where(m, np.nan, arr.astype(float))

    # Pad edges so every pixel has a full window
    padded = np.pad(work, pad_width=pad, mode='edge')

    # Sliding windows: (H, W, win, win)
    w = sliding_window_view(padded, (win_size, win_size))
    p = np.nanpercentile(w, pct_thresh, axis=(-2, -1))

    return p  # same shape as arr

# %% 8.0 Run the functions for local minima filter. 

with rio.open(wbt_off_terrain_path) as src:
    print(src.crs)
    sm_profile = src.profile.copy()
    nodata  = sm_profile.get("nodata", None)
    sm_profile.update(dtype='float32')

    with rio.open(final_out_path, 'w', **sm_profile)  as dst_sm:

        for win in chunk_windows(src.width, src.height, tile_w, tile_h):
            # 1) Read core tile
            orig = src.read(1, window=win)

            # Quick skip if all nodata
            if nodata is not None and np.all(orig == nodata):
                dst_sm.write(orig.astype('float32'), 1, window=win)
                continue

            # 2) Focal percentile
            pctv = focal_percentile(orig, nodata,
                                    minima_window_size,
                                    mininma_pct_thresh)

            # 3) Only lower
            diff    = orig - pctv
            lowered = np.where(diff > 0, pctv, orig)

            # 4) Cap drop if desired
            if max_drop is not None:
                lowered = np.where(diff > max_drop, orig - max_drop, lowered)

            # 5) Restore nodata
            if nodata is not None:
                core_mask = mask_arr(orig, nodata)
                #pctv      = np.where(core_mask, nodata, pctv)
                lowered   = np.where(core_mask, nodata, lowered)

            # 6) Write
            dst_sm.write(lowered.astype('float32'), 1, window=win)

# %%
