# %% Libraries and file paths
import os
import numpy as np
import rasterio as rio
from rasterio.features import shapes
import geopandas as gpd
from skimage import morphology
from shapely.geometry import Polygon

os.chdir('D:/depressional_lidar/data/')

site_name = 'bradford'
basin = 'all_basins'
smoothing_window = 1_000
resampling_resolution = 2 
min_feature_size = 300

if resampling_resolution != 'native':
    detrend_path = f'./{site_name}/in_data/resampled_detrended_DEMs/detrended_dem_{basin}_resampled{resampling_resolution}_size{smoothing_window}.tif'
else:
    detrend_path = f'./{site_name}/in_data/detrended_dems/detrended_dem_{basin}_size{smoothing_window}.tif'

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

    with rio.open(f'{out_path}.tif', 'w', **prof) as dst:
        dst.write(bool_mask.astype(np.uint8), 1)
    print(f'Wrote inundation mask to {out_path}.tif')
    np.savetxt(
        f'{out_path}.csv', 
        bool_mask.astype(np.uint8), 
        delimiter=',',
        fmt='%d'
    )
    print(f'Wrote txt file to {out_path}.csv')


def write_inundated_polygons(
    bool_mask: np.array,
    array_transform: dict,
    out_path: str,
    out_crs: str,
):
    mask_uint8 = bool_mask.astype(np.uint8)
    features = (
        {"properties": {"label": v}, "geometry": geom}
        for geom, v in shapes(mask_uint8, transform=array_transform)
    )

    gdf = gpd.GeoDataFrame.from_features(list(features), crs=out_crs)
    gdf = gdf[gdf['label'] > 0] # Filter out non-inundated geometries
    gdf.drop(columns='label', inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    gdf.to_file(f'{out_path}.shp')
    gdf.to_csv(f'{out_path}.csv')

# %% 

write_thresholds = [-0.80, -0.20]
out_dir = f'./{site_name}/out_data/modeled_inundations/'

with rio.open(detrend_path) as src:
    dem = src.read(1, masked=True)
    profile = src.profile
    out_crs = profile.get('crs')
    transform = profile.get('transform')

for t in write_thresholds:
    mask = dem < t
    mask_cleaned = morphology.remove_small_objects(mask, min_size=min_feature_size, connectivity=1)
    out_path_raster = f'{out_dir}inundation_mask_smoothed{smoothing_window}_resampled{resampling_resolution}_{t:.2f}m'
    write_binary_inundation_raster(mask_cleaned, out_path_raster, profile)
    out_path_polygon = f'{out_dir}inundation_polygons_smoothed{smoothing_window}_resampled{resampling_resolution}_{t:.2f}m'
    write_inundated_polygons(mask_cleaned, transform, out_path_polygon, out_crs)
    

# %%
