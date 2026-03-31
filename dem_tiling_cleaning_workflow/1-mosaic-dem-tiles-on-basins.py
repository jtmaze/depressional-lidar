"""
This script takes the raw DEM tiles downloaded from Florida's LiDAR and selects tiles within.
It then mosaics the DEM tiles that are within the basin shapes and masks the mosaic to the basin shapes.
"""
# %% 1.0 Libraries and Directories
import os
import glob 
import math
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_origin

import geopandas as gpd

target_crs = 'EPSG:26918'
site_name = 'delmarva'  # bradford or osbs, or delmarva
sub_site_name = 'JL' # delmarva has two sites None for osbs and bradford

# multiple neon flights at OSBS (e.g., sep2016)
# For delmarva, use USGS_1m, because data is UTM
lidar_data = 'USGS_1m'  
US_SURVEY_FOOT_TO_METER = 0.304800609601219 # FL USGS Lidar data is in feet

os.chdir('D:/depressional_lidar/data/')

if lidar_data == 'USGS' and site_name == 'bradford':
    dem_tile_paths = glob.glob('./raw_usgs_tiles_fl/*.tif')
    basin_shapes_path = f'./{site_name}/in_data/original_basins/watershed_delineations.shp'

elif site_name == 'osbs' and lidar_data == 'USGS':
    dem_tile_paths = glob.glob('./raw_usgs_tiles_fl/*.tif')
    basin_shapes_path = f'./{site_name}/OSBS_boundary.shp'

elif site_name == 'osbs' and 'neon' in lidar_data:
    dem_tile_paths = glob.glob(f'./{site_name}/in_data/raw_DEM_tiles_fl/{lidar_data}/*DTM.tif')
    basin_shapes_path = f'./{site_name}/OSBS_boundary.shp'

elif site_name == 'delmarva' and lidar_data == 'USGS_1m':
    dem_tile_paths = glob.glob('./raw_usgs_tiles_de/*.tif')
    basin_shapes_path = f'./{site_name}/in_data/sites_and_boundaries/{sub_site_name}_boundary.shp'

else:
    raise ValueError('Check your site_name and/or lidar_data')

basin_shapes = gpd.read_file(basin_shapes_path)

print(basin_shapes)

# %%

# Convert basin shapes to UTM coordinate system and buffer for landscape detrending
basin_shapes = basin_shapes.to_crs(basin_shapes.estimate_utm_crs(datum_name='WGS 84')) #NOTE: Is OSBS NAD83??
basin_shapes['geometry'] = basin_shapes.geometry.buffer(2_500)

# Create unified crop shape from all basins
crop_shape = gpd.GeoDataFrame(geometry=[basin_shapes.union_all()], crs=basin_shapes.crs)

# %% 2.0 Find DEM tiles that overlap with crop bounds
valid_paths = []
crs_list = []
for fp in dem_tile_paths:
    with rio.open(fp) as src:
        try:
            temp_shape = crop_shape.to_crs(src.crs)
            _ = mask(src, temp_shape.geometry, crop=True)
            valid_paths.append(fp)
            crs_list.append(src.crs)
        except ValueError as e:
            if "not overlap" not in str(e):
                raise

print(f"Found {len(valid_paths)} tiles overlapping with study area")

# Check that overlapping tiles have the same CRS
identical_crs = all(crs == crs_list[0] for crs in crs_list)
print(f'Each tile CRS is identical: {identical_crs}')
if not identical_crs:
    unique_crs = set(str(crs) for crs in crs_list)
    print(f"Unique CRS values found: {unique_crs}")

# Check nodata consistency
nodata_list = []
unique_nodata = set(nodata_list)
print(f"Unique nodata values found: {unique_nodata}")

with rio.open(valid_paths[0]) as src0:
    no_data = src0.nodata
    src_dtype = src0.dtypes[0]
    src_res_x, src_res_y = src0.res

if lidar_data == "USGS":
    src_res_x = src_res_x * US_SURVEY_FOOT_TO_METER
    src_res_y = src_res_y * US_SURVEY_FOOT_TO_METER

# %% 3.0 Build one shared target grid in target_crs

# Use source resolution magnitude as target resolution
# For north-up rasters, src.res is usually positive magnitudes
target_res_x = abs(src_res_x)
target_res_y = abs(src_res_y)

# Compute the unioned bounds of all valid tiles in target_crs
all_bounds = []
for fp in valid_paths:
    with rio.open(fp) as src:
        b = transform_bounds(src.crs, target_crs, *src.bounds, densify_pts=21)
        all_bounds.append(b)

left = min(b[0] for b in all_bounds)
bottom = min(b[1] for b in all_bounds)
right = max(b[2] for b in all_bounds)
top = max(b[3] for b in all_bounds)

# Snap the mosaic bounds to the target resolution
left_snap = math.floor(left / target_res_x) * target_res_x
right_snap = math.ceil(right / target_res_x) * target_res_x
bottom_snap = math.floor(bottom / target_res_y) * target_res_y
top_snap = math.ceil(top / target_res_y) * target_res_y

dst_width = int(round((right_snap - left_snap) / target_res_x))
dst_height = int(round((top_snap - bottom_snap) / target_res_y))
dst_transform = from_origin(left_snap, top_snap, target_res_x, target_res_y)

print("Shared target grid:")
print(f"  resolution: ({target_res_x}, {target_res_y})")
print(f"  bounds: ({left_snap}, {bottom_snap}, {right_snap}, {top_snap})")
print(f"  shape: ({dst_height}, {dst_width})")

# %% 4.0 Reproject and mosaic tiles

reprojected_tiles = []
tile_temp_filepaths = []

for idx, p in enumerate(valid_paths):
    with rio.open(p) as src:
        meta = src.meta.copy()
        meta.update({
            'crs': target_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'driver': 'GTiff',
            'nodata': no_data
        })

        reprojected_data = np.full((1, dst_height, dst_width), no_data, dtype=src_dtype)

        reproject(
            source=rio.band(src, 1),
            destination=reprojected_data[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.cubic,
            src_nodata=no_data,
            dst_nodata=no_data
        )

    tile_temp_filepath = f'./{site_name}/temp/reprojected_tile_{idx}.tif'
    tile_temp_filepaths.append(tile_temp_filepath)

    with rio.open(tile_temp_filepath, 'w', **meta) as dst:
        dst.write(reprojected_data)

    reprojected_tiles.append(rio.open(tile_temp_filepath))

    print(f"Processed {idx + 1} of {len(valid_paths)}")

# %% 5.0 Mosaic the aligned tiles

mosaic, out_transform = merge(reprojected_tiles, nodata=no_data)

for ds in reprojected_tiles:
    ds.close()

# %% 6.0 Convert USGS data from US survey feet to meters
if lidar_data == 'USGS':
    out_mosaic = np.where(
        mosaic == no_data,
        no_data,
        mosaic * US_SURVEY_FOOT_TO_METER
    ).astype(np.float32)
else:
    out_mosaic = mosaic.astype(np.float32)


# %% 7.0 Write mosaic to temp file

with rio.open(valid_paths[0]) as src:
    first_meta = src.meta.copy()

final_meta = first_meta.copy()
final_meta.update({
    'driver': 'GTiff',
    'height': out_mosaic.shape[1],
    'width': out_mosaic.shape[2],
    'transform': out_transform,
    'crs': target_crs,
    'nodata': no_data,
    'dtype': 'float32'
})


if sub_site_name is None:
    mosaic_fp = f'./{site_name}/temp/dem_mosaic_all_basins.tif'
else:
    mosaic_fp = f'./{site_name}/temp/dem_mosaic_{sub_site_name}.tif'

with rio.open(mosaic_fp, 'w', **final_meta) as dst:
    dst.write(out_mosaic)

# %% 8.0 Mask the mosaic to the study area's buffered shape and write output
with rio.open(mosaic_fp) as src:
    crop_geom = crop_shape.to_crs(target_crs).geometry
    masked_mosaic, masked_trans = mask(src, crop_geom, crop=True, nodata=no_data)

    out_meta = src.meta.copy()
    out_meta.update({
        'height': masked_mosaic.shape[1],
        'width': masked_mosaic.shape[2],
        'transform': masked_trans,
        'nodata': no_data
    })

output_path = f'./{site_name}/in_data/dem_mosaic_all_basins.tif'
if site_name == 'osbs':
    output_path = f'./{site_name}/in_data/dem_mosaic_all_basins_{lidar_data}.tif'
if sub_site_name is not None:
    output_path = f'./{site_name}/in_data/dem_mosaic_{sub_site_name}.tif'

with rio.open(output_path, 'w', **out_meta) as dst:
    dst.write(masked_mosaic)

# %% 9.0 Clean up temp directory
for temp_fp in tile_temp_filepaths:
    try:
        os.remove(temp_fp)
    except Exception:
        pass

# %%

