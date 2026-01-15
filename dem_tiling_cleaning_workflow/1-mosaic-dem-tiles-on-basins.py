"""
This script takes the raw DEM tiles downloaded from Florida's LiDAR and selects tiles within.
It then mosaics the DEM tiles that are within the basin shapes and masks the mosaic to the basin shapes.
"""

# %% 1.0 Libraries and Directories
import os
import glob 
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

import geopandas as gpd

target_crs = 'EPSG:26917'
site_name = 'bradford'  # bradford or osbs
lidar_data = 'USGS'  # multiple neon flights at OSBS (e.g., sep2016), tbd which works best
US_SURVEY_FOOT_TO_METER = 0.304800609601219

os.chdir('D:/depressional_lidar/data/')

if lidar_data == 'USGS' and site_name == 'bradford':
    dem_tile_paths = glob.glob('./raw_usgs_tiles/*.tif')
    basin_shapes_path = f'./{site_name}/in_data/original_basins/watershed_delineations.shp'

elif site_name == 'osbs' and lidar_data == 'USGS':
    dem_tile_paths = glob.glob('./raw_usgs_tiles/*.tif')
    basin_shapes_path = f'./{site_name}/in_data/OSBS_boundary.shp'

elif site_name == 'osbs' and 'neon' in lidar_data:
    dem_tile_paths = glob.glob(f'./{site_name}/in_data/raw_DEM_tiles/{lidar_data}/*.tif')
    basin_shapes_path = f'./{site_name}/in_data/OSBS_boundary.shp'

else:
    raise ValueError('Check your site_name and/or lidar_data')

basin_shapes = gpd.read_file(basin_shapes_path)

# Convert basin shapes to UTM coordinate system and buffer for landscape detrending
basin_shapes = basin_shapes.to_crs(basin_shapes.estimate_utm_crs(datum_name='NAD 83')) #NOTE: Is OSBS NAD83??
basin_shapes['geometry'] = basin_shapes.geometry.buffer(2_000)

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

# %% 3.0 Reproject and mosaic tiles
# Get nodata value from first tile before reprojection loop
with rio.open(valid_paths[0]) as src:
    no_data = src.meta.get('nodata')

reprojected_tiles = []
tile_temp_filepaths = []

for idx, p in enumerate(valid_paths):
    with rio.open(p) as src:
        trans, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        meta = src.meta.copy()
        meta.update({
            'crs': target_crs,
            'transform': trans,
            'width': width,
            'height': height,
            'driver': 'GTiff'
        })

        reprojected_data = np.zeros((src.count, height, width), dtype=meta['dtype'])
        reproject(
            source=rio.band(src, 1),
            destination=reprojected_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=trans,
            dst_crs=target_crs,
            resampling=Resampling.cubic
        )

    tile_temp_filepath = f'./{site_name}/temp/reprojected_tile_{idx}.tif'
    tile_temp_filepaths.append(tile_temp_filepath)
    with rio.open(tile_temp_filepath, 'w', **meta) as dst:
        dst.write(reprojected_data)

    reprojected_tiles.append(rio.open(tile_temp_filepath))

# Mosaic the reprojected tiles
mosaic, out_transform = merge(reprojected_tiles)

# Close tiles after mosaicing
for ds in reprojected_tiles:
    ds.close()

# Convert USGS data from US survey feet to meters
if lidar_data == 'USGS':
    out_mosaic = np.where(
        mosaic * US_SURVEY_FOOT_TO_METER < 0,
        no_data,
        mosaic * US_SURVEY_FOOT_TO_METER
    )
else:
    out_mosaic = mosaic  # NEON data already in meters

# %% 4.0 Write mosaic to temp file
with rio.open(valid_paths[0]) as src:
    first_meta = src.meta.copy()

final_meta = first_meta.copy()
final_meta.update({
    'height': mosaic.shape[1],
    'width': mosaic.shape[2],
    'transform': out_transform,
    'crs': target_crs
})

mosaic_fp = f'./{site_name}/temp/dem_mosaic_all_basins.tif'
with rio.open(mosaic_fp, 'w', **final_meta) as dst:
    dst.write(out_mosaic)

# %% 5.0 Mask the mosaic to the study area's buffered shape and write output
with rio.open(mosaic_fp) as src:
    crop_geom = crop_shape.to_crs(target_crs).geometry
    masked_mosaic, masked_trans = mask(src, crop_geom, crop=True)
    out_meta = src.meta.copy()
    out_meta.update({
        'height': masked_mosaic.shape[1],
        'width': masked_mosaic.shape[2],
        'transform': masked_trans
    })

output_path = f'./{site_name}/in_data/dem_mosaic_all_basins.tif'
if site_name == 'osbs':
    output_path = f'./{site_name}/in_data/dem_mosaic_all_basins_{lidar_data}.tif'

with rio.open(output_path, 'w', **out_meta) as dst:
    dst.write(masked_mosaic)

# %% 6.0 Clean up temp directory
for temp_fp in tile_temp_filepaths:
    try:
        os.remove(temp_fp)
    except Exception:
        pass

# %%

