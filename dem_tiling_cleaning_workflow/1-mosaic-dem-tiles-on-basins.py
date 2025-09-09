"""
This script takes the raw DEM tiles downloaded from Florida's LiDAR and selects tiles within.
It then mosaics the DEM tiles that are within the basin shapes and masks the mosaic to the basin shapes.
"""

# %% 1.0 Libaries and Directories
import os
import glob 
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

import geopandas as gpd

target_crs = 'EPSG:26917'
site_name = 'osbs' # bradford or osbs
lidar_data = 'sep2016' # multiple neon flights at OSBS (e.g., sep2016), tbd which on works best
US_SURVEY_FOOT_TO_METER = 0.304800609601219

os.chdir('D:/depressional_lidar/data/')

if lidar_data == 'USGS' and site_name == 'bradford':
    dem_tile_paths = glob.glob(f'./raw_usgs_tiles/*.tif')
    basin_shapes_path = f'./{site_name}/in_data/original_basins/watershed_delineations.shp'
    basin_shapes = gpd.read_file(basin_shapes_path)
    unique_basin_ids = basin_shapes['Basin_Name'].unique().tolist()
    unique_basin_ids.append('all_basins')

elif site_name == 'osbs' and lidar_data == 'USGS':
    dem_tile_paths = glob.glob(f'./raw_usgs_tiles/*.tif')
    basin_shapes_path = f'./{site_name}/in_data/OSBS_boundary.shp'
    basin_shapes = gpd.read_file(basin_shapes_path)
    basin_shapes['Basin_Name'] = 'all_basins'

elif site_name == 'osbs' and 'neon' in lidar_data:
    dem_tile_paths = glob.glob(f'./{site_name}/in_data/raw_DEM_tiles/{lidar_data}/*.tif')
    basin_shapes_path = f'./{site_name}/in_data/OSBS_boundary.shp'
    basin_shapes = gpd.read_file(basin_shapes_path)
    basin_shapes['Basin_Name'] = 'all_basins'

else:
    print('Check your site name_name and/or lidar_data')

# NOTE: Just processing the union of 'all_basins' 
unique_basin_ids = ['all_basins']

# Convert basin shapes to UTM coordinate system
basin_shapes_utm = basin_shapes.estimate_utm_crs(datum_name='WGS 84')
basin_shapes = basin_shapes.to_crs(basin_shapes_utm)

# Buffer/dilate the basin shapes for landscape detrending
basin_shapes['geometry'] = basin_shapes.geometry.buffer(5_000)

# %% 2.0 Run the DEM cropping code for each basin

# 2.1 Crop the DEM for the basin/basin(s)
for basin_id in unique_basin_ids:
    print(f"Processing basin: {basin_id}")
    if basin_id == 'all_basins':
        # For the combined DEM, use the union of all basin geometries
        crop_shape = gpd.GeoDataFrame(geometry=[basin_shapes.union_all()], crs=basin_shapes.crs)
    else:
        crop_shape = basin_shapes[basin_shapes['Basin_Name'] == basin_id]
        crop_shape = crop_shape.reset_index(drop=True)


    # 2.2 Check that each DEM tile has the same crs
    crs_list = []

    for fp in dem_tile_paths:
        with rio.open(fp) as src:
            crs_list.append(src.crs)

    def check_crs_strings(crs_list):
        """
        Check if all coordinate reference system (CRS) strings in the provided list are identical.
        """
        return all(s == crs_list[0] for s in crs_list)

    identical_crs = check_crs_strings(crs_list)
    print(f'Each tile crs is identical: {identical_crs}')
    if not identical_crs:
        unique_crs = set(str(crs) for crs in crs_list)
        print(f"Unique CRS values found: {unique_crs}")

    # 2.3 Check if DEM files are within crop bounds. If within bounds, keep in mosaic list. Discard if outside bounds.
    valid_paths = []
    for fp in dem_tile_paths:
        with rio.open(fp) as src:
            try:
                tile_crs = src.crs
                temp_shape = crop_shape.to_crs(tile_crs)
                crop_geom = temp_shape.geometry
                _ = mask(src, crop_geom, crop=True)
                valid_paths.append(fp)
            except ValueError as e:
                if "not overlap" in str(e):
                    continue
                else:
                    raise

    # 3.0 Mosaic the files that are within the crop bounds
    reprojected_tiles = []
    tile_temp_filepaths = []
    for idx, p in enumerate(valid_paths):
        with rio.open(p) as src:
            # Get affine transform for reprojection based on tgt trands
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
            # Reproject each tile into target crs
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
        # Write the reprojected tiles into temp directory       
        tile_temp_filepath = f'./{site_name}/temp/reprojected_tile_{basin_id}_{idx}.tif'
        tile_temp_filepaths.append(tile_temp_filepath)
        with rio.open(tile_temp_filepath, 'w', **meta) as dst:
            dst.write(reprojected_data)

        # Store the reproject tiles as opened rio datasets for mosaicing
        reprojected_tiles.append(rio.open(tile_temp_filepath))

    # Mosaic the reprojected tiles
    mosaic, out_transform = merge(reprojected_tiles)
    # Close tiles after mosaicing
    for ds in reprojected_tiles:
        ds.close()
    
    # Convert USGS data from meters to feet
    no_data = src.meta.get('nodata')
    print(no_data)
    out_mosaic = (np.where(
        mosaic * US_SURVEY_FOOT_TO_METER < 0,
        no_data,
        mosaic * US_SURVEY_FOOT_TO_METER
    ) if lidar_data == 'USGS' else mosaic) # If data comes from NEON elevation is already in meters
    

    # Update the metadata and write the reprojected mosaic to temp
    with rio.open(valid_paths[0]) as src:
        first_meta = src.meta.copy()

    final_meta = first_meta.copy()
    final_meta.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_transform, 
        'crs': target_crs
    })
    
    mosaic_fp = f'./{site_name}/temp/dem_mosaic_basin_{basin_id}.tif'
    with rio.open(mosaic_fp, 'w', **final_meta) as dst:
        dst.write(out_mosaic)

    # 4.0 Mask the mosaic'd DEM to the study area's buffered shape and write to out_data
    with rio.open(mosaic_fp) as src:

        crop_geom = crop_shape.to_crs(target_crs).geometry
        masked_mosaic, masked_trans = mask(src, crop_geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            'height': masked_mosaic.shape[1],
            'width': masked_mosaic.shape[2],
            'transform': masked_trans
        })
        
        output_path = f'./{site_name}/in_data/dem_mosaic_basin_{basin_id}.tif'

        # Accomodating multiple flights into osbs file name. 
        if site_name == 'osbs':
            output_path = f'./{site_name}/in_data/dem_mosaic_basin_{basin_id}_{lidar_data}.tif'
        with rio.open(output_path, 'w', **out_meta) as dst:
            dst.write(masked_mosaic)

    # 5.0 Clean up temp directory
    for temp_fp in tile_temp_filepaths:
        try:
            os.remove(temp_fp)
        except Exception as e:
            continue

    if basin_id != 'all_basins':
        try:
            os.remove(mosaic_fp)
        except Exception as e:
            continue

# %%
