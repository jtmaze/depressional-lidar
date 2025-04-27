"""
This script takes the raw DEM tiles downloaded from Florida's LiDAR and selects tiles within.
It then mosaics the DEM tiles that are within the basin shapes and masks the mosaic to the basin shapes.
"""

# %% 1.0 Libaries and Directories

import glob 
import pprint as pp

import rasterio as rio
from rasterio.mask import mask
from rasterio.merge import merge

import geopandas as gpd


dem_tile_paths = glob.glob('./in_data/raw_DEM_tiles/*.tif')
basin_shapes = gpd.read_file('./in_data/Final_Basins/Final_Basins.shp')
unique_basin_ids = basin_shapes['Basin_Name'].unique().tolist()
unique_basin_ids = unique_basin_ids.append('all_basins')
unique_basin_ids = ['all_basins'] 
# %% Run the code for each basin

# Crop the DEM for each basin
for basin_id in unique_basin_ids:
    print(f"Processing basin: {basin_id}")
    if basin_id == 'all_basins':
        # For the combined DEM, use the union of all basin geometries
        crop_shape = gpd.GeoDataFrame(geometry=[basin_shapes.unary_union], crs=basin_shapes.crs)
    else:
        crop_shape = basin_shapes[basin_shapes['Basin_Name'] == basin_id]
        crop_shape = crop_shape.reset_index(drop=True)

    # 1.1 Check the DEM tile paths
    with rio.open(dem_tile_paths[0]) as src:
        target_crs = src.crs

    crop_shape_reproj = crop_shape.to_crs(target_crs)
    print(crop_shape_reproj.crs)

    # 2.0 Check that each DEM has the same crs

    crs_list = []

    for fp in dem_tile_paths:
        with rio.open(fp) as src:
            crs_list.append(src.crs)

    def check_crs_strings(crs_list):
        """
        Check if all coordinate reference system (CRS) strings in the provided list are identical.
        """
        check_list = all(s == crs_list[0] for s in crs_list)
        return check_list

    print(f'Each tile crs is identical: {check_crs_strings(crs_list)}')

    # 3.0 Mosaic the files, which are within the crop bounds

    crop_geom = crop_shape_reproj.geometry
    print(crop_geom)

    valid_paths = []
    for fp in dem_tile_paths:
        with rio.open(fp) as src:
            try:
                _ = mask(src, crop_geom, crop=True)
                valid_paths.append(fp)
            except ValueError as e:
                if "not overlap" in str(e):
                    print(f"Skipped {fp}, not overlapping crop shape")
                else:
                    raise


    src_files_to_mosaic = [rio.open(fp) for fp in valid_paths]

    mosaic, out_transform = merge(src_files_to_mosaic)
    with rio.open(valid_paths[0]) as src:
        first_meta = src.meta

    final_meta = first_meta.copy()
    final_meta.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_transform
    })

    with rio.open(f'./temp/dem_mosaic_basin_{basin_id}.tif', 'w', **final_meta) as dst:
        dst.write(mosaic)

    # 4.0 Mask the mosaic to the crop shape

    with rio.open(f'./temp/dem_mosaic_basin_{basin_id}.tif') as src:
        masked_mosaic, masked_trans = mask(src, crop_geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            'height': masked_mosaic.shape[1],
            'width': masked_mosaic.shape[2],
            'transform': masked_trans
        })
        
        output_path = f'./out_data/dem_mosaic_basin_{basin_id}.tif'
        with rio.open(output_path, 'w', **out_meta) as dst:
            dst.write(masked_mosaic)

# %%
