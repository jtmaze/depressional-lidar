# %% 1.0 Files and directories

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask

import pprint as pp

well_pts = gpd.read_file('./out_data/wetland_well_points_vRTK.shp')
pp.pp(well_pts.crs)

dem_path = './out_data/dem_mosaic.tif'

with rio.open(dem_path) as src:
    target_crs = src.crs

well_pts_repoj = well_pts.to_crs(target_crs)

# %% 2.0 Clip DEM locally around each well

def calc_local_dem_diff(
    masked_dem: np.array 
):
    valid_data = masked_dem > 0
    pixel_count = np.sum(valid_data)
    mean_elevation = np.mean(masked_dem[valid_data])
    sd_elevation = np.std(masked_dem[valid_data])
    cv_elevation = sd_elevation / mean_elevation
    range_elevation = np.max(masked_dem[valid_data]) - np.min(masked_dem[valid_data]) 
    
    # Calculate the filtered mean elevation. Remove obs > 1 sd
    outlier_filter = (masked_dem < (mean_elevation + 1 * sd_elevation)) & (masked_dem > ((mean_elevation - 1 * sd_elevation)))
    filtered_mean_elevation = np.mean(masked_dem[outlier_filter])

    # Calculate the differenced DEM
    differenced_dem = np.where(valid_data, masked_dem - mean_elevation, masked_dem)

    elevation_stats = {
        'pixel_count': int(pixel_count),
        'mean_elevation': float(mean_elevation),
        'filtered_mean_elevation': float(filtered_mean_elevation),
        'cv_elevation': float(cv_elevation),
        'range_elevation': float(range_elevation)
    }

    return elevation_stats, differenced_dem

def make_local_well_dem(
    well_pts: gpd.GeoDataFrame, # Should already be in the local CRS
    buffer_size: int, # the distance around the well to make the local DEM
    write_raster: bool # should the local raster be written?
):
    """
    Makes a local DEM illustrating the roughness around a well.
    """
    elevation_stats_list = []
    with rio.open(dem_path) as src:
        out_meta = src.meta.copy()
        full_diff_dem = np.zeros((out_meta['count'], 
                                 out_meta['height'], 
                                 out_meta['width']), dtype=out_meta['dtype'])

        full_diff_dem.fill(src.nodata)

    
    for idx, pt in well_pts.iterrows():

        # Get the well's site_id and reformat if necessary. 
        site_id = pt['site_id']
        site_id = site_id.replace('/', '.')

        # Create a buffer around the well point for cropping the DEM
        crop_geom = pt['geometry'].buffer(buffer_size)

        # Crop the DEM to the well's buffer zone
        with rio.open(dem_path) as src:

            window = rio.features.geometry_window(src, [crop_geom])
            window_transform = rio.windows.transform(window, src.transform)
            masked, masked_transform = mask(src, [crop_geom], crop=True)

            elevation_stats, dem_diff = calc_local_dem_diff(masked)
            elevation_stats['site_id'] = site_id
            elevation_stats['buffer_size'] = buffer_size

            elevation_stats_list.append(elevation_stats)
        
        if write_raster:
            # Get the row/col bounds
            row_start, col_start = window.row_off, window.col_off
            row_end = row_start + window.height
            col_end = col_start + window.width
            
            # Place the differenced DEM in the full raster
            # We need to handle potential size mismatches
            diff_height, diff_width = dem_diff.shape[1], dem_diff.shape[2]
            valid_height = min(diff_height, window.height)
            valid_width = min(diff_width, window.width)
            
            # Copy the differenced DEM into the full raster
            full_diff_dem[:, row_start:row_start+valid_height, 
                            col_start:col_start+valid_width] = dem_diff[:, :valid_height, :valid_width]

        
    metrics = pd.DataFrame(elevation_stats_list)

    if write_raster:
        out_path = f'./out_data/well_dems/all_well_DEM_diffs_buffered{buffer_size}_RTK.tif'
        with rio.open(out_path, 'w', **out_meta) as dst:
            dst.write(full_diff_dem)
        return metrics
    else:
        return metrics
    
# %% Run the function

metrics100 = make_local_well_dem(well_pts_repoj, buffer_size=100, write_raster=True)
metrics25 = make_local_well_dem(well_pts_repoj, buffer_size=25, write_raster=True)
metrics15 = make_local_well_dem(well_pts_repoj, buffer_size=15, write_raster=False)
metrics10 = make_local_well_dem(well_pts_repoj, buffer_size=10, write_raster=False)
metrics5 = make_local_well_dem(well_pts_repoj, buffer_size=5, write_raster=False)
metrics2 = make_local_well_dem(well_pts_repoj, buffer_size=2, write_raster=False)
metrics1_5 = make_local_well_dem(well_pts_repoj, buffer_size=1.5, write_raster=False)
# %% Write the metrics 

out_data = pd.concat([metrics100, metrics25, metrics15, metrics10, metrics5, metrics2, metrics1_5])
out_data.to_csv('./out_data/well_dem_metrics_wells_RTK.csv', index=False)

# %%
