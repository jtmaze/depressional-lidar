# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio

data_dir = 'D:/depressional_lidar/data/'

source_dem_path = f'{data_dir}/bradford/in_data/bradford_DEM_cleaned_USGS.tif'
well_points_path = f'{data_dir}/rtk_pts_with_dem_elevations.shp'
cores_points_path = f'{data_dir}/bradford/in_data/ancillary_data/bradford_soil_samples_locations.shp'
well_summary_path = f'{data_dir}/bradford/out_data/bradford_wetland_well_summary.csv'

# %% 2.0 Read and format the data

core_points = gpd.read_file(cores_points_path)
well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'geometry', 'site', 'z_dem']]
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)
well_points = well_points[~well_points['wetland_id'].isin(['Donnor_wetland', 'Reciever_wetland'])]

well_summary = pd.read_csv(well_summary_path)
well_points = well_points.merge(
    well_summary,
    on='wetland_id',
    how='left'
)

# %% 3.0 Define functions to estimate core wtd depth

def find_wells_in_dist(
    core_pt: gpd.GeoDataFrame,
    well_gdf: gpd.GeoDataFrame,
    buffer_radius: float
):
    
    area = core_pt.copy()
    area = area.geometry.buffer(buffer_radius)
    
    selected_wells = well_gdf[well_gdf.geometry.within(area)].copy()

    if len(selected_wells) > 0:
        core_geom = core_pt.geometry
        dists = selected_wells.geometry.distance(core_geom)
        mean_dist = dists.mean()
    else:
        mean_dist = np.nan

    return selected_wells, mean_dist

def est_core_pt_z(src, geom, window_size):
    try:
        x, y = geom.x, geom.y
        row, col = src.index(x, y)
        # Define window around the point
        half_window = window_size // 2
        window = rio.windows.Window(
            col - half_window, 
            row - half_window, 
            window_size, 
            window_size
        )
        data = src.read(1, window=window)
        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        
        return np.nanmedian(data)
    
    except:
        print(f"Error sampling point {geom}")
        return np.nan
            

# %% 4.0 Estimate each core's wtd depth

results = []

dist_thresholds = [1_000, 250]

with rio.open(source_dem_path) as dem_src:
    for idx, row in core_points.iterrows():
        if row.geometry is None:
            r = {
                # Core info
                'core_id': row['sample_id'],
                'type': row['type'], 
                'geom': "MISSING COORDS",
                'core_z': np.nan, 
                'core_est_med_wtd': np.nan, 
                'core_est_p75_wtd': np.nan,
                # Well info
                'dist_threshold': np.nan,
                'n_wells': np.nan,
                'avg_well_dist': np.nan,
                'mean_well_z': np.nan,
                'wells_core_z_diff': np.nan,
                'wells_median_wtd': np.nan,
                'wells_p75_wtd': np.nan, 
            }
            results.append(r)
            continue
        geom = row.geometry
        for d in dist_thresholds:
            wells_within, mean_dist = find_wells_in_dist(row, well_points, buffer_radius=d)
            n_wells = len(wells_within)
            if n_wells == 0:
                r = {
                    # Core info
                    'core_id': row['sample_id'],
                    'type': row['type'], 
                    'geom': row.geometry,
                    'core_z': np.nan, 
                    'core_est_med_wtd': np.nan, 
                    'core_est_p75_wtd': np.nan,
                    # Well info
                    'dist_threshold': d,
                    'n_wells': n_wells,
                    'avg_well_dist': np.nan,
                    'mean_well_z': np.nan,
                    'wells_core_z_diff': np.nan,
                    'wells_median_wtd': np.nan,
                    'wells_p75_wtd': np.nan, 
                }
                results.append(r)
                continue

            core_dem_z = est_core_pt_z(dem_src, geom, window_size=4)
            mean_well_z = np.nanmean(wells_within['z_dem'])
            well_core_z_diff = core_dem_z - mean_well_z

            wells_wtd = np.nanmean(wells_within['median'])
            wells_p75_wtd = np.nanmean(wells_within['p75'])

            core_est_med_wtd = wells_wtd - well_core_z_diff
            core_est_p75_wtd = wells_p75_wtd - well_core_z_diff

            r = {
                # Core info
                'core_id': row['sample_id'],
                'type': row['type'], 
                'geom': row.geometry,
                'core_z': core_dem_z, 
                'core_est_med_wtd': core_est_med_wtd, 
                'core_est_p75_wtd': core_est_p75_wtd,
                # Well info
                'dist_threshold': d,
                'n_wells': n_wells,
                'avg_well_dist': mean_dist,
                'mean_well_z': mean_well_z,
                'wells_core_z_diff': well_core_z_diff,
                'wells_median_wtd': wells_wtd,
                'wells_p75_wtd': wells_p75_wtd, 
            }
            
            results.append(r)


# %% 4.0 Concatonate results and write the file

out_df = pd.DataFrame(results)

# Transform geometry to EPSG:4326 before extracting coordinates
# Handle rows with valid geometries
valid_geom_mask = out_df['geom'].apply(lambda x: hasattr(x, 'x') and hasattr(x, 'y'))
out_df['lon_epsg4326'] = "MISSING COORDS"
out_df['lat_epsg4326'] = "MISSING COORDS"

if valid_geom_mask.any():
    valid_df = out_df[valid_geom_mask].copy()
    out_gdf = gpd.GeoDataFrame(valid_df, geometry='geom', crs=core_points.crs)
    out_gdf = out_gdf.to_crs('EPSG:4326')
    out_df.loc[valid_geom_mask, 'lon_epsg4326'] = out_gdf.geometry.x.values
    out_df.loc[valid_geom_mask, 'lat_epsg4326'] = out_gdf.geometry.y.values

# %% 5.0 Write the file
out_df.to_csv(f'{data_dir}/bradford/out_data/est_wtd_depth_at_cores.csv', index=False)


# %%

valid = out_df[~out_df['mean_well_z'].isna()]
print(len(valid['core_id'].unique()))

valid_upland = valid[valid['type'] == 'Azade_soil_core']
print(len(valid_upland['core_id'].unique()))

# %%
