# %% 1.0 Libraries and file paths
import pandas as pd
import geopandas as gpd

buffer_dist = 150
data_dir = 'D:/depressional_lidar/data/'

well_points_path = f'{data_dir}rtk_pts_with_dem_elevations.shp'
out_points_path = f'{data_dir}/bradford/mapping_well_points/well_points.shp'
out_buffered_path = f'{data_dir}/bradford/mapping_well_points/buffered_wells_{buffer_dist}m.shp'
connectivity_key_path = f'{data_dir}/bradford/bradford_wetland_connect_logging_key.xlsx'

# %% 2.0 Read and merge data

well_pts = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)

connect_key = pd.read_excel(connectivity_key_path)

well_pts = well_pts.merge(
    connect_key,
    on='wetland_id',
    how='inner'
)

print(well_pts)

# %% 3.0 Buffer the well points by the buffer dist to generate a new polygon GeoDataFrame

buffered_wells = well_pts.copy()
buffered_wells['geometry'] = well_pts.geometry.buffer(buffer_dist)

# %% 4.0 Write the files

well_pts.to_file(out_points_path)
buffered_wells.to_file(out_buffered_path)

# %%
