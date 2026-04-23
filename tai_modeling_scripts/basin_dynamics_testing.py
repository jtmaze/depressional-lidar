# %% 
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wetland_utilities.basin_attributes import WetlandBasin
from wetland_utilities.basin_dynamics import WellStageTimeseries, BasinDynamics

site = 'bradford'
tgt_wetland_id = "6_300"
buffer_dist = 150


if site == 'bradford':
    well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
    source_dem = f'D:/depressional_lidar/data/{site}/in_data/{site}_DEM_cleaned_USGS.tif'
    well_stage_path = f'D:/depressional_lidar/data/{site}/in_data/stage_data/{site}_daily_well_depth_Winter2025.csv'
elif site == 'osbs': 
    well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
    source_dem = f'D:/depressional_lidar/data/{site}/in_data/{site}_DEM_cleaned_neon_sep2016.tif'
    well_stage_path = f'D:depressional_lidar/data/{site}/in_data/stage_data/{site}_daily_well_depth_Fall2025.csv'
elif site == 'delmarva':
    source_dem = f'D:/depressional_lidar/data/{site}/in_data/2007_1m_DEM_modified.tif'
    well_stage_path = f'D:/depressional_lidar/data/{site}/in_data/waterlevel_data/daily_well_depth_Fall2025.csv'
    well_points_path = f'D:/depressional_lidar/data/{site}/{site}_well_points.shp'

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'geometry', 'rtk_z']]
    .query("type in ['main_doe_well', 'aux_wetland_well', 'SW', 'CH', 'UW']")
)

well_point = well_points[well_points['wetland_id'] == tgt_wetland_id]
print(well_point.crs)

# %%

basin = WetlandBasin(
    wetland_id=tgt_wetland_id,
    source_dem_path=source_dem,
    footprint=None,
    well_point_info=well_point,
    transect_method=None,
    transect_n=None,
    transect_buffer=buffer_dist
)

basin.plot_basin_hypsometry(plot_points=True)
basin.visualize_shape()

timeseries = WellStageTimeseries.from_csv(
    well_stage_path, 
    well_id=tgt_wetland_id,
    basin=basin,
    date_column='day',
    water_level_column='well_depth_m', 
    well_id_column='wetland_id',
    crop_dates=('2022-03-20', '2026-03-20')
)

timeseries.plot()


dynamics = BasinDynamics(
    basin=basin,
    well_stage=timeseries,
    well_to_dem_offset=0
)

# dynamics.plot_inundated_area_timeseries()
# # Curve params
# ch4_params = {'y_0': -0.01, 'y_f': 0.035, 'k': 2, 'x_mid': -0.2}
# co2_params = {'y_0': 15, 'y_f': -10, 'k': 3, 'x_mid': -0.25}

# dynamics.map_depth_stacks()
# dynamics.map_inundation_stacks()
dynamics.map_tai_stacks(min_depth=-0.2, max_depth=0.2)

# dynamics.plot_sigmoid_curve(ch4_params, depth_range=(-2, 2))
# dynamics.plot_ch4_timeseries(sigmoid_params=ch4_params)
# dynamics.map_ch4_stacks(sigmoid_params=ch4_params)


# dynamics.plot_sigmoid_curve(co2_params, depth_range=(-2, 2))
# dynamics.plot_co2_timeseries(sigmoid_params=co2_params)
# dynamics.map_co2_stacks(sigmoid_params=co2_params)




# %%
