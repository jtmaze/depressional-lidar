# %% 1.0 Libraries, function imports and file paths.

import pandas as pd
import geopandas as gpd
import numpy as np 
import matplotlib.pyplot as plt

from lai_wy_scripts.dmc_vis_functions import (plot_ts, plot_stage_ts, 
    remove_flagged_buffer)

from wetland_dem_models.basin_attributes import WetlandBasin

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv"
source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'


# %% 1.1 Select reference/logging wetlands & read the data

logging_info = pd.DataFrame([
    {'referend_id': '13_410', 'logged_id': '13_267', 'logged_date': '4/15/2023'}, #0
    {'referend_id': '15_409', 'logged_id': '15_268', 'logged_date': '2/15/2024'}, #1
    {'referend_id': '3_34', 'logged_id': '3_173', 'logged_date': '7/15/2023'}, #2
    {'referend_id': '14_612', 'logged_id': '14_500', 'logged_date': '2/15/2024'}, # 3 
    {'referend_id': '14_15', 'logged_id': '7_626', 'logged_date': '10/15/2022'}, #4
])

select_idx = 4
logged_id = logging_info.iloc[select_idx]['logged_id']
reference_id = logging_info.iloc[select_idx]['referend_id']
logged_date = pd.to_datetime(logging_info.iloc[select_idx]['logged_date'])

stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['day'])
logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()
plot_stage_ts(logged_ts, reference_ts, logged_date, 'Well Depth')

# %% 1.2 Read the spatial data and set up the wetland basin classes

footprints = gpd.read_file(footprints_path)
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
    .rename(columns={'rtk_elevat': 'rtk_elevation'})
    .query("type in ['core_well', 'wetland_well']")
)

transect_buffer = 20 # NOTE: Subject to change
# Set up the basin classes
ref_basin = WetlandBasin(
    wetland_id=reference_id, 
    well_point_info=well_point[well_point['wetland_id'] == reference_id],
    source_dem_path=source_dem_path, 
    footprint=footprints[footprints['wetland_id'] == reference_id],
    transect_buffer=transect_buffer
)
print("Reference DEM")
ref_basin.visualize_shape(show_well=True, show_deepest=True)
log_basin = WetlandBasin(
    wetland_id=logged_id,
    well_point_info=well_point[well_point['wetland_id'] == logged_id],
    source_dem_path=source_dem_path, 
    footprint=footprints[footprints['wetland_id'] == logged_id],
    transect_buffer=transect_buffer
)
print("Logged DEM")
log_basin.visualize_shape(show_well=True, show_deepest=True)

# %% 2.0 Remove flagged records and adjust well timeseries to wetland bottom

logged_ts = remove_flagged_buffer(logged_ts, buffer_days=3)
reference_ts = remove_flagged_buffer(reference_ts, buffer_days=3)
plot_stage_ts(logged_ts, reference_ts, logged_date, 'Filtered Well Depth')

# %% 2.1 Adjust well timeseries wetland depth timeseries

logged_well_diff = log_basin.well_point.elevation_dem - log_basin.deepest_point.elevation
logged_ts['wetland_depth'] = logged_ts['well_depth'] + logged_well_diff
ref_well_diff = ref_basin.well_point.elevation_dem - ref_basin.deepest_point.elevation
reference_ts['wetland_depth'] = reference_ts['well_depth'] + ref_well_diff

# Changing all dry wetland days to zero
logged_ts['wetland_depth'] = logged_ts['wetland_depth'].clip(lower=0)
reference_ts['wetland_depth'] = reference_ts['wetland_depth'].clip(lower=0)

print('logged')
plot_ts(logged_ts, y_col='wetland_depth')
print('reference')
plot_ts(reference_ts, y_col='wetland_depth')


# %% 2.1 Correlation plot
from lai_wy_scripts.dmc_vis_functions import plot_correlations

comparison = pd.merge(
    reference_ts, 
    logged_ts, 
    how='inner', 
    on='day', 
    suffixes=('_ref', '_log')
).drop(columns=['flag_ref', 'flag_log', 'well_depth_ref', 'well_depth_log'])


plot_stage_ts(
    comparison[['day', 'wetland_depth_log']], 
    comparison[['day', 'wetland_depth_ref']], 
    logged_date,
    y_label="Wetland Depth"
)

comparison['wetland_depth_chg_ref'] = comparison['wetland_depth_ref'].shift(1) - comparison['wetland_depth_ref']
comparison['wetland_depth_chg_log'] = comparison['wetland_depth_log'].shift(1) - comparison['wetland_depth_log']


plot_correlations(
    comparison, 
    x_series_name='wetland_depth_ref', 
    y_series_name='wetland_depth_log',
    log_date=logged_date,
)

plot_correlations(
    comparison,
    x_series_name='wetland_depth_chg_ref',
    y_series_name='wetland_depth_chg_log',
    log_date=logged_date,
)

# %% 3.0 Plot the wetland stage double mass curves

from lai_wy_scripts.dmc_vis_functions import plot_dmc

min_val = 0.01
filtered = comparison[(
        (comparison['wetland_depth_ref'] >= min_val) &
        (comparison['wetland_depth_log'] >= min_val)
)].copy()


comparison['cum_wetland_depth_ref'] = comparison['wetland_depth_ref'].cumsum()
comparison['cum_wetland_depth_log'] = comparison['wetland_depth_log'].cumsum()

plot_dmc(
    comparison_df=comparison,
    x_series_name='cum_wetland_depth_ref',
    y_series_name='cum_wetland_depth_log',
    log_date=logged_date
)

filtered['cum_wetland_depth_ref'] = filtered['wetland_depth_ref'].cumsum()
filtered['cum_wetland_depth_log'] = filtered['wetland_depth_log'].cumsum()

plot_dmc(
    comparison_df=filtered,
    x_series_name='cum_wetland_depth_ref',
    y_series_name='cum_wetland_depth_log',
    log_date=logged_date
)




# %% 4.0 Plot the wetland inundation double mass curves

# %% 5.0 Plot the wetland storage double mass curves. 