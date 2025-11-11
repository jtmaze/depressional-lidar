# %% 1.0 Libraries, function imports and file paths.
import pprint as pp
import pandas as pd
import geopandas as gpd
import numpy as np 
import matplotlib.pyplot as plt

from lai_wy_scripts.dmc_vis_functions import (
    plot_ts, plot_stage_ts, remove_flagged_buffer, plot_correlations, fit_interaction_model, plot_correlations_from_model,
    plot_dmc, plot_dmc_residuals, residual_change_vs_depth, sample_reference_ts, generate_model_distributions,
    plot_hypothetical_distributions, summarize_depth_shift, summarize_inundation_shift
)

from lai_wy_scripts.lai_vis_functions import read_concatonate_lai, visualize_lai, lai_comparison_vis

from wetland_dem_models.basin_attributes import WetlandBasin

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Spring2025.csv"
source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'
lai_dir = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/well_buffer_250m_includes_wetlands/'


wetland_pairs_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs.csv'
wetland_pairs = pd.read_csv(wetland_pairs_path)

# Strong idxs 93, 124, 81, 42, 16
# Weak idxs 100, 51, 169
pair = wetland_pairs.iloc[106]
reference_id = pair['reference_id']
logged_id = pair['logged_id']
logged_date = pd.to_datetime(pair['logging_date'])


# %% 1.1 Select reference/logging wetlands & read the stage and LAI data

stage_data = pd.read_csv(stage_path)
stage_data['well_id'] = stage_data['well_id'].str.replace('/', '.')
stage_data['day'] = pd.to_datetime(stage_data['day'])
logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()
plot_stage_ts(logged_ts, reference_ts, logged_date, 'Well Depth')

logged_lai = read_concatonate_lai(lai_dir, well_id=logged_id, lai_method='test', upper_bound=5.5, lower_bound=0.5)
reference_lai = read_concatonate_lai(lai_dir, well_id=reference_id, lai_method='test', upper_bound=5.5, lower_bound=0.5)
lai_comparison_vis(
    logged_lai_df=logged_lai[logged_lai['date'] >= pd.to_datetime('2019-01-01')], 
    reference_lai_df=reference_lai[reference_lai['date'] >= pd.to_datetime('2019-01-01')], 
    logged_id=logged_id, 
    reference_id=reference_id
)

# %% 1.2 Read the spatial data and set up the wetland basin classes

footprints = gpd.read_file(footprints_path)
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
    .rename(columns={'rtk_elevat': 'rtk_elevation'})
    .query("type in ['core_well', 'wetland_well']")
)

buffer = 100 # NOTE: Subject to change
# Set up the basin classes
ref_basin = WetlandBasin(
    wetland_id=reference_id, 
    well_point_info=well_point[well_point['wetland_id'] == reference_id],
    source_dem_path=source_dem_path, 
    footprint=None,
    transect_buffer=buffer
)
print("Reference DEM")
ref_basin.visualize_shape(show_well=True, show_deepest=True)
log_basin = WetlandBasin(
    wetland_id=logged_id,
    well_point_info=well_point[well_point['wetland_id'] == logged_id],
    source_dem_path=source_dem_path, 
    footprint=None,
    transect_buffer=buffer
)
print("Logged DEM")
log_basin.visualize_shape(show_well=True, show_deepest=True)

# %% 2.0 Find correlations between well depths

reference_ts_flag_out = remove_flagged_buffer(reference_ts, buffer_days=1)
logged_ts_flag_out = remove_flagged_buffer(logged_ts, buffer_days=1)

comparison_flag_out = pd.merge(
    reference_ts_flag_out, 
    logged_ts_flag_out, 
    how='inner', 
    on='day', 
    suffixes=('_ref', '_log')
).drop(columns=['flag_ref', 'flag_log'])

plot_correlations(
    comparison_flag_out, 
    x_series_name='well_depth_ref',
    y_series_name='well_depth_log',
    log_date=logged_date
)

r, m = fit_interaction_model(
    comparison_flag_out,
    x_series_name='well_depth_ref',
    y_series_name='well_depth_log',
    log_date=logged_date,
    cov_type="HC3"
)

plot_correlations_from_model(
    comparison_flag_out, 
    x_series_name='well_depth_ref', 
    y_series_name='well_depth_log',
    log_date=logged_date,
    model_results=r
)

ref_sample = sample_reference_ts(df=comparison_flag_out, only_pre_log=False, column_name="well_depth_ref", n=10_000)
modeled_distributions = generate_model_distributions(f_dist=ref_sample, models=r)
plot_hypothetical_distributions(model_distributions=modeled_distributions, f_dist=ref_sample, bins=50)

pp.pp(summarize_inundation_shift(modeled_distributions, z_thresh=0))
pp.pp(summarize_depth_shift(modeled_distributions))


# %% 2.1 Adjust well depth timeseries to wetland depth timeseries

logged_well_diff = log_basin.well_point.elevation_dem - log_basin.deepest_point.elevation
logged_ts['wetland_depth'] = logged_ts['well_depth'] + logged_well_diff
ref_well_diff = ref_basin.well_point.elevation_dem - ref_basin.deepest_point.elevation
reference_ts['wetland_depth'] = reference_ts['well_depth'] + ref_well_diff

# Changing all dry wetland days to zero
logged_ts['wetland_depth'] = logged_ts['wetland_depth'].clip(lower=0)
reference_ts['wetland_depth'] = reference_ts['wetland_depth'].clip(lower=0)


# %% 2.1 Correlation plot

comparison = pd.merge(
    reference_ts, 
    logged_ts, 
    how='inner', 
    on='day', 
    suffixes=('_ref', '_log')
).drop(columns=['flag_ref', 'flag_log'])


plot_correlations(
    comparison,
    x_series_name='wetland_depth_ref',
    y_series_name='wetland_depth_log',
    log_date=logged_date,
    filter_obs=None
)

r, m = fit_interaction_model(
    comparison,
    x_series_name='well_depth_ref',
    y_series_name='well_depth_log',
    log_date=logged_date,
    cov_type="HC3"
)

plot_correlations_from_model(
    comparison, 
    x_series_name='wetland_depth_ref', 
    y_series_name='wetland_depth_log',
    log_date=logged_date,
    model_results=r
)

ref_sample = sample_reference_ts(df=comparison, only_pre_log=False, column_name="wetland_depth_ref", n=10_000)
modeled_distributions = generate_model_distributions(f_dist=ref_sample, models=r)
plot_hypothetical_distributions(model_distributions=modeled_distributions, f_dist=ref_sample, bins=50)


# %% 3.0 Plot the wetland stage double mass curves

comparison['cum_wetland_depth_ref'] = comparison['wetland_depth_ref'].cumsum()
comparison['cum_wetland_depth_log'] = comparison['wetland_depth_log'].cumsum()

slope = plot_dmc(
    comparison_df=comparison,
    x_series_name='cum_wetland_depth_ref',
    y_series_name='cum_wetland_depth_log',
    log_date=logged_date
)

residual_df = plot_dmc_residuals(
    comparison_df=comparison,
    x_series_name='cum_wetland_depth_ref',
    y_series_name='cum_wetland_depth_log',
    dmc_slope=slope,
    log_date=logged_date,
    stage=True
)

#residual_change_vs_depth(residual_df, logged_date)



# %%
