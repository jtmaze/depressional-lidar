# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lai_wy_scripts.dmc_vis_functions import plot_stage_ts, plot_ts

from wetland_dem_models.basin_attributes import WetlandBasin

# Paths to datasets
stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv"
climate_path = "D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/ERA-5_daily_mean.csv"
source_dem_path = f'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
footprints_path = f'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
well_point = gpd.read_file(well_points_path)
well_point.rename(
        columns={
            'rtk_elevat': 'rtk_elevation'
        },
        inplace=True
    )
well_point = well_point[
    (well_point['type'] == 'core_well') | (well_point['type'] == 'wetland_well')
]
footprints = gpd.read_file(footprints_path)
stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['day'])
climate_ts = pd.read_csv(climate_path)[['date_local', 'pet_m', 'precip_m']]
climate_ts.rename(columns={'date_local': 'day'}, inplace=True)
# PET is negative in the data, changed to positive. 
climate_ts['pet_m'] = climate_ts['pet_m'] * -1

# Table for reference and logged wells
logging_info = pd.DataFrame([
    {'referend_id': '13_410', 'logged_id': '13_267', 'logged_date': '4/15/2023'}, #0
    {'referend_id': '15_409', 'logged_id': '15_268', 'logged_date': '2/15/2024'}, #1
    {'referend_id': '3_34', 'logged_id': '3_173', 'logged_date': '7/15/2023'}, #2
    {'referend_id': '5_546', 'logged_id': '9_77', 'logged_date': '7/15/2023'}, #3
    {'referend_id': '14_612', 'logged_id': '14_500', 'logged_date': '2/15/2024'}, #4 NOTE: Fairly gradual LAI decline
    {'referend_id': '9_332', 'logged_id': '9_439', 'logged_date': '12/15/2023'}, #5 
    {'referend_id': '14_15', 'logged_id': '7_626', 'logged_date': '10/15/2022'}, #6
    {'referend_id': '3_21', 'logged_id': '7_341', 'logged_date': '8/15/2022'}, #7
    {'referend_id': '6_300', 'logged_id': '6_20', 'logged_date': '10/15/2022'}, #8
    {'referend_id': '5_546', 'logged_id': '5_510', 'logged_date': '12/15/2023'}, #9
    {'referend_id': '3_23', 'logged_id': '3_311', 'logged_date': '5/15/2023'}, #10
    {'referend_id': '14_15', 'logged_id': '14_610', 'logged_date': '12/15/2023'}, #11
])

# %%
selected_idx = 0
logged_id = logging_info.iloc[selected_idx]['logged_id']
reference_id = logging_info.iloc[selected_idx]['referend_id']
logged_date = pd.to_datetime(logging_info.iloc[selected_idx]['logged_date'])

logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()

plot_stage_ts(logged_ts, reference_ts, logged_date)

# %% Generate Estimated SW Storage Time Series

logged_basin = WetlandBasin(
    wetland_id=logged_id,
    well_point_info=well_point[well_point['wetland_id'] == logged_id],
    source_dem_path=source_dem_path, 
    footprint=footprints[footprints['wetland_id'] == logged_id],
    transect_buffer=20
)

reference_basin = WetlandBasin(
    wetland_id=reference_id,
    well_point_info=well_point[well_point['wetland_id'] == reference_id],
    source_dem_path=source_dem_path, 
    footprint=footprints[footprints['wetland_id'] == reference_id],
    transect_buffer=20
)

logged_basin.visualize_shape(show_well=True)
logged_well_elevation = logged_basin.well_point.elevation_dem
print(f'Logged well elevation: {logged_well_elevation:.2f} m')
logged_basin.plot_basin_hypsometry(True)
logged_hypsometry = logged_basin.calculate_hypsometry(method='pct_trim')
logged_hypsometry = pd.DataFrame({'area': logged_hypsometry[0], 'elevation': logged_hypsometry[1]})
logged_hypsometry['well_depth'] = logged_hypsometry['elevation'] - logged_well_elevation
logged_hypsometry['wetland_depth'] = logged_hypsometry['elevation'] - logged_hypsometry['elevation'].min() + 0.01
logged_hypsometry['volume_m3'] = 0.5 * (logged_hypsometry['area'].shift(1) + logged_hypsometry['area']) * 0.01 # NOTE: 1 cm increments

reference_basin.visualize_shape(show_well=True)
reference_well_elevation = reference_basin.well_point.elevation_dem
print(f'Reference well elevation: {reference_well_elevation:.2f} m')
reference_basin.plot_basin_hypsometry(plot_points=True)
reference_hypsometry = reference_basin.calculate_hypsometry(method='pct_trim')
reference_hypsometry = pd.DataFrame({'area': reference_hypsometry[0], 'elevation': reference_hypsometry[1]})
reference_hypsometry['well_depth'] = reference_hypsometry['elevation'] - reference_well_elevation
reference_hypsometry['wetland_depth'] = reference_hypsometry['elevation'] - reference_hypsometry['elevation'].min() + 0.01
reference_hypsometry['volume_m3'] = 0.5 * (reference_hypsometry['area'].shift(1) + reference_hypsometry['area']) * 0.01 # NOTE: 1 cm increments

# %% Plot volume vs. depth from logged and reference basins

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(
    logged_hypsometry['well_depth'], 
    logged_hypsometry['volume_m3'].cumsum(), 
    label='Logged Basin', 
    color='tab:orange'
)

ax.plot(
    reference_hypsometry['well_depth'], 
    reference_hypsometry['volume_m3'].cumsum(), 
    label='Reference Basin', 
    color='tab:blue'
)

ax.set_xlabel('Well Depth (m)')
ax.set_ylabel('Volume (m³)')
ax.set_title('Volume vs Well Depth')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Filter flags from stage data
logged_ts = logged_ts[logged_ts['flag'] != 2].copy()
reference_ts = reference_ts[reference_ts['flag'] != 2].copy()

# %% Make a timeseries of wetland volume from stage data

logged_ts['well_depth'] = logged_ts['well_depth'].round(2)
logged_hypsometry['well_depth'] = logged_hypsometry['well_depth'].round(2)
logged_ts = pd.merge(
    logged_ts, 
    logged_hypsometry[['well_depth', 'volume_m3']], 
    how='left'
)
logged_ts['volume_m3'] = np.where(
    logged_ts['well_depth'] <= logged_hypsometry['well_depth'].min(), 
    0, 
    logged_ts['volume_m3']
)
logged_ts['volume_m3'] = np.where(
    logged_ts['well_depth'] >= logged_hypsometry['well_depth'].max(), 
    logged_hypsometry['volume_m3'].max(), 
    logged_ts['volume_m3']
)
plot_ts(logged_ts, 'volume_m3')

reference_ts['well_depth'] = reference_ts['well_depth'].round(2)
reference_hypsometry['well_depth'] = reference_hypsometry['well_depth'].round(2)
reference_ts = pd.merge(
    reference_ts, 
    reference_hypsometry[['well_depth', 'volume_m3']], 
    how='left'
)
reference_ts['volume_m3'] = np.where(
    reference_ts['well_depth'] <= reference_hypsometry['well_depth'].min(), 
    0, 
    reference_ts['volume_m3']
)
reference_ts['volume_m3'] = np.where(
    reference_ts['well_depth'] >= reference_hypsometry['well_depth'].max(), 
    reference_hypsometry['volume_m3'].max(), 
    reference_ts['volume_m3']
)
plot_ts(reference_ts, 'volume_m3')


# %% Join the logged and reference time series
comparison = pd.merge(
    reference_ts[['day', 'volume_m3']],
    logged_ts[['day', 'volume_m3']],
    how='inner',
    on='day',
    suffixes=('_ref', '_log')
)

comparison['cum_volume_ref'] = comparison['volume_m3_ref'].cumsum()
comparison['cum_volume_log'] = comparison['volume_m3_log'].cumsum()
comparison['is_post_logging'] = comparison['day'] >= logged_date

# %% Merge the climate data

start_date = comparison['day'].min()
climate_ts['day'] = pd.to_datetime(climate_ts['day'])
climate_ts = climate_ts[climate_ts['day'] >= start_date].copy()
climate_ts['p_pet'] = climate_ts['precip_m'] - climate_ts['pet_m']
comparison = pd.merge(comparison, climate_ts, how='left', on='day')
comparison['cum_precip'] = comparison['precip_m'].cumsum()

# %%
x_full = comparison['cum_precip'].to_numpy()
pre_logged = comparison[comparison['day'] < logged_date].copy()
x_pre = pre_logged['cum_precip'].to_numpy()
y_pre_log = pre_logged['cum_volume_log'].to_numpy()
y_pre_ref = pre_logged['cum_volume_ref'].to_numpy()
result_pre_log = np.linalg.lstsq(x_pre[:, None], y_pre_log, rcond=None)
result_pre_ref = np.linalg.lstsq(x_pre[:, None], y_pre_ref, rcond=None)
m_pre_log = result_pre_log[0][0]
m_pre_ref = result_pre_ref[0][0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
# Panel 1: Reference Basin
ax1.scatter(
    comparison['cum_precip'], 
    comparison['cum_volume_ref'], 
    label='Reference Basin', 
    color='tab:blue', 
    alpha=0.5,
)
ax1.set_ylabel('Cumulative Volume (m³)', fontsize=12)
ax1.set_title(f'Reference Basin: {reference_id}', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax1.plot(
    x_full, 
    m_pre_ref * x_full, 
    color='tab:blue', 
    linestyle='--', 
    linewidth=2, 
    label='Pre-logging Fit'
)

# Panel 2: Logged Basin
ax2.scatter(
    comparison['cum_precip'], 
    comparison['cum_volume_log'], 
    label='Logged Basin', 
    color='tab:orange', 
    alpha=0.5
)

ax2.plot(
    x_full, 
    m_pre_log * x_full, 
    color='tab:orange', 
    linestyle='--', 
    linewidth=2, 
    label='Pre-logging Fit'
)

ax2.set_xlabel('Cumulative Precipitation (m)', fontsize=12)
ax2.set_ylabel('Cumulative Volume (m³)', fontsize=12)
ax2.set_title(f'Logged Basin: {logged_id}', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add vertical line at logging date to both panels
logging_cum_precip = comparison[comparison['day'] == logged_date]['cum_precip'].values
if len(logging_cum_precip) > 0:
    logging_cum_precip = logging_cum_precip[0]
else:
    logging_cum_precip = comparison[comparison['day'] <= logged_date]['cum_precip'].iloc[-1]

ax1.axvline(logging_cum_precip, color='red', linestyle='--', linewidth=2, 
            label=f'Logging Date ({logged_date.strftime("%Y-%m-%d")})')
ax2.axvline(logging_cum_precip, color='red', linestyle='--', linewidth=2, 
            label=f'Logging Date ({logged_date.strftime("%Y-%m-%d")})')

plt.tight_layout()
plt.show()

# %%

comparison['predicted_log'] = comparison['cum_precip'] * m_pre_log
comparison['predicted_ref'] = comparison['cum_precip'] * m_pre_ref
comparison['residual_log'] = comparison['cum_volume_log'] - comparison['predicted_log']
comparison['residual_ref'] = comparison['cum_volume_ref'] - comparison['predicted_ref']

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(comparison['day'], comparison['residual_ref'], label=f"Reference {reference_id} Residuals")
ax.plot(comparison['day'], comparison['residual_log'], label=f"Logged {logged_id} Residuals")
ax.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Clean date formatting
locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
# Optional: minor ticks for months
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.set_xlabel('Date')
ax.set_ylabel('Residual [m³]')
ax.legend()
ax.grid(True, which='both', linestyle=':', alpha=0.5)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# %%

logged_basin_area = logged_hypsometry['area'].max()
reference_basin_area = reference_hypsometry['area'].max()

comparison['normalized_logged_residual'] = comparison['residual_log'] / logged_basin_area
comparison['normalized_reference_residual'] = comparison['residual_ref'] / reference_basin_area

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(
    comparison['day'], 
    comparison['normalized_reference_residual'], 
    label=f"Reference {reference_id} Normalized Residuals"
)
ax.plot(
    comparison['day'], 
    comparison['normalized_logged_residual'], 
    label=f"Logged {logged_id} Normalized Residuals"
)
ax.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Clean date formatting
locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
# Optional: minor ticks for months
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.set_xlabel('Date')
ax.set_ylabel('Area Normalized Residual [m]')
ax.legend()
ax.grid(True, which='both', linestyle=':', alpha=0.5)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
# %%
