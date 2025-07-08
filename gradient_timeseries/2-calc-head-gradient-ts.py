# %% 1.0 Libraries and directories

import os
import pandas as pd
import geopandas as gpd

os.chdir('/Users/jmaze/Documents/projects/depressional_lidar/')
catchment = 'jl'

z_points = gpd.read_file('./delmarva/out_data/well_pts_clean.shp')
elevation_gradients = pd.read_csv(f'./delmarva/out_data/{catchment}_elevation_gradients.csv')
wl_path = './delmarva/waterlevel_data/output_JM_2019_2022.csv'
# Get unique wells from both columns
gradient_wells = list(
    set(
        elevation_gradients['well0'].unique().tolist()
        + elevation_gradients['well1'].unique().tolist()
    )
)

# %% 2.0 Read, filter sites and aggregate to daily, adjust for well elevation

wl = pd.read_csv(wl_path)
wl = wl[wl['Site_Name'].isin(gradient_wells)]
wl['Timestamp'] = pd.to_datetime(wl['Timestamp'])
wl['Date'] = wl['Timestamp'].dt.date

wl_daily = wl.groupby(['Site_Name', 'Date']).agg({
    'waterLevel': 'mean',
    'Flag': 'first',
    'Notes': 'first'
}).reset_index()

wl_daily['Date'] = pd.to_datetime(wl_daily['Date'])

wl_daily = pd.merge(wl_daily, z_points[['Elevation', 'Site_Name']], how='left', on='Site_Name')
wl_daily['Elevation_m'] = wl_daily['Elevation'] / 100
wl_daily['rel_wl'] = wl_daily['waterLevel'] - wl_daily['Elevation_m']

# %% 3.0 Make a gradients dataframe to hold timeseries

if catchment == 'bc':
    start_dt = '2021-03-10'
    end_dt = '2022-10-07'

elif catchment == 'jl':
    start_dt = '2021-03-02'
    end_dt = '2022-10-07'

date_range = pd.date_range(start_dt, end_dt, freq='D')
well_pairs = elevation_gradients['well_pair'].unique()

gradient_ts = pd.DataFrame(
    [(date, pair) for date in date_range for pair in well_pairs],
    columns=['Date', 'well_pair']
)
gradient_ts = pd.merge(gradient_ts, elevation_gradients, how='left', on='well_pair')

# %% 4.0 Populate gradients timeseries with water level gradients

# Merging wl time series for w0
gradient_ts = (
    pd.merge(
        gradient_ts,
        wl_daily[['rel_wl', 'Site_Name', 'Date']],
        how='left',
        left_on=['well0', 'Date'],
        right_on=['Site_Name', 'Date']
    )
    .rename(columns={'rel_wl': 'rel_wl0'})
    .drop(columns='Site_Name')
)

# Merging wl timeseries for w1
gradient_ts = (
    pd.merge(
        gradient_ts,
        wl_daily[['rel_wl', 'Site_Name', 'Date']],
        how='left',
        left_on=['well1', 'Date'],
        right_on=['Site_Name', 'Date']
    )
    .rename(columns={'rel_wl': 'rel_wl1'})
    .drop(columns='Site_Name')
)

gradient_ts['dh'] = gradient_ts['rel_wl0'] - gradient_ts['rel_wl1']
gradient_ts['head_gradient_cm_m'] = gradient_ts['dh'] / gradient_ts['well_dist_m'] * 100
gradient_ts['adj_gradient'] = gradient_ts['head_gradient_cm_m'] / gradient_ts['elevation_gradient_cm_m']

# %% 4.0 Write output to csv

gradient_ts.to_csv(f'./delmarva/out_data/{catchment}_gradient_timeseries.csv', index=False)

# %%
