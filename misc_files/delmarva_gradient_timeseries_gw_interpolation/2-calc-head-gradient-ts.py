# %% 1.0 Libraries and directories

import os
import pandas as pd


os.chdir('D:/depressional_lidar/')
catchment = 'jl'

elevation_gradients = pd.read_csv(f'./delmarva/out_data/{catchment}_elevation_gradients.csv')

wl_path = './delmarva/waterlevel_data/combined_output.csv'
# Get unique wells from both columns
gradient_wells = list(
    set(
        elevation_gradients['well0'].unique().tolist()
        + elevation_gradients['well1'].unique().tolist()
    )
)

well0_elevations = elevation_gradients[['well0', 'z0']].rename(
    columns={'well0': 'well_id', 'z0': 'elevation'}
).drop_duplicates()

well1_elevations = elevation_gradients[['well1', 'z1']].rename(
    columns={'well1': 'well_id', 'z1': 'elevation'}
).drop_duplicates()

z_points = pd.concat([well0_elevations, well1_elevations]).drop_duplicates()


# %% 2.0 Read, filter sites and aggregate to daily, adjust for well elevation

wl = pd.read_csv(wl_path)
wl = wl[wl['well_id'].isin(gradient_wells)]
wl['Timestamp'] = pd.to_datetime(wl['Timestamp'])
wl['Date'] = wl['Timestamp'].dt.date

wl_daily = wl.groupby(['well_id', 'Date']).agg({
    'water_level': 'mean',
    'Flag': 'first',
}).reset_index()

wl_daily['Date'] = pd.to_datetime(wl_daily['Date'])

wl_daily = pd.merge(wl_daily, z_points[['elevation', 'well_id']], how='left', on='well_id')
wl_daily['elevation_m'] = wl_daily['elevation'] / 100
wl_daily['rel_wl'] = wl_daily['water_level'] + wl_daily['elevation_m']

# %% 3.0 Make a gradients dataframe to hold timeseries

"""
Not bothering with date filtering
"""
start_dt = wl_daily['Date'].min()
end_dt = wl_daily['Date'].max()

print(f"Start date: {start_dt}, End date: {end_dt}")

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
        wl_daily[['rel_wl', 'well_id', 'Date']],
        how='left',
        left_on=['well0', 'Date'],
        right_on=['well_id', 'Date']
    )
    .rename(columns={'rel_wl': 'rel_wl0'})
    .drop(columns='well_id')
)

# Merging wl timeseries for w1
gradient_ts = (
    pd.merge(
        gradient_ts,
        wl_daily[['rel_wl', 'well_id', 'Date']],
        how='left',
        left_on=['well1', 'Date'],
        right_on=['well_id', 'Date']
    )
    .rename(columns={'rel_wl': 'rel_wl1'})
    .drop(columns='well_id')
)

gradient_ts['dh'] = gradient_ts['rel_wl0'] - gradient_ts['rel_wl1']
gradient_ts['head_gradient_cm_m'] = gradient_ts['dh'] / gradient_ts['well_dist_m'] * 100
gradient_ts['adj_gradient'] = gradient_ts['head_gradient_cm_m'] / gradient_ts['elevation_gradient_cm_m']

# %% 5.0 Classify the well pair relationships

def classify_well_pairs(row):
    w0 = row['well0']
    w1 = row['well1']

    def find_well_type(well: str):
        # Use 'in' operator for string pattern matching
        if '-UW' in well:
            return 'UW'
        elif '-CH' in well:
            return 'CH'
        elif '-SW' in well:
            return 'SW'
        else:
            return 'Unknown'
        
    w0_type = find_well_type(w0)
    w1_type = find_well_type(w1)

    return f'{w0_type}__to__{w1_type}'

gradient_ts['pair_type'] = gradient_ts.apply(classify_well_pairs, axis=1)


# %% 6.0 Write output to csv

output_path = f'./delmarva/out_data/{catchment}_gradient_timeseries.csv'
gradient_ts.to_csv(output_path, index=False)

# %%
