# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib as plt
import geopandas as gpd

data_dir = 'D:/depressional_lidar/data/osbs/'

imputed_wells_path = f'{data_dir}/in_data/stage_data/osbs_dail_well_depth_gapfilled.csv'
lidar_conditioning_path = f'{data_dir}/in_data/osbs_kriging_conditioning_pts.shp'

# %% 2.0 Read data

conditioning_pts = gpd.read_file(lidar_conditioning_path)
conditioning_pts.drop(columns=['notes', 'geometry', 'id'], inplace=True)

lidar_times = {
    'oct2018': '2018-09-28', # 2018-09-15, 2018-10-05, 2018-10-12                Teledyne Optech Gemini 12SEN311
    'apr2019': '2019-04-18', # 2019-04-15, 2019-04-16, 2019-04-21, 2019-04-22    Teledyne Optech Gemini 12SEN311
    'sep2021': '2021-09-13', # 2021-08-31, 2021-09-04, 2021-09-13, 2021-09-27 	Teledyne Optech Galaxy Prime 5060445
    'apr2023': '2023-04-28', # 2023-04-23, 2023-05-01, 2023-05-03                Teledyne Optech Galaxy Prime 5060445
    'may2025': '2025-05-10'  # 2025-05-05, 2025-05-15                            Riegl LMS-Q780 2220855
}

lidar_ts = conditioning_pts.melt(
    id_vars=['wetland_id', 'bad_dates'],
    value_vars=['oct2018', 'apr2019', 'sep2021', 'apr2023', 'may2025'],
    var_name='lidar_flight',
    value_name='wse_m'
)

lidar_ts['bad_dates_list'] = (
    lidar_ts['bad_dates']
    .fillna('')
    .str.split(r',\s*')
)

lidar_ts['date'] = pd.to_datetime(lidar_ts['lidar_flight'].map(lidar_times))

lidar_ts['flag'] = lidar_ts.apply(
    lambda r: int(r['lidar_flight'] in r['bad_dates_list']),
    axis=1
)

lidar_ts = (
    lidar_ts.drop(columns=['bad_dates_list', 'bad_dates'])
      .sort_values(['wetland_id', 'date'])
      .reset_index(drop=True)
)

imputed_wells = pd.read_csv(imputed_wells_path)

imputed_wells['date'] = pd.to_datetime(imputed_wells['date'])

avg_imputed_wells = (
    imputed_wells
    .groupby('date', as_index=False)
    .agg(avg_wetland_ts=('well_depth_m', 'mean'))
)

# %% 3.0 Find correlation and slope between LiDAR on lakes and wetland wells average timeseries

unique_conditioning_pts = lidar_ts['wetland_id'].unique()

results = []
for i in unique_conditioning_pts:

    temp = lidar_ts[lidar_ts['wetland_id'] == i]
    temp = temp[['wse_m', 'date']]

    temp = temp.merge(
        avg_imputed_wells,
        how='left',
        on='date'
    )

    print(temp)


# %%
