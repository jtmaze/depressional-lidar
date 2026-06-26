# %% 1.0 Libraries and file paths

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

data_dir = 'D:/depressional_lidar/data/osbs/'
conditioning_pts_path = f'{data_dir}/in_data/osbs_kriging_conditioning_pts.shp'
neon_lakes_path = f'{data_dir}/in_data/neon_wse_data/neon_wse_data.csv'
well_data_path = f'{data_dir}/in_data/stage_data/osbs_daily_well_depth_Fall2025.csv'

# %% 2.0 Reformat conditioning points to match well data

conditioning_pts = gpd.read_file(conditioning_pts_path)
conditioning_pts.drop(columns=['notes', 'geometry', 'id'], inplace=True)

lidar_times = {
    'oct2018': '2018-09-28', # 2018-09-15, 2018-10-05, 2018-10-12                Teledyne Optech Gemini 12SEN311
    'apr2019': '2019-04-18', # 2019-04-15, 2019-04-16, 2019-04-21, 2019-04-22    Teledyne Optech Gemini 12SEN311
    'sep2021': '2021-09-13', # 2021-08-31, 2021-09-04, 2021-09-13, 2021-09-27 	Teledyne Optech Galaxy Prime 5060445
    'apr2023': '2023-04-28', # 2023-04-23, 2023-05-01, 2023-05-03                Teledyne Optech Galaxy Prime 5060445
    'may2025': '2025-05-10'  # 2025-05-05, 2025-05-15                            Riegl LMS-Q780 2220855
}

ts = conditioning_pts.melt(
    id_vars=['wetland_id', 'bad_dates'],
    value_vars=['oct2018', 'apr2019', 'sep2021', 'apr2023', 'may2025'],
    var_name='lidar_flight',
    value_name='wse_m'
)
ts['bad_dates_list'] = (
    ts['bad_dates']
    .fillna('')
    .str.split(r',\s*')
)

ts['date'] = pd.to_datetime(ts['lidar_flight'].map(lidar_times))

ts['flag'] = ts.apply(
    lambda r: int(r['lidar_flight'] in r['bad_dates_list']),
    axis=1
)

ts = (
    ts.drop(columns=['bad_dates_list', 'bad_dates'])
      .sort_values(['wetland_id', 'date'])
      .reset_index(drop=True)
)


# %% 3.0 Compare the LiDAR flights to Barco and Sugg timeseries

# neon_daily = (
#     pd.read_csv(neon_lakes_path)
#     .query("wetland_id in ['lake_BARC_140','lake_SUGG_140']")
#     .assign(
#         timestamp=lambda d: pd.to_datetime(d['timestamp'], utc=True),
#         date=lambda d: d['timestamp'].dt.tz_convert(None).dt.normalize()
#     )
#     .groupby(['date', 'wetland_id'], as_index=False)['wse_m'].mean()
# )

# pairs = [
#     ('lake_barco_pt1', 'lake_BARC_140', 'BARCO'),
#     ('lake_suggs_pt1', 'lake_SUGG_140', 'SUGGS')
# ]

# plot_df = pd.concat([
#     ts.loc[ts['wetland_id'].eq(pt), ['date', 'wse_m', 'flag']]
#       .merge(
#           neon_daily.loc[neon_daily['wetland_id'].eq(lake), ['date', 'wse_m']],
#           on='date', suffixes=('_lidar', '_neon')
#       )
#       .assign(pair=label, diff_m=lambda d: d['wse_m_lidar'] - d['wse_m_neon'])
#       [['pair', 'diff_m', 'flag']]
#     for pt, lake, label in pairs
# ], ignore_index=True)

# fig, ax = plt.subplots(figsize=(6.2, 3.8))

# for flag, color, name in [(0, 'royalblue', 'Good (>2021, flag=0)'), (1, 'crimson', 'Flagged (<2021, flag=1)')]:
#     d = plot_df[plot_df['flag'].eq(flag)]
#     ax.scatter(d['pair'], d['diff_m'], s=85, color=color, edgecolors='black', linewidths=0.5, label=name)

# mean_qf = plot_df[plot_df['flag'].eq(0)].groupby('pair')['diff_m'].mean()
# ax.scatter(
#     mean_qf.index, mean_qf.values,
#     marker='D', s=115, color='black', edgecolors='black',
#     label='Mean (quality filtered > 2021)'
# )

# ax.axhline(0, color='gray', linestyle='--', linewidth=1)
# ax.set_ylabel('LiDAR - NEON WSE (m)')
# ax.set_title('LiDAR vs NEON Differences')
# ax.grid(True, axis='y', alpha=0.25)
# ax.margins(x=0.08)

# ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, borderaxespad=0.0)
# fig.subplots_adjust(right=0.76)
# plt.show()

# %% 4.0 Read the wetland well data

well_data = pd.read_csv(well_data_path)
well_data = well_data.drop(columns=['well_depth_m'])

ts['indexed_well_depth_m'] = ts['wse_m']
ts = ts.drop(columns=['lidar_flight', 'wse_m'])

# %% 5.0 Write the outdata

out_dir = f'{data_dir}/temp/'

out_data_long = pd.concat([well_data, ts])
out_data_long.to_csv(f'{out_dir}/long_osbs_data_for_Haki_interp.csv')

out_data_long['date'] = pd.to_datetime(out_data_long['date'])

out_data_wide = (
    out_data_long
    .drop(columns='flag')
    .pivot_table(
        index='date',
        columns='wetland_id',
        values='indexed_well_depth_m',
        aggfunc='first'
    )
    .sort_index()
)

out_data_wide.to_csv(f'{out_dir}/wide_osbs_data_for_Haki_interp.csv')
                                   
# %%
