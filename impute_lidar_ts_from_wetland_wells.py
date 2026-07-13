# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import stats

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
    temp = temp[['wse_m', 'date', 'flag']]

    temp = temp.merge(
        avg_imputed_wells,
        how='left',
        on='date'
    )

    valid = temp.dropna(subset=['wse_m', 'avg_wetland_ts'])

  
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        valid['avg_wetland_ts'], valid['wse_m']
    )

    results.append({
        'wetland_id': i,
        'slope': slope,
        'intercept': intercept,
        'r': r_value,
        'n': len(valid)
    })

    fig, ax = plt.subplots(figsize=(5, 4))
    unflagged = valid[valid['flag'] == 0]
    flagged = valid[valid['flag'] == 1]
    ax.scatter(unflagged['avg_wetland_ts'], unflagged['wse_m'], s=25, label='valid')
    ax.scatter(flagged['avg_wetland_ts'], flagged['wse_m'], s=25, marker='x',
               color='red', label='flagged')

    x_line = np.linspace(valid['avg_wetland_ts'].min(), valid['avg_wetland_ts'].max(), 50)
    ax.plot(x_line, slope * x_line + intercept, color='black', linewidth=1.5)

    ax.set_title(f'{i}\nr={r_value:.2f}, slope={slope:.2f}')
    ax.set_xlabel('avg wetland well depth (m)')
    ax.set_ylabel('LiDAR WSE (m)')
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results)

# %% 4.0 Boxplots for the slopes and r values

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].boxplot(results_df['slope'])
axes[0].set_ylabel('slope')
axes[0].set_title('Slope Distribution')
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(results_df['r'])
axes[1].set_ylabel('r value')
axes[1].set_title('Correlation Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% 5.0 Impute timeseries for LiDAR points, plot the imputed data with points

imputed_dfs = []
for i in unique_conditioning_pts:

    temp = lidar_ts[lidar_ts['wetland_id'] == i].copy()
    temp = temp[['date', 'wse_m', 'wetland_id', 'flag']]

    model = results_df[results_df['wetland_id'] == i].iloc[0]
    m = model['slope']
    b = model['intercept']

    temp = temp.merge(
        avg_imputed_wells,
        how='outer',
        on='date'
    )
    temp['wetland_id'] = i

    temp['predicted_wse_m'] = temp['avg_wetland_ts'] * m + b

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(temp['date'], temp['predicted_wse_m'],
            color='red', lw=1, alpha=0.6, label='Predicted (imputed)')
    ax.scatter(temp['date'], temp['wse_m'],
               color='black', s=30, label='Observed')
    ax.set_ylabel('WSE (m)')
    ax.set_title(f'{i} — Observed vs Imputed WSE', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()

    imputed_dfs.append(temp)

# %% 6.0 Plot each WSE timeseries on the same plot

fig, ax = plt.subplots(figsize=(14, 6))

for df in imputed_dfs:
    ax.plot(df['date'], df['predicted_wse_m'], color='blue', lw=1, alpha=0.7)

ax.set_ylabel('WSE (m)')
ax.set_xlabel('Date')
ax.set_title('All Imputed WSE Timeseries')
ax.grid(alpha=0.25)
fig.tight_layout()
plt.ylim(22.5, 30.5)
plt.show()

# %%
