# %% 1.0 Libraries and file paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'D:/depressional_lidar/data/osbs/'

neon_ts_path = f'{data_dir}/in_data/neon_wse_data/neon_wse_data.csv'
neon_meta_path = f'{data_dir}/in_data/neon_wse_data/neon_logger_meta.csv'
well_ts_path = f"{data_dir}/in_data/stage_data/osbs_daily_well_depth_Fall2025.csv"

# %% 2.0 Read data

neon_ts = pd.read_csv(neon_ts_path)
tgt_ids = ['lake_BARC_130', 'lake_SUGG_140', 'lake_BARC_140', 'lake_SUGG_130'] # 'lake_SUGG_130', 'lake_BARC_140',

neon_ts = neon_ts[neon_ts['wetland_id'].isin(tgt_ids)]
neon_ts['timestamp'] = pd.to_datetime(neon_ts['timestamp'], utc=True)
neon_ts['date'] = neon_ts['timestamp'].dt.tz_convert(None).dt.normalize()

neon_daily = (
    neon_ts
    .groupby(['date', 'wetland_id'], as_index=False)['wse_m']
    .mean()
)

neon_meta = pd.read_csv(neon_meta_path)

well_ts = pd.read_csv(well_ts_path)
well_ts = well_ts[well_ts['flag'] != 3]
well_ts.drop(columns=['well_depth_m'], inplace=True) #NOTE: Calling 'indexed' well_depth_m the 'well_depth'
well_ts['date'] = pd.to_datetime(well_ts['date'])
#well_ts = well_ts[well_ts['date'] <= pd.Timestamp('2021-01-01')]
well_ts.rename(
    columns={
        'indexed_well_depth_m': 'well_depth_m'
    }, 
    inplace=True
)

# %% 2.0 Quick NEON lake TS

fig, ax = plt.subplots(figsize=(14, 6))

for wetland_id in neon_daily['wetland_id'].unique():
    data = neon_daily[neon_daily['wetland_id'] == wetland_id].sort_values('date')
    ax.plot(data['date'], data['wse_m'], marker='o', markersize=3, label=wetland_id, alpha=0.7)

ax.set_xlabel('Date')
ax.set_ylabel('Water Surface Elevation (m)')
ax.set_title('NEON WSE Daily Mean Timeseries')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% 3.0 Asses correlation between NEON lakes and wetland wells

correlation_results = []

for wetland_id in well_ts['wetland_id'].dropna().unique():
    wetland_data = well_ts[well_ts['wetland_id'] == wetland_id][['date', 'wetland_id', 'well_depth_m']].copy()

    for lake_id in neon_daily['wetland_id'].dropna().unique():
        lake_data = neon_daily[neon_daily['wetland_id'] == lake_id][['date', 'wetland_id', 'wse_m']].copy()

        pair = wetland_data.merge(lake_data, on='date', how='inner', suffixes=('_well', '_lake'))
        pair = pair.dropna(subset=['well_depth_m', 'wse_m'])

        if pair.empty:
            continue

        pearson_r = pair['well_depth_m'].corr(pair['wse_m'], method='pearson')

        correlation_results.append({
            'wetland_id': wetland_id,
            'lake_id': lake_id,
            'n': len(pair),
            'pearson_r': pearson_r,
        })

correlation_df = pd.DataFrame(correlation_results).sort_values(['wetland_id', 'lake_id']).reset_index(drop=True)
print(correlation_df)

# %% 4.0 Boxplot of lake to wetland correlations

plot_groups = {
    'All correlations': correlation_df['pearson_r'].dropna().to_numpy(),
    'lake_BARC_130': correlation_df.loc[correlation_df['lake_id'] == 'lake_BARC_130', 'pearson_r'].dropna().to_numpy(),
    'lake_SUGG_140': correlation_df.loc[correlation_df['lake_id'] == 'lake_SUGG_140', 'pearson_r'].dropna().to_numpy(),
}

fig, ax = plt.subplots(figsize=(6, 6))

labels = list(plot_groups.keys())
values = [plot_groups[label] for label in labels]
positions = np.arange(1, len(labels) + 1)

ax.boxplot(
    values,
    positions=positions,
    widths=0.5,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor='#d9e8f5', edgecolor='black', linewidth=1.2),
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
)

for xpos, label, group_values in zip(positions, labels, values):
    if len(group_values) == 0:
        continue
    jitter = np.random.uniform(-0.12, 0.12, size=len(group_values))
    ax.scatter(
        np.full(len(group_values), xpos) + jitter,
        group_values,
        s=45,
        alpha=0.75,
        color='#2c7fb8',
        edgecolors='white',
        linewidths=0.5,
        zorder=3,
    )

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("Pearson's r")
ax.set_title('Correlation summary across wetland-lake pairings')
ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.6)
ax.grid(True, axis='y', alpha=0.25)
plt.tight_layout()
plt.show()

# %% 5.0 Simple biplot of lakes versus wetland (average all wetland_ids) timeseries

tgt_lake = 'lake_SUGG_140'  # Options: 'lake_BARC_130', 'lake_BARC_140', 'lake_SUGG_130', 'lake_SUGG_140'

lidar_dates = pd.to_datetime([
    '2018-09-28',  # oct2018
    '2019-04-18',  # apr2019
    '2021-09-13',  # sep2021
    '2023-04-28',  # apr2023
    '2025-05-10',  # may2025
])

well_ts_avg = well_ts.groupby(['date']).agg(
    avg_well_depth=('well_depth_m', 'mean')
)

lake_data = neon_daily[neon_daily['wetland_id'] == tgt_lake][['date', 'wse_m']].copy()
biplot_df = well_ts_avg.reset_index().merge(lake_data, on='date', how='inner').dropna(subset=['avg_well_depth', 'wse_m'])

lidar_overlay = pd.merge_asof(
    pd.DataFrame({'date': lidar_dates}).sort_values('date'),
    biplot_df.sort_values('date'),
    on='date',
    direction='nearest',
    tolerance=pd.Timedelta('10 days')
).dropna(subset=['avg_well_depth', 'wse_m'])

pearson_r = biplot_df['avg_well_depth'].corr(biplot_df['wse_m'])
m, b = np.polyfit(biplot_df['avg_well_depth'], biplot_df['wse_m'], 1)
x_line = np.linspace(biplot_df['avg_well_depth'].min(), biplot_df['avg_well_depth'].max(), 100)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(biplot_df['avg_well_depth'], biplot_df['wse_m'], s=40, alpha=0.7, color='#2c7fb8', edgecolors='white', linewidths=0.5, zorder=3)
ax.plot(x_line, m * x_line + b, color='#d62728', linewidth=1.5, label=f'slope={m:.3f}, r={pearson_r:.2f}  (n={len(biplot_df)})')
if not lidar_overlay.empty:
    ax.scatter(lidar_overlay['avg_well_depth'], lidar_overlay['wse_m'], s=120, color='orange', edgecolors='black', linewidths=0.8, zorder=5, label='LiDAR flight dates')
ax.set_xlabel('Avg Well Depth (m)')
ax.set_ylabel(f'{tgt_lake} WSE (m)')
ax.set_title(f'{tgt_lake} WSE vs. Mean Wetland Well Depth')
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# %%
