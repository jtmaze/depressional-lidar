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
tgt_ids = ['lake_BARC_130', 'lake_SUGG_140'] # 'lake_SUGG_130', 'lake_BARC_140',

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
well_ts.drop(columns=['well_depth_m'], inplace=True)
well_ts['date'] = pd.to_datetime(well_ts['date'])
well_ts = well_ts[well_ts['date'] <= pd.Timestamp('2021-01-01')]
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

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pair['well_depth_m'], pair['wse_m'], s=18, alpha=0.55, color='steelblue', edgecolors='none')
        ax.set_title(f'{wetland_id} vs {lake_id} (r={pearson_r:.2f}, n={len(pair)})')
        ax.set_xlabel('Wetland well depth (m)')
        ax.set_ylabel('NEON lake WSE (m)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

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

# %% 5.0 Plot the distribution of daily lake water levels after 2021 

post_2021 = neon_daily[neon_daily['date'] >= pd.Timestamp('2021-01-01')].copy()
lake_ids = post_2021['wetland_id'].dropna().unique()

fig, axes = plt.subplots(
    nrows=len(lake_ids),
    ncols=1,
    figsize=(6, 6),
    sharex=False,
)

if len(lake_ids) == 1:
    axes = [axes]

for ax, lake_id in zip(axes, lake_ids):
    vals = post_2021.loc[post_2021['wetland_id'] == lake_id, 'wse_m'].dropna()
    ax.hist(vals, bins=30, color='steelblue', alpha=0.85, edgecolor='white')
    ax.set_title(f'{lake_id} (n={len(vals)})')
    ax.set_xlabel('Daily mean water surface elevation (m)')
    ax.set_ylabel('Count')
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle('Distribution of NEON daily lake water levels after 2021', y=0.995)
plt.tight_layout()
plt.show()

# %%
