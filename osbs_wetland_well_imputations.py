# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data_dir = 'D:/depressional_lidar/data/osbs/'
wells_ts_path = f'{data_dir}/in_data/stage_data/osbs_daily_well_depth_Fall2025.csv'
imputed_path = f'{data_dir}/in_data/stage_data/osbs_dail_well_depth_gapfilled.csv'

# %% 2.0 Read and format the well data

wells_ts = pd.read_csv(wells_ts_path)
wells_ts['date'] = pd.to_datetime(wells_ts['date'])
print(len(wells_ts['wetland_id'].unique()))

# %% 3.0 Plot data with flags removed

plot_ts = wells_ts[~wells_ts['flag'].isin([2, 3])]

plot_ts = plot_ts[~((plot_ts['wetland_id'] == 'Hansford') & (plot_ts['date'] <= pd.to_datetime('2022-01-01')))]

wetland_ids = sorted(plot_ts['wetland_id'].unique())
n_wells = len(wetland_ids)


distinct_colors = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#9A6324',
    '#800000', '#aaffc3', '#808000', '#000075', '#a9a9a9',
    '#ff4500', '#1e90ff', '#32cd32', '#ff1493', '#ffd700',
]

# Full date range to break lines at data gaps
full_dates = pd.date_range(
    start=plot_ts['date'].min(),
    end=plot_ts['date'].max(),
    freq='D'
)

fig, ax = plt.subplots(figsize=(15, 8))

for idx, wetland_id in enumerate(wetland_ids):
    well_data = (
        plot_ts[plot_ts['wetland_id'] == wetland_id]
        .sort_values('date')
        .set_index('date')['indexed_well_depth_m']
        .reindex(full_dates)  # gaps become NaN, breaking the line
    )

    ax.plot(
        well_data.index,
        well_data.values,
        color=distinct_colors[idx % len(distinct_colors)],
        linewidth=1.5,
        alpha=0.9,
        label=str(wetland_id)
    )

ax.set_ylabel('Indexed Well Depth (m)', fontsize=16)
ax.set_title(
    f'Timeseries of OSBS Wells',
    fontsize=16,
    fontweight='bold'
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    ncol=1,
    fontsize=16
)

ax.tick_params(axis='both', labelsize=14)
ax.grid(True, alpha=0.25)
fig.tight_layout()
plt.show()

# %% 4.0 Make a daily average of all wells plot timeseries

well_ts_avg = plot_ts.groupby(['date']).agg(
    avg_well_depth=('indexed_well_depth_m', 'mean')
)

well_ts_count = plot_ts.groupby(['date'])['indexed_well_depth_m'].count()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

axes[0].plot(well_ts_avg.index, well_ts_avg['avg_well_depth'], color='black', lw=1.5)
axes[0].set_ylabel('well depth (m)', fontsize=15)
axes[0].set_title('Daily Average Well Depth (OSBS)', fontsize=15)
axes[0].tick_params(axis='both', labelsize=13)
axes[0].grid(alpha=0.3)

axes[1].plot(well_ts_count.index, well_ts_count.values, color='steelblue', lw=1.2)
axes[1].set_ylabel('# wells', fontsize=15)
axes[1].set_xlabel('Date', fontsize=15)
axes[1].tick_params(axis='both', labelsize=13)
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()

# %% 5.0 Calculate each well's r value with well_ts_avg

results = []
for idx, i in enumerate(wetland_ids):
    well_data = (
        plot_ts[plot_ts['wetland_id'] == i]
        .set_index('date')['indexed_well_depth_m']
    )
    merged = well_data.rename('well').to_frame().join(
        well_ts_avg['avg_well_depth'], how='inner'
    ).dropna()

    slope, intercept, r, p, _ = stats.linregress(
        merged['avg_well_depth'], merged['well']
    )
    results.append({'wetland_id': i, 'r': r, 'slope': slope, 'intercept': intercept, 'n': len(merged)})

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(merged['avg_well_depth'], merged['well'],
               s=8, alpha=0.5, color='blue')
    x_line = np.linspace(merged['avg_well_depth'].min(), merged['avg_well_depth'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='k', lw=1.2)
    ax.set_xlabel('Daily mean depth (m)', fontsize=14)
    ax.set_ylabel('Well depth (m)', fontsize=14)
    ax.set_title(f'{i}  |  r={r:.2f}  slope={slope:.2f}', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()

results_df = pd.DataFrame(results).sort_values('r', ascending=False)
print(results_df.to_string(index=False))

# %% 5.1 Print stats on slope and r value

print(f"Slope Mean: {results_df['slope'].mean():.2f}")
print(f"Slope Std Dev: {results_df['slope'].std():.2f}")
print(f"Slope Min: {results_df['slope'].min():.2f}")
print(f"Slope Max: {results_df['slope'].max():.2f}")

# Statistics for R
print(f"R Mean: {results_df['r'].mean():.2f}")
print(f"R Std Dev: {results_df['r'].std():.2f}")
print(f"R Min: {results_df['r'].min():.2f}")

# %% 6.0 Use the average of all wells to imput missing days

imputed_dfs = []
for i in wetland_ids:

    temp = plot_ts[plot_ts['wetland_id'] == i].copy()
    temp = temp[['date', 'indexed_well_depth_m', 'wetland_id']]
    temp.rename(columns={'original_well_depth': 'indexed_well_depth_m'}, inplace=True)

    model = results_df[results_df['wetland_id'] == i].iloc[0]
    b = model['intercept']
    m = model['slope']

    # Join the average well depth to temp
    temp = temp.merge(
        well_ts_avg,
        how='outer',
        on='date'
    )
    temp['wetland_id'] = i # adding wetland_id to new rows from missing dates

    temp['predicted_well_depth_m'] = temp['avg_well_depth'] * m + b

    # Plot each well's observed and imputed timeseries
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temp['date'], temp['predicted_well_depth_m'],
            color='red', lw=1, alpha=0.6, label='Predicted (imputed)')
    ax.plot(temp['date'], temp['indexed_well_depth_m'],
            color='black', lw=1.5, label='Observed')
    ax.set_ylabel('Well Depth (m)', fontsize=15)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_title(f'{i} — Observed vs Imputed Well Depth', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()

    imputed_dfs.append(temp)

# %% 7.0 Concatonate the imputed data and write to a file

imputed = pd.concat(imputed_dfs)
imputed.rename(columns={'predicted_well_depth_m': 'well_depth_m'}, inplace=True)
imputed = imputed[['date', 'wetland_id', 'well_depth_m']]

imputed.to_csv(imputed_path, index=False)

# %%
