# %% 1.0 Libraries and file paths

import itertools
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bradford_wy_scripts.functions.wetland_logging_functions import timeseries_qaqc

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"

dataset = 'no_dry_days' # full_obs, no_dry_days
out_path = f"D:/depressional_lidar/data/bradford/out_data/all_wells_correlations_domain_{dataset}.csv"

# %% 2.0 Read the stage data

stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['date'])
stage_data.drop(columns=['date'], inplace=True)
stage_data = stage_data[~stage_data['wetland_id'].isin(['Receiver_wetland', 'Donor_wetland'])]

ids = stage_data['wetland_id'].unique()

pairs = list(itertools.combinations(ids, 2))
print(pairs)

# %% 3.0 Look at global correlations for each possible combination of logged and ref

correlations = []
plot_idx = np.random.choice(len(pairs), size=min(2, len(pairs)), replace=False)

for pair_idx, (w1, w2) in enumerate(pairs):

    if dataset == 'no_dry_days':
        keep_below_obs = False
    elif dataset == 'full_obs':
        keep_below_obs = True
    else:
        raise TypeError("Choose a different dataset")

    ts1 = stage_data[stage_data['wetland_id'] == w1][['well_depth_m', 'day', 'flag']]
    ts1_qaqc = timeseries_qaqc(df=ts1, keep_below_obs=keep_below_obs)
    ts1 = ts1_qaqc['clean_ts']

    ts2 = stage_data[stage_data['wetland_id'] == w2][['well_depth_m', 'day', 'flag']]
    ts2_qaqc = timeseries_qaqc(df=ts2, keep_below_obs=keep_below_obs)
    ts2 = ts2_qaqc['clean_ts']

    comparison = ts1.merge(
        ts2,
        how='inner',
        on='day'
    )
        
    x = comparison['well_depth_m_x']
    y = comparison['well_depth_m_y']

    common_min_depth = max(x.min(), y.min())
    common_max_depth = min(x.max(), y.max())

    correlation = x.corr(y)
    r_squared = correlation ** 2

    correlations.append({
        'wetland1': w1,
        'wetland2': w2,
        'correlation': correlation,
        'r_squared': r_squared,
        'n_observations': len(comparison),
        'min_depth': common_min_depth,
        'max_depth': common_max_depth
    })
    
    if pair_idx in plot_idx:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, alpha=0.6, s=20)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', alpha=0.8)
        ax.set_xlabel(f'Wetland {w1} (m)')
        ax.set_ylabel(f'Wetland {w2} (m)')
        ax.set_title(f'r = {r_squared:.3f}, n = {len(comparison)}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# %% 4.0 Combine correlations into dataframe

correlation_df = pd.DataFrame(correlations)

correlation_df['depth_range'] = correlation_df['max_depth'] - correlation_df['min_depth']

correlation_df.to_csv(out_path, index=False)

# %% 5.0 Biplot for depth range vs correlation

fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(correlation_df['depth_range'], 
                    correlation_df['r_squared'], 
                    c=correlation_df['n_observations'], 
                    s=60, 
                    alpha=0.7, 
                    cmap='viridis',
                    edgecolors='black', 
                    linewidth=0.5)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Observations', rotation=270, labelpad=20)

z = np.polyfit(correlation_df['depth_range'], correlation_df['r_squared'], 1)
p = np.poly1d(z)
ax.plot(correlation_df['depth_range'], p(correlation_df['depth_range']), 
        "r--", alpha=0.8, linewidth=2, label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')

range_corr, p_value = stats.pearsonr(correlation_df['depth_range'], correlation_df['r_squared'])

if p_value < 0.001:
    sig_text = '***'
elif p_value < 0.01:
    sig_text = '**'
elif p_value < 0.05:
    sig_text = '*'
else:
    sig_text = 'ns'

ax.set_xlabel('Depth Range (m)', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_title(f'Wetland Pair Correlations vs Depth Range\n(r = {range_corr:.3f}, p = {p_value:.3f} {sig_text})', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

print(f"\nDepth Range vs R² Correlation: r = {range_corr:.3f}, p = {p_value:.3f} {sig_text}")
print(f"Statistical significance: {sig_text} ({'Highly significant' if p_value < 0.001 else 'Significant' if p_value < 0.05 else 'Not significant'})")
print(f"Interpretation: {'Positive' if range_corr > 0 else 'Negative'} relationship between depth range and correlation strength")


# %%
