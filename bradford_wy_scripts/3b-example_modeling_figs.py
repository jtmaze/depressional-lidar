# %% 1.0 Libraries and file paths
import sys
PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde

from bradford_wy_scripts.functions.wetland_logging_functions import (
    timeseries_qaqc, fit_interaction_model_ols, plot_correlations_from_model
)

from bradford_wy_scripts.functions.lai_vis_functions import read_concatonate_lai

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
pairs_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_150m_all_wells.csv'
distributions_path = 'D:/depressional_lidar/data/bradford//out_data/modeled_logging_stages/hypothetical_distributions_LAI150m_domain_no_dry_days.csv'
spills_path = 'D:/depressional_lidar/data/bradford/out_data/bradford_estimated_basin_spills.csv'
lai_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/basin_buffer_150m_maskedwetland/'

tgt_log = ''
tgt_ref = ''

stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['date'])
stage_data.drop(columns=['date'], inplace=True)

wetland_pairs = pd.read_csv(pairs_path)

spills = pd.read_csv(spills_path)
spills['well_to_spill'] = spills['well_elev'] - spills['max_fill_elev']
log_spill = spills[spills['wetland_id'] == tgt_log].copy()
ref_spill = spills[spills['wetland_id'] == tgt_ref].copy()

log_date = wetland_pairs[
    (wetland_pairs['logged_id'] == tgt_log) & (wetland_pairs['reference_id'] == tgt_ref)
]['planet_logging_date'].iloc[0]

ref_lai = read_concatonate_lai(lai_path, tgt_ref, '150m_well', 5.6, 0)
log_lai = read_concatonate_lai(lai_path, tgt_log, '150m_well', 5.6, 0)

# %% 2.0 Convert depth to spill depth

ref = stage_data[stage_data['wetland_id'] == tgt_ref].copy()
ref_qaqc = timeseries_qaqc(ref, keep_below_obs=False)
ref_ts = ref_qaqc['clean_ts']
ref_ts['depth'] = ref_ts['well_depth_m'] + ref_spill['well_to_spill'].iloc[0] # NOTE: Depth is relative to spill

log = stage_data[stage_data['wetland_id'] == tgt_log].copy()
log_qaqc = timeseries_qaqc(log, keep_below_obs=False)
log_ts = log_qaqc['clean_ts']
log_ts['depth'] = log_ts['well_depth_m'] + log_spill['well_to_spill'].iloc[0] # NOTE: Depth is relative to spill

# %% 3.0 Quick timeseries plot

fig, (ax_depth, ax_lai) = plt.subplots(
    2, 1,
    figsize=(9, 11),
    sharex=True,
    gridspec_kw={'height_ratios': [2, 1]}
)

ref_plot = ref_ts.sort_values('day').dropna(subset=['day', 'depth'])
log_plot = log_ts.sort_values('day').dropna(subset=['day', 'depth'])
log_date_dt = pd.to_datetime(log_date)

log_pre = log_plot[log_plot['day'] < log_date_dt]
log_post = log_plot[log_plot['day'] >= log_date_dt]

# Panel 1: Depth
ax_depth.scatter(ref_plot['day'], ref_plot['depth'], color='blue', s=25, alpha=0.75, label='Reference', zorder=2)
ax_depth.scatter(log_pre['day'], log_pre['depth'], color='#333333', s=25, alpha=0.75, label='Logged Pre', zorder=2)
ax_depth.scatter(log_post['day'], log_post['depth'], color='#E69F00', s=25, alpha=0.75, label='Logged Post', zorder=2)
ax_depth.axvline(log_date_dt, color='red', linestyle='-', linewidth=2.5, label='Planet logging date', zorder=3)
ax_depth.set_ylabel('Depth (m)', fontsize=20, fontweight='bold')
ax_depth.tick_params(axis='both', which='major', labelsize=14)
ax_depth.legend(loc='upper right', fontsize=18, framealpha=1)
ax_depth.set_ylim(-1, 1.2)
ax_depth.grid(alpha=0.2)

# Panel 2: Rolling LAI
ref_lai_plot = ref_lai.copy()
log_lai_plot = log_lai.copy()
ref_lai_plot['date'] = pd.to_datetime(ref_lai_plot['date'])
log_lai_plot['date'] = pd.to_datetime(log_lai_plot['date'])
log_lai_pre = log_lai_plot[log_lai_plot['date'] < log_date_dt]
log_lai_post = log_lai_plot[log_lai_plot['date'] >= log_date_dt]

ax_lai.scatter(
    ref_lai_plot['date'],
    ref_lai_plot['LAI'],
    color='blue',
    s=18,
    alpha=0.5,
    label='Reference LAI (monthly)',
    zorder=1
)
ax_lai.plot(
    ref_lai_plot['date'],
    ref_lai_plot['roll_yr'],
    color='blue',
    linewidth=3,
    label='Reference LAI (12-mo rolling)'
)
ax_lai.scatter(
    log_lai_pre['date'],
    log_lai_pre['LAI'],
    color='#333333',
    s=18,
    alpha=0.5,
    label='Logged LAI pre (monthly)',
    zorder=1
)
ax_lai.scatter(
    log_lai_post['date'],
    log_lai_post['LAI'],
    color='#E69F00',
    s=18,
    alpha=0.5,
    label='Logged LAI post (monthly)',
    zorder=1
)
ax_lai.plot(
    log_lai_pre['date'],
    log_lai_pre['roll_yr'],
    color='#333333',
    linewidth=3,
    label='Logged LAI pre (12-mo rolling)'
)
ax_lai.plot(
    log_lai_post['date'],
    log_lai_post['roll_yr'],
    color='#E69F00',
    linewidth=3,
    label='Logged LAI post (12-mo rolling)'
)
ax_lai.axvline(log_date_dt, color='red', linestyle='-', linewidth=2.5)
ax_lai.set_ylabel('LAI', fontsize=20, fontweight='bold')
#ax_lai.set_xlabel('Date', fontsize=20, fontweight='bold')
ax_lai.tick_params(axis='x', which='major', labelsize=16)
ax_lai.tick_params(axis='y', which='major', labelsize=14)
ax_lai.legend(loc='upper right', fontsize=14, framealpha=1)
ax_lai.grid(alpha=0.2)

ax_lai.xaxis.set_major_locator(mdates.YearLocator())
ax_lai.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_lai.set_xlim(left=pd.to_datetime('2022-01-01'))
ax_lai.set_ylim(1, 4.0)

plt.tight_layout()
plt.show()

# %% 4.0 Run interaction model and generate example visualizations

comparison = pd.merge(
        ref_ts, 
        log_ts, 
        how='inner',
        on='day', 
        suffixes=('_ref', '_log')
).drop(columns=['flag_ref', 'flag_log', 'well_depth_m_log', 'well_depth_m_ref'])

print(comparison)

results = fit_interaction_model_ols(
    comparison,
    x_series_name='depth_ref',
    y_series_name='depth_log',
    log_date=pd.to_datetime(log_date),
    cov_type='HC3'
)

plot_correlations_from_model(
    comparison,
    x_series_name='depth_ref',
    y_series_name='depth_log',
    log_date=pd.to_datetime(log_date),
    model_results=results
)

# %% 5.0 Plot modeled distributions

distributions = pd.read_csv(distributions_path)
distributions = distributions[
    (distributions['log_id'] == tgt_log) & (distributions['ref_id'] == tgt_ref)
].copy()

print(distributions.head(10))

# %% 5.1 Plot pre/post KDE distributions (depth on y-axis, % of days on x-axis)

pre_data = distributions['pre'].dropna().values
pre_data = pre_data + log_spill['well_to_spill'].iloc[0]
post_data = distributions['post'].dropna().values
post_data = post_data + log_spill['well_to_spill'].iloc[0]

depth_grid = np.linspace(
    min(pre_data.min(), post_data.min()),
    max(pre_data.max(), post_data.max()),
    100
)

kde_pre = gaussian_kde(pre_data)
kde_post = gaussian_kde(post_data)


density_pre = kde_pre(depth_grid) #* 100
density_post = kde_post(depth_grid) #* 100

fig, ax = plt.subplots(figsize=(10, 12))

ax.axhline(0, color='red', alpha=1, label='Spill Threshold', linewidth=2.5)

ax.plot(density_pre, depth_grid, color='#333333', linewidth=2, label='Pre-logging')
ax.plot(density_post, depth_grid, color='#E69F00', linewidth=2, label='Post-logging')

# Add means
ax.axhline(pre_data.mean(), color='#333333', linestyle='--', label="Pre Mean Depth", linewidth=2)
ax.axhline(post_data.mean(), color='#E69F00', linestyle='--', label="Post Mean Depth", linewidth=2)

#ax.set_xlabel('Density', fontsize=24, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=24, fontweight='bold')
ax.set_xticks([])
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(frameon=False, fontsize=24)

plt.tight_layout()
plt.show()

# %%
