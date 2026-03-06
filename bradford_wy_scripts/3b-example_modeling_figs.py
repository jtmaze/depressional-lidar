# %% 1.0 Libraries and file paths
import sys
PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from bradford_wy_scripts.functions.wetland_logging_functions import (
    timeseries_qaqc, fit_interaction_model_ols, plot_correlations_from_model
)

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
pairs_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_150m_all_wells.csv'
distributions_path = 'D:/depressional_lidar/data/bradford//out_data/modeled_logging_stages/hypothetical_distributions_LAI150m_domain_no_dry_days.csv'

tgt_log = '9_508'
tgt_ref = '9_609'

stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['date'])
stage_data.drop(columns=['date'], inplace=True)
wetland_pairs = pd.read_csv(pairs_path)

log_date = wetland_pairs[
    (wetland_pairs['logged_id'] == tgt_log) & (wetland_pairs['reference_id'] == tgt_ref)
]['planet_logging_date'].iloc[0]

# %% 2.0 Make scatter plot to illustrate

ref = stage_data[stage_data['wetland_id'] == tgt_ref].copy()
ref_qaqc = timeseries_qaqc(ref, keep_below_obs=False)
ref_ts = ref_qaqc['clean_ts']
ref_ts['depth'] = ref_ts['well_depth_m']

log = stage_data[stage_data['wetland_id'] == tgt_log].copy()
log_qaqc = timeseries_qaqc(log, keep_below_obs=False)
log_ts = log_qaqc['clean_ts']
log_ts['depth'] = log_ts['well_depth_m']

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

# %% 3.0 Plot modeled distributions

distributions = pd.read_csv(distributions_path)
distributions = distributions[
    (distributions['log_id'] == tgt_log) & (distributions['ref_id'] == tgt_ref)
].copy()

print(distributions.head(10))

# %% 3.1 Plot pre/post KDE distributions (depth on y-axis, % of days on x-axis)

pre_data = distributions['pre'].dropna().values
post_data = distributions['post'].dropna().values

depth_grid = np.linspace(
    min(pre_data.min(), post_data.min()),
    max(pre_data.max(), post_data.max()),
    100
)

kde_pre = gaussian_kde(pre_data)
kde_post = gaussian_kde(post_data)


density_pre = kde_pre(depth_grid) #* 100
density_post = kde_post(depth_grid) #* 100

fig, ax = plt.subplots(figsize=(5, 5))

ax.axhspan(0.23, 0.35, color='red', alpha=0.3, label='Spill Threshold')

ax.plot(density_pre, depth_grid, color='#333333', linewidth=2, label='Pre-logging')
ax.plot(density_post, depth_grid, color='#E69F00', linewidth=2, label='Post-logging')

# Add means
ax.axhline(pre_data.mean(), color='#333333', linestyle='--', label="Pre Mean Depth", linewidth=2)
ax.axhline(post_data.mean(), color='#E69F00', linestyle='--', label="Post Mean Depth", linewidth=2)

ax.set_xlabel('Density', fontsize=14, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(frameon=False, fontsize=12)

plt.tight_layout()
plt.show()

# %%
