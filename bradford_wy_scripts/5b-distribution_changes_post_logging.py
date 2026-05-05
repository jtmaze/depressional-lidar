# %% 1.0 Libraries and file paths
"""
NOTE: I need to contemplate a minor issue.
The number of unique wells dramatically decreases at low and high spill depths. 
For example at -1.5m below, the spill depth there might only be a few logged wetlands with data.
How can I ensure the upper and lower bounds of the Q-Q plot aren't warped by data from a single well?
Maybe truncate the curves whenever less than three wells are represented?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

lai_buffer_dist = 150
data_set = 'no_dry_days'

data_dir = "D:/depressional_lidar/data/bradford/"
distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
strong_wetland_pairs_path = f'{data_dir}/out_data/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv'
connectivity_key_path = data_dir + '/bradford_wetland_connect_logging_key.xlsx'
est_spills_path = data_dir + '/out_data/bradford_estimated_basin_spills.csv'

# %% 2.0 Read and munge the data

spills = pd.read_csv(est_spills_path)
distributions = pd.read_csv(distributions_path)
connect = pd.read_excel(connectivity_key_path)

# Only keep strong models
strong_pairs = pd.read_csv(strong_wetland_pairs_path)
distributions = distributions.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id'],
    right_on=['log_id', 'ref_id'],
    how='inner'
)

distributions = distributions.merge(
    connect[['wetland_id', 'connectivity']], 
    left_on='log_id',
    right_on='wetland_id',
    how='left'
)

# %% 2.1 Adjust the distributions so that well data is relative to spill depth

spills = spills[['wetland_id', 'well_elev', 'max_fill_delineated', 'max_fill_elev']]
spills['well_to_spill'] = spills['well_elev'] - spills['max_fill_elev']
spills = spills[['wetland_id', 'well_to_spill']]
print(spills)

distributions = distributions.merge(
    spills,
    left_on='log_id',
    right_on='wetland_id',
    how='left'
)

distributions['pre_adj'] = distributions['pre'] + distributions['well_to_spill']
distributions['post_adj'] = distributions['post'] + distributions['well_to_spill']

distributions_clean = distributions.copy()

# %% 3.0 Simple histogram

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

pre_data = distributions_clean['pre_adj']
post_data = distributions_clean['post_adj']

# Create histograms with normalized counts (% of days)

bins = 50
bin_edges = np.linspace(-1.5, 1, bins + 1)

ax.hist(pre_data, bins=bin_edges, alpha=0.8, color='#333333', 
        edgecolor='black', linewidth=0.5,
        weights=np.ones(len(pre_data)) / len(pre_data) * 100,
        label='Pre-logging')
ax.hist(post_data, bins=bin_edges, alpha=0.8, color='#E69F00',
        edgecolor='black', linewidth=0.5,
        weights=np.ones(len(post_data)) / len(post_data) * 100,
        label='Post-logging')

ax.axvline(pre_data.mean(), color='#333333', linestyle='--', linewidth=2, 
           label=f'Pre mean: {pre_data.mean():.2f}m')
ax.axvline(post_data.mean(), color='#E69F00', linestyle='--', linewidth=2,
           label=f'Post mean: {post_data.mean():.2f}m')

# Formatting
ax.set_title('Aggregate Spill-Adjusted Well Depth Distributions (all ids)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Depth Relative to Spill [m]', fontsize=14)
ax.set_ylabel('% of Days', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.xlim(-1, 1)
plt.show()

# %% 4.0 Q-Q plot for all wells independent of connectivity class. 

dist_all = distributions_clean[
    (distributions_clean['pre_adj'] >= -2.1) & (distributions_clean['pre_adj'] <= 0.75) &
    (distributions_clean['post_adj'] >= -2.1) & (distributions_clean['post_adj'] <= 0.75)
].copy()

pre_all = dist_all['pre_adj'].dropna()
post_all = dist_all['post_adj'].dropna()

n_quantiles = 1000
quantiles_all = np.linspace(0, 1, n_quantiles)
pre_q_all = np.quantile(pre_all, quantiles_all)
post_q_all = np.quantile(post_all, quantiles_all)

fig, ax = plt.subplots(1, 1, figsize=(8, 7))

ax.scatter(pre_q_all, post_q_all, s=30, alpha=0.8, color='black', label='All wells (aggregate)')
ax.plot([pre_q_all.min(), pre_q_all.max()],
        [pre_q_all.min(), pre_q_all.max()],
        linestyle='--', linewidth=2, color='black', label='1:1 line')
ax.axvline(0, color='grey', linestyle='-', linewidth=10, alpha=0.5, label='Spill Depth')

ax.set_xlabel('Pre-Logging [m]', fontsize=15)
ax.set_ylabel('Post-Logging [m]', fontsize=15)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-2, 0.75)
ax.set_ylim(-2, 0.75)
ax.legend(fontsize=12, loc='upper left', markerscale=1.5)

plt.tight_layout()
plt.show()

# Print the quantile difference at -0.5 meters below spill and at the spill (0)
diff_all = post_q_all - pre_q_all

idx_neg05 = np.argmin(np.abs(pre_q_all - (-0.5)))
idx_spill = np.argmin(np.abs(pre_q_all - 0.0))

print(f"Quantile difference at pre = -0.5m: {diff_all[idx_neg05]:.3f} m  "
      f"(pre={pre_q_all[idx_neg05]:.3f}, post={post_q_all[idx_neg05]:.3f})")
print(f"Quantile difference at pre =  0.0m: {diff_all[idx_spill]:.3f} m  "
      f"(pre={pre_q_all[idx_spill]:.3f}, post={post_q_all[idx_spill]:.3f})")


# %% 5.0 Q-Q plot aggregated by connectivity class
connectivity_config = {
    'first order': {'color': 'green', 'label': 'Outlet Connected'},
    'giw': {'color': 'blue', 'label': 'Unconnected'}, 
    'flow-through': {'color': 'red', 'label': 'Flow-through Connected'}
}

fig, ax = plt.subplots(1, 1, figsize=(9, 8))

axis_label_fs = 15
tick_label_fs = 12

dist_filtered = distributions_clean[
    (distributions_clean['pre_adj'] >= -2.1) & (distributions_clean['pre_adj'] <= 0.75) &
    (distributions_clean['post_adj'] >= -2.1) & (distributions_clean['post_adj'] <= 0.75)
].copy()
            
pre_data = dist_filtered['pre_adj'].dropna()
post_data = dist_filtered['post_adj'].dropna()

n_quantiles = 1000
quantiles = np.linspace(0, 1, n_quantiles)

pre_quantiles = np.quantile(pre_data, quantiles)
post_quantiles = np.quantile(post_data, quantiles)

# Plot connectivity-class aggregate quantiles
for conn_class, config in connectivity_config.items():
    class_df = dist_filtered[dist_filtered['connectivity'] == conn_class]
    class_df = class_df.dropna(subset=['pre_adj', 'post_adj'])

    pre_q = np.quantile(class_df['pre_adj'], quantiles)
    post_q = np.quantile(class_df['post_adj'], quantiles)

    diff = post_q - pre_q

    idx_neg05 = np.argmin(np.abs(pre_q - (-0.5)))
    idx_spill = np.argmin(np.abs(pre_q - 0.0))

    print(conn_class)
    print(f"Quantile difference at pre = -0.5m: {diff[idx_neg05]:.3f} m  "
        f"(pre={pre_q[idx_neg05]:.3f}, post={post_q[idx_neg05]:.3f})")
    print(f"Quantile difference at pre =  0.0m: {diff[idx_spill]:.3f} m  "
        f"(pre={pre_q[idx_spill]:.3f}, post={post_q[idx_spill]:.3f})")

    ax.scatter(pre_q, post_q, alpha=0.5, s=10,
                    color=config['color'], label=config['label'])

ax.plot([pre_quantiles.min(), pre_quantiles.max()],
             [pre_quantiles.min(), pre_quantiles.max()],
             'r--', linewidth=2, label='1:1 line', color='black')
ax.set_xlabel('Pre-Logging [m]', fontsize=axis_label_fs)
ax.set_ylabel('Post-Logging [m]', fontsize=axis_label_fs)
ax.axvline(0, color='grey', linestyle='-', linewidth=10, label='Spill Depth', alpha=0.5)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=tick_label_fs)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-2, 0.75)
ax.set_ylim(-2, 0.75)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc='upper left',
    fontsize=12,
    frameon=True,
    markerscale=2.5
)

plt.tight_layout()
plt.show()

# %% 6.0 Q-Q plot for each individual wetland

wetland_filtered = distributions_clean[
    (distributions_clean['pre_adj'] >= -2.1) & (distributions_clean['pre_adj'] <= 0.75) &
    (distributions_clean['post_adj'] >= -2.1) & (distributions_clean['post_adj'] <= 0.75)
].copy()

wetland_ids = sorted(wetland_filtered['log_id'].dropna().unique())

xmin, xmax = -2, 0.75

for wetland_id in wetland_ids:
    wetland_df = wetland_filtered[wetland_filtered['log_id'] == wetland_id]
    wetland_df = wetland_df.dropna(subset=['pre_adj', 'post_adj'])

    if len(wetland_df) < 5:
        continue

    conn_class = wetland_df['connectivity'].iloc[0]
    conn_style = connectivity_config.get(
        conn_class,
        {'color': '#666666', 'label': 'Unknown'}
    )

    q_local = np.linspace(0, 1, 250)
    pre_q = np.quantile(wetland_df['pre_adj'], q_local)
    post_q = np.quantile(wetland_df['post_adj'], q_local)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(pre_q, post_q, s=15, alpha=0.7, color=conn_style['color'])
    ax.plot([xmin, xmax], [xmin, xmax], linestyle='--', linewidth=1.5, color='black', label='1:1 line')
    ax.axvline(0, color='grey', linestyle='-', linewidth=8, alpha=0.35, label='Spill Depth')

    ax.set_title(f'Wetland {wetland_id} | {conn_style["label"]}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pre-Logging [m]', fontsize=12)
    ax.set_ylabel('Post-Logging [m]', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

# %%
