# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

lai_buffer_dist = 150
data_set = 'no_dry_days'

data_dir = "D:/depressional_lidar/data/bradford/"
distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
strong_wetland_pairs_path = f'{data_dir}/out_data/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv'
connectivity_key_path = data_dir + '/bradford_wetland_connect_logging_key.xlsx'

# %% 2.0 Read and munge the data

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

print(distributions)

#distributions = distributions[distributions['log_id'] != '9_77']
# %% Plot the aggregated (over all pairs) pre and post logging distributions

# distributions['ref_log'] = distributions['ref_id'] + '_' + distributions['log_id']
# distributions_clean = distributions[distributions['ref_log'].isin(pairs['ref_log'])]

distributions_clean = distributions.copy()

# %% 3.0 Simple histogram

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

pre_data = distributions_clean['pre']
post_data = distributions_clean['post']

# Create histograms with normalized counts (% of days)
# Define common bin edges for consistent bin widths
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
ax.set_title('Aggregate Well Depth Distributions (all ids)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Depth [m]', fontsize=14)
ax.set_ylabel('% of Days', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.xlim(-1, 1)
plt.show()

# %% 4.0 Q-Q plot for entire dataset on aggregate. 
# %% 4.0 Q-Q plot for entire dataset on aggregate.

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

dist_filtered = distributions_clean[
    (distributions_clean['pre'] >= -1.5) & (distributions_clean['pre'] <= 1.0) &
    (distributions_clean['post'] >= -1.5) & (distributions_clean['post'] <= 1.0)
].copy()

pre_data = dist_filtered['pre'].dropna()
post_data = dist_filtered['post'].dropna()

n_quantiles = 1000
quantiles = np.linspace(0, 1, n_quantiles)

pre_quantiles = np.quantile(pre_data, quantiles)
post_quantiles = np.quantile(post_data, quantiles)

axes[0].scatter(pre_quantiles, post_quantiles, alpha=0.7, s=30, color='#333333')
axes[0].plot([pre_quantiles.min(), pre_quantiles.max()],
             [pre_quantiles.min(), pre_quantiles.max()],
             'r--', linewidth=2, label='1:1 line')
axes[0].set_xlabel('Pre-Logging [m]', fontsize=12)
axes[0].set_ylabel('Post-Logging [m]', fontsize=12)
axes[0].set_title('Q-Q Plot: Pre vs Post Logging', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].set_aspect('equal', adjustable='box')
axes[0].set_xlim(-1.25, 0.75)
axes[0].set_ylim(-1.25, 0.75)

quantile_diff = post_quantiles - pre_quantiles
axes[1].scatter(pre_quantiles, quantile_diff, alpha=0.7, s=30, color='#333333')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='No change')
axes[1].set_xlabel('Pre-Logging [m]', fontsize=12)
axes[1].set_ylabel('Depth Change (Post - Pre) [m]', fontsize=12)
axes[1].set_title('Quantile-by-Quantile Depth Change', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)
axes[1].set_xlim(-1.25, 0.75)

plt.show()

# %% 5.0 Facet Q-Q plot by connectivity
connectivity_config = {
    'first order': {'color': 'green', 'label': '1st Order Ditched'},
    'giw': {'color': 'blue', 'label': 'GIW'}, 
    'flow-through': {'color': 'red', 'label': 'flow-through'}
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

distributions_clean = distributions_clean[
    (distributions_clean['pre'] >= -1.5) & (distributions_clean['pre'] <= 1.0) &
    (distributions_clean['post'] >= -1.5) & (distributions_clean['post'] <= 1.0)
].copy()
            
pre_data = distributions_clean['pre'].dropna()
post_data = distributions_clean['post'].dropna()

n_quantiles = 1000
quantiles = np.linspace(0, 1, n_quantiles)

pre_quantiles = np.quantile(pre_data, quantiles)
post_quantiles = np.quantile(post_data, quantiles)

# Plot by connectivity class using config
for conn_class, config in connectivity_config.items():
    mask = distributions_clean['connectivity'] == conn_class
    if mask.sum() > 0:
        pre_q = np.quantile(distributions_clean[mask]['pre'].dropna(), quantiles)
        post_q = np.quantile(distributions_clean[mask]['post'].dropna(), quantiles)
        
        axes[0].scatter(pre_q, post_q, alpha=0.5, s=10, 
                       color=config['color'], label=config['label'])
        
        quantile_diff = post_q - pre_q
        axes[1].scatter(pre_q, quantile_diff, alpha=0.5, s=10, 
                       color=config['color'], label=config['label'])

axes[0].plot([pre_quantiles.min(), pre_quantiles.max()], 
             [pre_quantiles.min(), pre_quantiles.max()], 
             'r--', linewidth=2, label='1:1 line')
axes[0].set_xlabel('Pre-Logging [m]', fontsize=12)
axes[0].set_ylabel('Post-Logging [m]', fontsize=12)
axes[0].set_title('Q-Q Plot: Pre vs Post Logging', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].set_aspect('equal', adjustable='box')
axes[0].set_xlim(-1.25, 0.75)
axes[0].set_ylim(-1.25, 0.75)

axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='No change')
axes[1].set_xlabel('Pre-Logging [m]', fontsize=12)
axes[1].set_ylabel('Depth Change (Post - Pre) [m]', fontsize=12)
axes[1].set_title('Quantile-by-Quantile Depth Change', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)
axes[1].set_xlim(-1.25, 0.75)

plt.tight_layout()
plt.show()

# %%
