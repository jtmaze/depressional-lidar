# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/"
distributions_path = data_dir + f'/out_data/modeled_logging_stages/all_wells_hypothetical_distributions_LAI_{lai_buffer_dist}m.csv'
pairs_path = data_dir + f'out_data/strong_ols_models_{lai_buffer_dist}m_all_wells.csv'
pairs = pd.read_csv(pairs_path)
distributions = pd.read_csv(distributions_path)

unique_log_ids = distributions['log_id'].unique()
unique_ref_ids = distributions['ref_id'].unique()

combinations_list = []
for ref_id in unique_ref_ids:
    for log_id in unique_log_ids:
        combinations_list.append({
            'ref_id': ref_id,
            'log_id': log_id,
        })
pairs = pd.DataFrame(combinations_list)

pairs['ref_log'] = pairs['ref_id'] + '_' + pairs['log_id']
 

distributions = distributions[distributions['log_id'] == '15_268']
pairs = pairs[pairs['log_id'] == '15_268']

    
# %% Plot the aggregated (over all pairs) pre and post logging distributions

distributions['ref_log'] = distributions['ref_id'] + '_' + distributions['log_id']
distributions_clean = distributions[distributions['ref_log'].isin(pairs['ref_log'])]

# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

pre_data = distributions_clean['pre']
post_data = distributions_clean['post']

# Create histograms with normalized counts (% of days)
# Define common bin edges for consistent bin widths
bins = 50
bin_edges = np.linspace(-0.5, 1, bins + 1)

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
ax.set_title('Example Wetland Stage', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Depth [m]', fontsize=14)
ax.set_ylabel('% of Days', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.xlim(-0.5, 1)
plt.show()

# %% Make a Q-Q plot to compare the Pre and Post logging distributions

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

distributions_clean = distributions_clean[
    (distributions_clean['pre'] >= -1) & (distributions_clean['pre'] <= 1.0) &
    (distributions_clean['post'] >= -1) & (distributions_clean['post'] <= 1.0)
].copy()
            
pre_data = distributions_clean['pre'].dropna()
post_data = distributions_clean['post'].dropna()

n_quantiles = 1000
quantiles = np.linspace(0, 1, n_quantiles)

pre_quantiles = np.quantile(pre_data, quantiles)
post_quantiles = np.quantile(post_data, quantiles)


axes[0].scatter(pre_quantiles, post_quantiles, alpha=0.5, s=10, color='steelblue')
axes[0].plot([pre_quantiles.min(), pre_quantiles.max()], 
             [pre_quantiles.min(), pre_quantiles.max()], 
             'r--', linewidth=2, label='1:1 line')
axes[0].set_xlabel('Pre-Logging [m]', fontsize=12)
axes[0].set_ylabel('Post-Logging [m]', fontsize=12)
axes[0].set_title('Q-Q Plot: Pre vs Post Logging', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].set_aspect('equal', adjustable='box')

quantile_diff = post_quantiles - pre_quantiles
axes[1].scatter(pre_quantiles, quantile_diff, alpha=0.5, s=10, color='coral')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='No change')
# axes[1].axhline(quantile_diff.mean(), color='darkred', linestyle='-', linewidth=2, 
#                 label=f'Mean diff: {quantile_diff.mean():.3f}m')
axes[1].set_xlabel('Pre-Logging [m]', fontsize=12)
axes[1].set_ylabel('Depth Change (Post - Pre) [m]', fontsize=12)
axes[1].set_title('Quantile-by-Quantile Depth Change', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.show()

# %%
