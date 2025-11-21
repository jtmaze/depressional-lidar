# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 


data_dir = "D:/depressional_lidar/data/bradford/"
distributions_path = data_dir + '/out_data/logging_hypothetical_distributions.csv'
wetland_pairs_path = data_dir + '/out_data/strong_ols_models.csv' # NOTE changed this to only pull pairs above r-squared threshold


pairs = pd.read_csv(wetland_pairs_path)
distributions = pd.read_csv(distributions_path)

unique_log_ids = distributions['log_id'].unique()
unique_ref_ids = distributions['ref_id'].unique()

# combinations_list = []
# for ref_id in unique_ref_ids:
#     for log_id in unique_log_ids:
#         combinations_list.append({
#             'ref_id': ref_id,
#             'log_id': log_id,
#         })
# pairs = pd.DataFrame(combinations_list)
pairs['ref_log'] = pairs['ref_id'] + '_' + pairs['log_id']
 

distributions = distributions[distributions['log_id'] == '14_418']
pairs = pairs[pairs['log_id'] == '14_418']

# %%

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

for idx, row in pairs.iterrows():

    log_id = row['log_id']
    ref_id = row['ref_id']
    
    subset = distributions[
        (distributions['log_id'] == log_id) & (distributions['ref_id'] == ref_id)
    ].copy()

    pre_dist = subset['pre'].to_numpy()
    post_dist = subset['post'].to_numpy()

    kde_pre = stats.gaussian_kde(pre_dist)
    x_pre = np.linspace(pre_dist.min(), pre_dist.max(), 200)
    axes[1].plot(x_pre, kde_pre(x_pre), 
                color='grey', 
                linewidth=2.5, 
                alpha=0.3,
    )

    kde_post = stats.gaussian_kde(post_dist)
    x_post = np.linspace(post_dist.min(), post_dist.max(), 200)
    axes[0].plot(x_post, kde_post(x_post), 
                color='red', 
                linewidth=2.5, 
                alpha=0.3
    )

axes[0].set_title('Post-Logging Depth Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Pre-Logging Depth Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Depth [m]', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.xlim(-0.5, 1)
axes[0].set_ylim(0, 25)
axes[1].set_ylim(0, 25)
plt.show()
    
# %% Plot the aggregated (over all pairs) pre and post logging distributions

distributions['ref_log'] = distributions['ref_id'] + '_' + distributions['log_id']
distributions_clean = distributions[distributions['ref_log'].isin(pairs['ref_log'])]
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

pre_data = distributions_clean['pre']
post_data = distributions_clean['post']

kde_pre = stats.gaussian_kde(pre_data)
x_pre = np.linspace(pre_data.min(), pre_data.max(), 1000)
y_pre = kde_pre(x_pre)

kde_post = stats.gaussian_kde(post_data)
x_post = np.linspace(post_data.min(), post_data.max(), 1000)
y_post = kde_post(x_post)

# Plot both distributions
ax.plot(x_pre, y_pre, color='grey', linewidth=3, alpha=0.8, label='Pre-logging')
ax.plot(x_post, y_post, color='red', linewidth=3, alpha=0.8, label='Post-logging')

ax.axvline(pre_data.mean(), color='darkgrey', linestyle='--', linewidth=2, 
           label=f'Pre mean: {pre_data.mean():.3f}m')
ax.axvline(post_data.mean(), color='darkred', linestyle='--', linewidth=2,
           label=f'Post mean: {post_data.mean():.3f}m')
ax.axvline(0, color='black', linestyle=':', linewidth=4, alpha=1)

# Formatting
ax.set_title('Wetland Depth Distributions: Pre vs Post-Logging', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Depth [m]', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.xlim(-1, 1)
ax.set_ylim(0, 3)
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
