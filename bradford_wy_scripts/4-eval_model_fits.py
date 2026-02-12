# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/out_data/"
wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_all_wells.csv'
models_path = data_dir + f'/model_info/all_wells_model_estimates_LAI_{lai_buffer_dist}m.csv'

wetland_pairs = pd.read_csv(wetland_pairs_path)
model_data = pd.read_csv(models_path)
model_data = model_data[model_data['model_type'] == 'OLS']

# %% 2.0 Filter to strong model fits

print(len(model_data)) 

strong_pairs = model_data[
    (model_data['data_set'] == 'no_dry_days') & 
    (model_data['r2_joint'] >= 0.5)
][['log_id', 'log_date', 'ref_id']]

print(len(strong_pairs))

# %% 3.0 Write the output

#strong_pairs.to_csv(f'{data_dir}/strong_ols_models_{lai_buffer_dist}m_all_wells.csv', index=False)

# %% 4.0 Diagnostic plots of model fits

plot_data = model_data.copy()
datasets = plot_data['data_set'].unique()

# %% 4.1 Histograms of joint r-squared values for each dataset

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    subset = plot_data[plot_data['data_set'] == dataset]

    ax.hist(subset['r2_joint'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')

    mean_r2 = subset['r2_joint'].mean()
    median_r2 = subset['r2_joint'].median()
    ax.axvline(mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r2:.3f}')
    ax.axvline(median_r2, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_r2:.3f}')

    ax.set_xlabel('Joint R²', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{dataset.replace("_", " ").title()}\n(n={len(subset)})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, 1)

plt.suptitle('Model Fit Quality (R²) by Dataset', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# %% 4.3 Histograms of pre and post logging r-squared values for each dataset

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for row_idx, dataset in enumerate(datasets):
    # Filter data for this dataset
    subset = plot_data[plot_data['data_set'] == dataset]
    subset = subset[subset['model_type'] == 'OLS']
    
    # Pre R² (left column)
    ax_pre = axes[row_idx, 0]
    ax_pre.hist(subset['pre_r2'], bins=20, edgecolor='black', alpha=0.7, color='grey')
    mean_pre = subset['pre_r2'].mean()
    median_pre = subset['pre_r2'].median()
    ax_pre.axvline(mean_pre, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pre:.3f}')
    ax_pre.axvline(median_pre, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_pre:.3f}')
    ax_pre.set_xlabel('Pre-logging R²', fontsize=11)
    ax_pre.set_ylabel('Frequency', fontsize=11)
    ax_pre.set_title(f'{dataset.replace("_", " ").title()} - Pre\n(n={len(subset)})', 
                     fontsize=12, fontweight='bold')
    ax_pre.legend(loc='upper left', fontsize=9)
    ax_pre.grid(True, alpha=0.3, axis='y')
    ax_pre.set_xlim(0, 1)
    
    # Post R² (right column)
    ax_post = axes[row_idx, 1]
    ax_post.hist(subset['post_r2'], bins=20, edgecolor='black', alpha=0.7, color='coral')
    mean_post = subset['post_r2'].mean()
    median_post = subset['post_r2'].median()
    ax_post.axvline(mean_post, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_post:.3f}')
    ax_post.axvline(median_post, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_post:.3f}')
    ax_post.set_xlabel('Post-logging R²', fontsize=11)
    ax_post.set_ylabel('Frequency', fontsize=11)
    ax_post.set_title(f'{dataset.replace("_", " ").title()} - Post\n(n={len(subset)})', 
                      fontsize=12, fontweight='bold')
    ax_post.legend(loc='upper left', fontsize=9)
    ax_post.grid(True, alpha=0.3, axis='y')
    ax_post.set_xlim(0, 1)

plt.suptitle('Model Fit Quality: Pre vs Post-Logging by Dataset (OLS Models)', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()


# %% 4.4 Cumulative distribution of Pearson's R for each dataset

fig, ax = plt.subplots(figsize=(12, 5))

# Color scheme for datasets
colors = {
    'full': 'red',
    'above_-0.2': 'purple',
    'above_ground': 'blue'
}

for dataset in datasets:
    subset = plot_data[plot_data['data_set'] == dataset]
    subset = subset[subset['model_type'] == 'OLS']
    r2_sorted = np.sort(subset['r2_joint'].values)
    r_sorted = r2_sorted ** 0.5
    
    cumulative_prob = np.arange(1, len(r_sorted) + 1) / len(r_sorted)

    ax.plot(r_sorted, cumulative_prob, 
            color=colors.get(dataset, 'black'),
            linewidth=2.5,
            label=f'Cummulative Pearsons R')

# Formatting
ax.set_xlabel('Pearsons R', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
ax.set_title('Cumulative Distribution of Pearsons R', 
             fontsize=15, fontweight='bold')
ax.set_xlim(-0.1, 1.05)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)


plt.tight_layout()
plt.show()
# %%
