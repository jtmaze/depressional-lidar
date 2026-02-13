# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/out_data/"
wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_all_wells.csv'

datasets = ['wtd_above0_25', 'no_dry_days', 'full_obs']

model_dfs = []
shift_dfs = []

for d in datasets:
    model_path = f'{data_dir}/model_info/all_wells_model_estimates_LAI{lai_buffer_dist}m_domain_{d}.csv'
    model_df = pd.read_csv(model_path)
    model_dfs.append(model_df)

    shift_path = f'{data_dir}/modeled_logging_stages/all_wells_shift_results_LAI{lai_buffer_dist}m_domain_{d}.csv'
    shift_df = pd.read_csv(shift_path)
    shift_dfs.append(shift_df)


model_data = pd.concat(model_dfs)
shift_data = pd.concat(shift_dfs)
shift_data.head()


# %% 2.1 Compare pearson's-r for different datasets with threshold curves

colors = ['maroon', '#6fbf6f', '#7fb6d9']

for i, d in enumerate(datasets):
    subset = model_data[(model_data['data_set'] == d)
                         & (model_data['model_type'] == 'OLS')].copy()
    sorted_r = np.sort(subset['r2_joint'])
    exceedance = np.arange(len(sorted_r), 0, -1)
    plt.plot(sorted_r, exceedance, label=d, color=colors[i])

plt.ylabel("(N) Pairs Exceeding r-squared Threshold")
plt.xlabel('r-squared')
plt.axvline(0.3, color='black', linestyle=':', linewidth=2, label='Performance Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.title('Joint Models')
plt.show()


# %% 2.2 Compare pre and post r-squared for each dataset.

for i, d in enumerate(datasets):
    subset = model_data[(model_data['data_set'] == d)
                         & (model_data['model_type'] == 'OLS')].copy()
    
    # Plot pre_r2 (dashed)
    if 'pre_r2' in subset.columns:
        sorted_r = np.sort(subset['pre_r2'].dropna())
        exceedance = np.arange(len(sorted_r), 0, -1)
        plt.plot(sorted_r, exceedance, label=f'{d} - pre', linestyle='--', color=colors[i])
    
    # Plot post_r2 (solid)
    sorted_r = np.sort(subset['post_r2'])
    exceedance = np.arange(len(sorted_r), 0, -1)
    plt.plot(sorted_r, exceedance, label=f'{d} - post', linestyle='-', color=colors[i])

plt.xlabel("r-squared")
plt.xlim(0, 1)
plt.ylabel('(N) Pairs Exceeding r-squared Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Pre & Post Models')
plt.show()

# %% 3.0 Summary stats for each dataset

summary_stats = []

for d in datasets:
    subset = model_data[(model_data['data_set'] == d) 
                        & (model_data['model_type'] == 'OLS')].copy()
    
    stats_dict = {
        'dataset': d,
        'mean_r2_joint': subset['r2_joint'].mean(),
        'median_r2_joint': subset['r2_joint'].median(),
        'mean_post_r2': subset['post_r2'].mean(),
        'median_post_r2': subset['post_r2'].median(),
        'mean_pre_r2': subset['pre_r2'].mean(),
        'median_pre_r2': subset['pre_r2'].median()
    }
    summary_stats.append(stats_dict)

summary_table = pd.DataFrame(summary_stats)
print(summary_table.to_string(index=False, float_format='%.2f'))

# %% 4.0 Boxplot for shift data colored by dataset.

# x series should be dataset and 'model_type' colored by 'data_set'
fig, ax = plt.subplots(figsize=(5, 6))

box_data = []
labels = []
colors_list = []
means = []

for i, d in enumerate(datasets):
    for model_type in shift_data['model_type'].unique():
        subset = shift_data[(shift_data['data_set'] == d) & (shift_data['model_type'] == model_type)]
        if len(subset) > 0:
            box_data.append(subset['mean_depth_change'])
            labels.append(f'{d}\n{model_type}')
            colors_list.append(colors[i])
            means.append(subset['mean_depth_change'].mean())

bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)

for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for i, mean_val in enumerate(means):
    ax.plot(i + 1, mean_val, marker='D', color='red', markersize=6, markeredgecolor='darkred')

plt.ylabel('Shift (m)')
plt.xlabel('Dataset and Model Type')
plt.title('Model Impact on Depth Shift')
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linestyle=':', linewidth=2, label='No Change')
plt.grid(True, alpha=0.3)

# Add legend for dataset colors
handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.7) for i in range(len(datasets))]
plt.legend(handles, datasets, title='Dataset', loc='upper left')

plt.tight_layout()
plt.show()

# Print shift statistics
print("\nShift Statistics by Dataset:")
print("=" * 60)
for d in datasets:
    subset = shift_data[(shift_data['data_set'] == d) & (shift_data['model_type'] == 'ols')]
    if len(subset) > 0:
        print(f"\n{d} - {model_type}:")
        print(f"  Mean shift: {subset['mean_depth_change'].mean():.3f} m")
        print(f"  Median shift: {subset['mean_depth_change'].median():.3f} m")
        print(f"  Std deviation: {subset['mean_depth_change'].std():.3f} m")


# %%
