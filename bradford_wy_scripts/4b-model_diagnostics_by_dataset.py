# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/out_data/"
wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_all_wells.csv'
strong_models_path = f"D:/depressional_lidar/data/bradford/out_data/strong_ols_models_{lai_buffer_dist}m_domain_no_dry_days.csv"

datasets = ['no_dry_days', 'full_obs']

# Create dataset label mapping
dataset_labels = {
    'no_dry_days': 'Excluding Dry Obs.',
    'full_obs': 'Including Dry Obs.'
}

model_dfs = []
shift_dfs = []

for d in datasets:
    model_path = f'{data_dir}/model_info/model_estimates_LAI{lai_buffer_dist}m_domain_{d}.csv'
    model_df = pd.read_csv(model_path)
    model_dfs.append(model_df)

    shift_path = f'{data_dir}/modeled_logging_stages/shift_results_LAI{lai_buffer_dist}m_domain_{d}.csv'
    shift_df = pd.read_csv(shift_path)
    shift_dfs.append(shift_df)


model_data = pd.concat(model_dfs)
shift_data = pd.concat(shift_dfs)
shift_data.head()

strong_models = pd.read_csv(strong_models_path)
shift_data = shift_data.merge(
    strong_models[['log_id', 'ref_id']].assign(in_strong_models=1),
    on=['log_id', 'ref_id'],
    how='left'
)
# Fill NaN values with 0 for records not in strong_models
shift_data['in_strong_models'] = shift_data['in_strong_models'].fillna(0).astype(int)

all_wells_correlations = pd.read_csv(
    f"D:/depressional_lidar/data/bradford/out_data/all_wells_correlations_domain_no_dry_days.csv"
)


# %% 2.1 Compare pearson's-r for different datasets with threshold curves

colors = ['#6fbf6f', '#7fb6d9']

fig, ax = plt.subplots(figsize=(7, 7))

for i, d in enumerate(datasets):
    subset = model_data[(model_data['data_set'] == d)
                         & (model_data['model_type'] == 'OLS')].copy()
    sorted_r = np.sort(subset['r2_joint'])
    exceedance = np.arange(len(sorted_r), 0, -1)
    exceedance_perc = (exceedance / len(exceedance)) * 100
    ax.plot(sorted_r, exceedance_perc, label=dataset_labels[d], color=colors[i], linewidth=2)

sorted_r_all_wells = np.sort(all_wells_correlations['r_squared'])
exceedance_all_wells = np.arange(len(sorted_r_all_wells), 0, -1)
exceedance_perc_all_wells = (exceedance_all_wells / len(exceedance_all_wells)) * 100
ax.plot(sorted_r_all_wells, exceedance_perc_all_wells, label='All Data (52 Wells, 1326 pairs)', 
        color='red', linewidth=2.5)

ax.set_ylabel("(%) Pairs Exceeding", fontsize=14)
ax.set_xlabel('r-squared', fontsize=14)
ax.axvline(0.3, color='black', linestyle=':', linewidth=2, label='Performance Threshold')
leg = ax.legend(fontsize=12, title='', frameon=True)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_title('Joint Models', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.show()


# %% 2.2 Compare pre and post r-squared for each dataset.

fig, ax = plt.subplots(figsize=(7, 7))

for i, d in enumerate(datasets):
    subset = model_data[(model_data['data_set'] == d)
                         & (model_data['model_type'] == 'OLS')].copy()
    
    # Plot pre_r2 (dashed)
    sorted_r_pre = np.sort(subset['pre_r2'].dropna())
    if len(sorted_r_pre) > 0:
        exceedance_pre = np.arange(len(sorted_r_pre), 0, -1)
        exceedance_perc_pre = (exceedance_pre / len(exceedance_pre)) * 100
        ax.plot(sorted_r_pre, exceedance_perc_pre, label=f'{dataset_labels[d]} - pre',
                linestyle='--', linewidth=2, color=colors[i])
    
    # Plot post_r2 (solid)
    sorted_r_post = np.sort(subset['post_r2'].dropna())
    if len(sorted_r_post) > 0:
        exceedance_post = np.arange(len(sorted_r_post), 0, -1)
        exceedance_perc_post = (exceedance_post / len(exceedance_post)) * 100
        ax.plot(sorted_r_post, exceedance_perc_post, label=f'{dataset_labels[d]} - post',
                linestyle='-', linewidth=2, color=colors[i])

ax.set_xlabel("r-squared", fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylabel('(%) Pairs Exceeding', fontsize=14)
#ax.axvline(0.3, color='black', linestyle=':', linewidth=2, label='Performance Threshold')
leg = ax.legend(fontsize=12, title='', frameon=True)
ax.grid(True, alpha=0.3)
ax.set_title('Pre & Post Models', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.show()

# %% 3.0 Summary stats for each dataset

summary_stats = []

for d in datasets:
    subset = model_data[(model_data['data_set'] == d) 
                        & (model_data['model_type'] == 'OLS')].copy()
    
    stats_dict = {
        'dataset': dataset_labels[d],
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

from matplotlib.patches import Patch

# x series should be dataset and 'model_type' colored by 'data_set'
fig, ax = plt.subplots(figsize=(7, 6))

box_data_all = []
box_data_strong = []
labels = []
colors_list = []
positions_all = []
positions_strong = []

pos = 1
for i, d in enumerate(datasets):
    for model_type in shift_data['model_type'].unique():
        subset_all = shift_data[(shift_data['data_set'] == d) & (shift_data['model_type'] == model_type)]
        subset_strong = shift_data[(shift_data['data_set'] == d) & (shift_data['model_type'] == model_type) & (shift_data['in_strong_models'] == 1)]

        box_data_all.append(subset_all['mean_depth_change'] * 100)
        box_data_strong.append(subset_strong['mean_depth_change'] * 100)
        labels.append(f'{dataset_labels[d]}\n{model_type}')
        colors_list.append(colors[i])
        positions_all.append(pos)
        positions_strong.append(pos + 0.35)
        pos += 1.2

# Full data boxplots (no hatching)
bp_all = ax.boxplot(box_data_all, positions=positions_all, patch_artist=True,
                    showfliers=False, widths=0.3)
for patch, color in zip(bp_all['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Strong models boxplots (hatching)
bp_strong = ax.boxplot(box_data_strong, positions=positions_strong, patch_artist=True,
                       showfliers=False, widths=0.3)
for patch, color in zip(bp_strong['boxes'], colors_list):
    patch.set_facecolor('white')
    patch.set_edgecolor(color)
    patch.set_linewidth(1.5)
    patch.set_hatch('///')
    patch.set(hatch_color=color) if hasattr(patch, 'set_hatch_color') else None

# Mean markers
for pos_a, data_a in zip(positions_all, box_data_all):
    ax.plot(pos_a, data_a.mean(), marker='D', color='red', markersize=5, markeredgecolor='darkred', zorder=5)
for pos_s, data_s in zip(positions_strong, box_data_strong):
    if len(data_s) > 0:
        ax.plot(pos_s, data_s.mean(), marker='D', color='red', markersize=5, markeredgecolor='darkred', zorder=5)

# X-axis labels centered between each pair
tick_positions = [(a + s) / 2 for a, s in zip(positions_all, positions_strong)]
ax.set_xticks(tick_positions)
ax.set_xticklabels(labels, rotation=0)

plt.ylabel('Mean Depth Change (cm)')
plt.axhline(0, color='black', linestyle=':', linewidth=2)
plt.grid(True, alpha=0.3)

# Legend
legend_elements = [
    Patch(facecolor=colors[i], alpha=0.7, label=dataset_labels[d]) for i, d in enumerate(datasets)
]
legend_elements.append(Patch(facecolor='gray', alpha=0.5, label='All Pairs'))
legend_elements.append(Patch(facecolor='white', edgecolor='gray', hatch='///', label='Strong Models'))
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4)

plt.subplots_adjust(bottom=0.28)
plt.show()

# %%
# Print shift statistics

for d in datasets:
    subset = shift_data[(shift_data['data_set'] == d) & (shift_data['model_type'] == 'OLS')]
    if len(subset) > 0:
        print('-----------')
        print(dataset_labels[d])
        print(f"  Mean shift: {subset['mean_depth_change'].mean():.3f} m")
        print(f"  Median shift: {subset['mean_depth_change'].median():.3f} m")
        print(f"  Std deviation: {subset['mean_depth_change'].std():.3f} m")


# %%
