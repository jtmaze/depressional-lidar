# %% 1.0 Libraries and file paths

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/out_data/"
wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_all_wells.csv'
strong_models_path = f"D:/depressional_lidar/data/bradford/out_data/strong_ols_models_{lai_buffer_dist}m_domain_no_dry_days.csv"
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'

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

all_well_correlations = pd.read_csv(
    f"D:/depressional_lidar/data/bradford/out_data/all_wells_correlations_domain_no_dry_days.csv"
)

# well points
well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry', 'site']]
    .rename(columns={'rtk_z': 'rtk_z'})
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)

print(well_points)


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

sorted_r_all_wells = np.sort(all_well_correlations['r_squared'])
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

# %% 3.0 Evaluate r's relationship with well distance
well_geom_lookup = (
    well_points[['wetland_id', 'geometry']]
    .drop_duplicates(subset='wetland_id')
    .assign(wetland_id=lambda df: df['wetland_id'].astype(str))
    .set_index('wetland_id')['geometry']
)

distance_corr_df = all_well_correlations.copy()
distance_corr_df['wetland1'] = distance_corr_df['wetland1'].astype(str)
distance_corr_df['wetland2'] = distance_corr_df['wetland2'].astype(str)
distance_corr_df['geom1'] = distance_corr_df['wetland1'].map(well_geom_lookup)
distance_corr_df['geom2'] = distance_corr_df['wetland2'].map(well_geom_lookup)

distance_corr_df = distance_corr_df.dropna(subset=['geom1', 'geom2', 'correlation']).copy()
distance_corr_df['distance_m'] = distance_corr_df.apply(
    lambda row: row['geom1'].distance(row['geom2']),
    axis=1
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(
    distance_corr_df['distance_m'],
    distance_corr_df['correlation'],
    s=28,
    alpha=0.6,
    color='steelblue',
    edgecolors='none'
)

lin_coef = np.polyfit(distance_corr_df['distance_m'], distance_corr_df['correlation'], 1)
lin_model = np.poly1d(lin_coef)
x_line = np.linspace(distance_corr_df['distance_m'].min(), distance_corr_df['distance_m'].max(), 200)
y_line = lin_model(x_line)
ax.plot(x_line, y_line, color='crimson', linewidth=2, label='Linear fit')

r_value = np.corrcoef(distance_corr_df['distance_m'], distance_corr_df['correlation'])[0, 1]
r_squared = r_value ** 2
eqn_text = f"y = {lin_coef[0]:.4f}x + {lin_coef[1]:.3f}\n$R^2$ = {r_squared:.3f}"
ax.text(
    0.02,
    0.98,
    eqn_text,
    transform=ax.transAxes,
    ha='left',
    va='top',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

ax.set_xlabel('Distance between wells (m)', fontsize=12)
ax.set_ylabel("Pearson correlation (r)", fontsize=12)
ax.set_title('Well Pair Correlation vs Distance', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
plt.show()

# %% Well correlations by depth range

well_data = pd.read_csv("D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv")
# %%
# find min and max well_depth_m for each wetland_id using groupby
well_depth_minmax = well_data.groupby('wetland_id')['well_depth_m'].agg(['min', 'max']).reset_index()
well_depth_minmax['depth_range'] = well_depth_minmax['max'] - well_depth_minmax['min']

well_depth_range_lookup = (
    well_depth_minmax.assign(wetland_id=lambda df: df['wetland_id'].astype(str))
    .set_index('wetland_id')['depth_range']
)

depth_range_corr_df = all_well_correlations.copy()
depth_range_corr_df['wetland1'] = depth_range_corr_df['wetland1'].astype(str)
depth_range_corr_df['wetland2'] = depth_range_corr_df['wetland2'].astype(str)
depth_range_corr_df['range1'] = depth_range_corr_df['wetland1'].map(well_depth_range_lookup)
depth_range_corr_df['range2'] = depth_range_corr_df['wetland2'].map(well_depth_range_lookup)
depth_range_corr_df['pair_min_depth_range'] = depth_range_corr_df[['range1', 'range2']].min(axis=1)
depth_range_corr_df = depth_range_corr_df.dropna(subset=['pair_min_depth_range', 'correlation']).copy()

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(
    depth_range_corr_df['pair_min_depth_range'],
    depth_range_corr_df['correlation'],
    s=28,
    alpha=0.6,
    color='steelblue',
    edgecolors='none'
)

lin_coef = np.polyfit(depth_range_corr_df['pair_min_depth_range'], depth_range_corr_df['correlation'], 1)
lin_model = np.poly1d(lin_coef)
x_line = np.linspace(depth_range_corr_df['pair_min_depth_range'].min(), depth_range_corr_df['pair_min_depth_range'].max(), 200)
y_line = lin_model(x_line)
ax.plot(x_line, y_line, color='crimson', linewidth=2, label='Linear fit')

r_value = np.corrcoef(depth_range_corr_df['pair_min_depth_range'], depth_range_corr_df['correlation'])[0, 1]
r_squared = r_value ** 2
eqn_text = f"y = {lin_coef[0]:.4f}x + {lin_coef[1]:.3f}\n$R^2$ = {r_squared:.3f}"
ax.text(
    0.02,
    0.98,
    eqn_text,
    transform=ax.transAxes,
    ha='left',
    va='top',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

ax.set_xlabel('Pair minimum well depth range (m)', fontsize=12)
ax.set_ylabel("Pearson correlation (r)", fontsize=12)
ax.set_title('Well Pair Correlation vs Minimum Depth Range', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
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

bp_all = ax.boxplot(box_data_all, positions=positions_all, patch_artist=True,
                    showfliers=False, widths=0.3)
for patch, color in zip(bp_all['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

bp_strong = ax.boxplot(box_data_strong, positions=positions_strong, patch_artist=True,
                       showfliers=False, widths=0.3)
for patch, color in zip(bp_strong['boxes'], colors_list):
    patch.set_facecolor('white')
    patch.set_edgecolor(color)
    patch.set_linewidth(1.5)
    patch.set_hatch('///')
    patch.set(hatch_color=color) if hasattr(patch, 'set_hatch_color') else None

for pos_a, data_a in zip(positions_all, box_data_all):
    ax.plot(pos_a, data_a.mean(), marker='D', color='red', markersize=5, markeredgecolor='darkred', zorder=5)
for pos_s, data_s in zip(positions_strong, box_data_strong):
    if len(data_s) > 0:
        ax.plot(pos_s, data_s.mean(), marker='D', color='red', markersize=5, markeredgecolor='darkred', zorder=5)

tick_positions = [(a + s) / 2 for a, s in zip(positions_all, positions_strong)]
ax.set_xticks(tick_positions)
ax.set_xticklabels(labels, rotation=0)

plt.ylabel('Mean Depth Change (cm)')
plt.axhline(0, color='black', linestyle=':', linewidth=2)
plt.grid(True, alpha=0.3)

legend_elements = [
    Patch(facecolor=colors[i], alpha=0.7, label=dataset_labels[d]) for i, d in enumerate(datasets)
]
legend_elements.append(Patch(facecolor='gray', alpha=0.5, label='All Pairs'))
legend_elements.append(Patch(facecolor='white', edgecolor='gray', hatch='///', label='Strong Models'))
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4)

plt.subplots_adjust(bottom=0.28)
plt.show()

# %% 5.0 Shift statistics by filtering and dataset
# Print shift statistics

for d in datasets:
    subset = shift_data[(shift_data['data_set'] == d) & (shift_data['model_type'] == 'OLS')]
    if len(subset) > 0:
        print('-----------')
        print(dataset_labels[d])
        print(f"  Mean shift: {subset['mean_depth_change'].mean():.3f} m")
        print(f"  Median shift: {subset['mean_depth_change'].median():.3f} m")
        print(f"  Std deviation: {subset['mean_depth_change'].std():.3f} m")


# %% 6.0 Boxplot showing difference between all pairs and strong models for Exc. dry obs data
# %% 6.0 Boxplot showing difference between all pairs and strong models for Exc. dry obs data

subset = shift_data[
    (shift_data['data_set'] == 'no_dry_days') &
    (shift_data['model_type'] == 'OLS')
].copy()

all_pairs = subset['mean_depth_change'].dropna() * 100
strong_pairs = subset.loc[
    subset['in_strong_models'] == 1, 'mean_depth_change'
].dropna() * 100

fig, ax = plt.subplots(figsize=(6, 5))

bp = ax.boxplot(
    [all_pairs, strong_pairs],
    tick_labels=['All Pairs', 'Strong Models'],
    patch_artist=True,
    showfliers=False,
    widths=0.55
)

# Simple contrast: filled for all pairs, hatched for strong models
bp['boxes'][0].set_facecolor('#d9d9d9')
bp['boxes'][0].set_edgecolor('#666666')
bp['boxes'][1].set_facecolor('white')
bp['boxes'][1].set_edgecolor('#666666')
bp['boxes'][1].set_hatch('///')

for med in bp['medians']:
    med.set_color('#d95f02')
    med.set_linewidth(1.8)

ax.scatter(
    [1, 2],
    [all_pairs.mean(), strong_pairs.mean()],
    marker='D',
    color='red',
    edgecolor='darkred',
    s=30,
    zorder=3
)

ax.axhline(0, color='black', linestyle=':', linewidth=1.5)
ax.set_ylabel('Mean Depth Change (cm)')
ax.grid(axis='y', alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# %%
