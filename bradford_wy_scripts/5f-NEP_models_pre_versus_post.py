# %% 1.0 Libraries and file paths
import sys
# shim for imports across directories
PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from wetland_utilities.basin_attributes import WetlandBasin
import matplotlib as mpl

data_dir = "D:/depressional_lidar/data/bradford/"

lai_buffer_dist = 150
nep_mapping_dist = 200
data_set = 'no_dry_days'

# Li et al equation info
slope_cm=0.0582 # per cm
slope_m=slope_cm * 100
b=1.9 # NOTE need to email authors for exact estimate. 
model_domain = (-1, 1)

# Model results paths
distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
strong_wetland_pairs_path = f'{data_dir}/out_data/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv'
agg_shift_data_path = f'{data_dir}/out_data/modeled_logging_stages/shift_results_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
connectivity_key_path = f'{data_dir}/bradford_wetland_connect_logging_key.xlsx'

# Geospatial paths
source_dem_path = data_dir + '/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'

# %% 2.0 Read and merge data

distributions = pd.read_csv(distributions_path)

# Only keep strong models
strong_pairs = pd.read_csv(strong_wetland_pairs_path)
distributions = distributions.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id'],
    right_on=['log_id', 'ref_id'],
    how='inner'
)

# For tracking omitted low days
dry_days = pd.read_csv(agg_shift_data_path)
dry_days = dry_days[['log_id', 'ref_id', 'total_obs', 'n_bottomed_out']] 
dry_days['modeled_pct'] = (1 - (dry_days['n_bottomed_out'] / dry_days['total_obs'])) * 100

distributions = distributions.merge(
    dry_days[['log_id', 'ref_id', 'modeled_pct']],
    on=['log_id', 'ref_id'],
    how='inner'
)

unique_ref_ids = distributions['ref_id'].unique()
unique_log_ids = distributions['log_id'].unique()

well_pts = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)

# %% 3.0 Estimate Pre versus Post NEP for each logged wetland

def swap_dry_days(depths, not_modeled_pct):
        """Replace random values with low depth based on proportion of dry days"""
        swap_depths = depths.copy().to_numpy()
        proportion = not_modeled_pct / 100

        n_to_swap = int(len(depths) * proportion)

        if n_to_swap > 0:
            swap_idx = np.random.choice(len(depths), size=n_to_swap, replace=False)
            swap_depths[swap_idx] = -1.5 # NOTE this values is arbitrary, but far below DEM elevations

        return swap_depths

def plot_depth_and_nep_maps(pre_depth_map, post_depth_map, pre_nep_map, post_nep_map, clipped_dem, nodata):
    """Plot 2x2 grid of pre/post depth and NEP maps."""
    # Mask nodata pixels
    pre_depth_viz = np.where(clipped_dem == nodata, np.nan, pre_depth_map)
    post_depth_viz = np.where(clipped_dem == nodata, np.nan, post_depth_map)
    pre_nep_viz = np.where(clipped_dem == nodata, np.nan, pre_nep_map)
    post_nep_viz = np.where(clipped_dem == nodata, np.nan, post_nep_map)

    # Color limits
    depth_vmin = np.nanmin([np.nanmin(pre_depth_viz), np.nanmin(post_depth_viz)])
    depth_vmax = np.nanmax([np.nanmax(pre_depth_viz), np.nanmax(post_depth_viz)])
    nep_vmin = np.nanmin([np.nanmin(pre_nep_viz), np.nanmin(post_nep_viz)])
    nep_vmax = np.nanmax([np.nanmax(pre_nep_viz), np.nanmax(post_nep_viz)])

    # Custom colormaps
    cmap_bwb = mpl.colors.LinearSegmentedColormap.from_list('brown_white_blue', ['saddlebrown', 'white', 'royalblue'])
    cmap_bwb.set_bad('lightgrey')
    cmap_rd_gr = mpl.colors.LinearSegmentedColormap.from_list('red_green', ['firebrick', 'white', 'forestgreen'])
    cmap_rd_gr.set_bad('lightgrey')

    # Create diverging norm centered at zero for NEP
    nep_norm = mpl.colors.TwoSlopeNorm(vmin=nep_vmin, vcenter=0, vmax=nep_vmax)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    # Top row: Depth maps
    im_d0 = axes[0, 0].imshow(pre_depth_viz, cmap=cmap_bwb, vmin=-1, vmax=depth_vmax)
    axes[0, 0].set_title('Pre-Logging Depth (m)')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    for spine in axes[0, 0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)

    im_d1 = axes[0, 1].imshow(post_depth_viz, cmap=cmap_bwb, vmin=-1, vmax=depth_vmax)
    axes[0, 1].set_title('Post-Logging Depth (m)')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    for spine in axes[0, 1].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)

    # Bottom row: NEP maps
    im_n0 = axes[1, 0].imshow(pre_nep_viz, cmap=cmap_rd_gr, norm=nep_norm)
    axes[1, 0].set_title('Pre-Logging NEP (t C ha-1 yr⁻¹)')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    for spine in axes[1, 0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)

    im_n1 = axes[1, 1].imshow(post_nep_viz, cmap=cmap_rd_gr, norm=nep_norm)
    axes[1, 1].set_title('Post-Logging NEP (t C ha-1 yr⁻¹)')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    for spine in axes[1, 1].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)

    # Colorbars
    fig.colorbar(im_d1, ax=axes[0, :], orientation='vertical', fraction=0.046, pad=0.04, label='Depth (m)')
    fig.colorbar(im_n1, ax=axes[1, :], orientation='vertical', fraction=0.046, pad=0.04, label='NEP (g C m⁻² yr⁻¹)')

    plt.show()

results = []

for i in unique_log_ids:
    depth_summaries = [] # To hold modeled depths for each log_id
    for j in unique_ref_ids:

        pair_dist = distributions[
             (distributions['ref_id'] == j) & (distributions['log_id'] == i)
        ].copy()
        if pair_dist.empty:
             continue

        modeled_pct = pair_dist['modeled_pct'].iloc[0]
        if modeled_pct < 55:
            continue

        not_modeled = 100 - modeled_pct
        pre_depths = pair_dist['pre']
        pre_depths_with_dry = swap_dry_days(pre_depths, not_modeled)
        post_depths = pair_dist['post']
        post_depths_with_dry = swap_dry_days(post_depths, not_modeled)

        pre_med = pre_depths.median()
        post_med = post_depths.median()
        summary = pd.DataFrame({
             'pre_median': [pre_med],
             'post_median': [post_med],
             'log_id': [i],
             'ref_id': [j]
        })

        depth_summaries.append(summary)

    summary_df = pd.concat(depth_summaries)
    pre_depth = summary_df['pre_median'].mean()
    post_depth = summary_df['post_median'].mean()

    well_pt = well_pts[well_pts['wetland_id'] == i]
    log_basin = WetlandBasin(
         wetland_id=i,
         source_dem_path=source_dem_path, 
         footprint=None,
         well_point_info=well_pt,
         transect_buffer=nep_mapping_dist
    )

    well_z = log_basin.well_point.elevation_dem
    clipped_dem = log_basin.clipped_dem.dem
    nodata = log_basin.clipped_dem.nodata

    pre_depth_map = (well_z - clipped_dem) + pre_depth
    post_depth_map = (well_z - clipped_dem) + post_depth

    # Mask out cells where depth < -1m. They're upland and not in Li et al 2023's model domain.
    depth_mask = (pre_depth_map >= -1) & (post_depth_map >= -1)
    pre_depth_map = np.where(depth_mask, pre_depth_map, np.nan)
    post_depth_map = np.where(depth_mask, post_depth_map, np.nan)

    pre_nep_map = (pre_depth_map * slope_m) + b
    post_nep_map = (post_depth_map * slope_m) + b

    # plot_depth_and_nep_maps(
    #     pre_depth_map, 
    #     post_depth_map, 
    #     pre_nep_map, 
    #     post_nep_map, 
    #     clipped_dem, 
    #     nodata
    # )

    pre_nep_mean = np.nanmean(pre_nep_map)
    post_nep_mean = np.nanmean(post_nep_map)

    print(pre_nep_mean, post_nep_mean)

    result = pd.DataFrame({
        'pre_nep_mean': [pre_nep_mean],
        'post_nep_mean': [post_nep_mean],
        'log_id': [i]
    })

    results.append(result)


# %% 4.0 Combine results

results_df = pd.concat(results)

connect_key = pd.read_excel(connectivity_key_path)

results_df = results_df.merge(
    connect_key[['wetland_id', 'connectivity']],
    how='left',
    left_on=['log_id'],
    right_on=['wetland_id'],
)


# %% 5.0 Simple bar graph plotting pre and post NEP by log id. 

fig, ax = plt.subplots(figsize=(14, 6))

pre_color = '#333333'
post_color = '#E69F00'

x = np.arange(len(results_df))
width = 0.25

bars1 = ax.bar(x - width/2, results_df['pre_nep_mean'], width, 
               label='Pre-Logging', color=pre_color, alpha=0.9)
bars2 = ax.bar(x + width/2, results_df['post_nep_mean'], width, 
               label='Post-Logging', color=post_color, alpha=0.9)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Wetland ID', fontsize=12)
ax.set_ylabel('NEP Mean (t C ha-1 yr⁻¹)', fontsize=12)
ax.set_title('Pre vs Post-Logging NEP Estimates by Wetland', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['log_id'], rotation=45, ha='right')
ax.legend(framealpha=0.95)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% 6.0 Group by connectivity class and calculate mean NEP increase

connectivity_config = {
    'first order': {'color': 'green', 'label': '1st Order Ditched'},
    'giw': {'color': 'blue', 'label': 'GIW'}, 
    'flow-through': {'color': 'red', 'label': 'flow-through'}
}

results_df['post_pre_nep_diff'] = results_df['post_nep_mean'] - results_df['pre_nep_mean']

# Determine if there's a sign change (pre and post NEP have different signs)
results_df['sign_change'] = (np.sign(results_df['pre_nep_mean']) != np.sign(results_df['post_nep_mean']))

fig, ax = plt.subplots(figsize=(8, 6))

connectivity_types = list(connectivity_config.keys())
x_positions = {connectivity_type: i for i, connectivity_type in enumerate(connectivity_types)}

# Plot dots/x's for each wetland colored by connectivity
for connectivity_type, config in connectivity_config.items():
    subset = results_df[results_df['connectivity'] == connectivity_type]
    x_pos = x_positions[connectivity_type]
    
    # No sign change: use circles
    no_change = subset[~subset['sign_change']]
    if len(no_change) > 0:
        ax.scatter(
            [x_pos] * len(no_change),
            no_change['post_pre_nep_diff'],
            color=config['color'],
            s=100,
            alpha=0.7,
            marker='o',
            edgecolors='black',
            linewidth=0.5
        )
    
    # Sign change: use x's
    has_change = subset[subset['sign_change']]
    if len(has_change) > 0:
        ax.scatter(
            [x_pos] * len(has_change),
            has_change['post_pre_nep_diff'],
            color=config['color'],
            s=100,
            alpha=0.7,
            marker='x',
            linewidths=2
        )

# Calculate and plot mean horizontal bars
bar_width = 0.4
for connectivity_type, config in connectivity_config.items():
    subset = results_df[results_df['connectivity'] == connectivity_type]
    if len(subset) > 0:
        mean_diff = subset['post_pre_nep_diff'].mean()
        x_pos = x_positions[connectivity_type]
        ax.hlines(
            y=mean_diff,
            xmin=x_pos - bar_width/2,
            xmax=x_pos + bar_width/2,
            color=config['color'],
            linewidth=4,
            alpha=0.9,
            zorder=10
        )

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Connectivity Type', fontsize=12)
ax.set_ylabel('Post - Pre NEP Difference (t C ha-1 yr⁻¹)', fontsize=12)
ax.set_title('NEP Change by Connectivity Type', fontsize=14, fontweight='bold')
ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels([connectivity_config[ct]['label'] for ct in connectivity_types])

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
           markeredgecolor='black', label='No Sign Change'),
    Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=8, 
           markeredgecolor='gray', label='Sign Change', linewidth=2),
    Line2D([0], [0], color='black', linewidth=4, label='Mean')
]
for connectivity_type, config in connectivity_config.items():
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=config['color'],
               markersize=8, label=config['label'])
    )

ax.legend(handles=legend_elements, framealpha=0.95, loc='best')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% 7.0 Show the pre and post NEP grouped by connectivity

# Group by connectivity and calculate mean pre and post NEP
grouped_nep = results_df.groupby('connectivity').agg({
    'pre_nep_mean': 'mean',
    'post_nep_mean': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(8, 8))

pre_color = '#333333'
post_color = '#E69F00'

connectivity_types = list(connectivity_config.keys())
x = np.arange(len(grouped_nep))
width = 0.35

# Get the values in the order of connectivity_config
pre_values = []
post_values = []
labels = []
for connectivity_type in connectivity_types:
    row = grouped_nep[grouped_nep['connectivity'] == connectivity_type]
    if len(row) > 0:
        pre_values.append(row['pre_nep_mean'].values[0])
        post_values.append(row['post_nep_mean'].values[0])
        labels.append(connectivity_config[connectivity_type]['label'])

x = np.arange(len(labels))

bars1 = ax.bar(x - width/2, pre_values, width, 
               label='Pre-Logging', color=pre_color, alpha=0.9)
bars2 = ax.bar(x + width/2, post_values, width, 
               label='Post-Logging', color=post_color, alpha=0.9)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Connectivity Type', fontsize=12)
ax.set_ylabel('Mean NEP (t C ha⁻1 yr⁻¹)', fontsize=12)
ax.set_title('Pre vs Post-Logging NEP by Connectivity Type', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, ha='center')
ax.legend(framealpha=0.95)

plt.tight_layout()
plt.show()

# %%
