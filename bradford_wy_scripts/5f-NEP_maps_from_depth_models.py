# %% 1.0 Libraries and Packages

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
tgt_log_id = '14_500'

distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
strong_wetland_pairs_path = f'{data_dir}/out_data/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv'
agg_shift_data_path = f'{data_dir}/out_data/modeled_logging_stages/shift_results_LAI{lai_buffer_dist}m_domain_{data_set}.csv'

# Geospatial paths
source_dem_path = data_dir + '/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'


# %% 2.0 Read and merge the data

distributions = pd.read_csv(distributions_path)
distributions = distributions[distributions['log_id'] == tgt_log_id]
print(distributions.head(10))

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

plt.figure(figsize=(6, 4))
vals = distributions['modeled_pct'].unique()
plt.hist(vals, bins=20, color='gray', edgecolor='black', alpha=0.8)
plt.axvline(vals.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean = {vals.mean():.1f}%')
plt.xlabel('Modeled portion of days (%)')
plt.ylabel('Pair Count')
plt.legend(framealpha=0.8)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% 3.0 Adjust the depth distributions to account for bottomed-out days

def swap_dry_days(depths, not_modeled_pct):
        """Replace random values with low depth based on proportion of dry days"""
        swap_depths = depths.copy().to_numpy()
        proportion = not_modeled_pct / 100

        n_to_swap = int(len(depths) * proportion)

        if n_to_swap > 0:
            swap_idx = np.random.choice(len(depths), size=n_to_swap, replace=False)
            swap_depths[swap_idx] = -1.5 # NOTE this values is arbitrary, but far below DEM elevations

        return swap_depths

unique_ref_ids = distributions['ref_id'].unique()
unique_log_ids = distributions['log_id'].unique()
print(unique_ref_ids)
print(unique_log_ids)

cleaned_results = []
summaries = []

for i in unique_log_ids:
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

        summaries.append(summary)


# %% Concatonate Results and inspect distributions with medians

summary_df = pd.concat(summaries)

pre_depth = summary_df['pre_median'].mean()
post_depth = summary_df['post_median'].mean()

print(pre_depth, post_depth)

# %% Li et al equation info

slope_cm=0.0582 # per cm
slope_m=slope_cm * 100
b=1.9 # NOTE need to email authors for exact estimate. 
model_domain = (-1, 1)


# %%  Establish logged basin and extract DEM as an array

well_pt = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)
well_pt = well_pt[well_pt['wetland_id'] == tgt_log_id]

log_basin = WetlandBasin(
     wetland_id=tgt_log_id,
     source_dem_path=source_dem_path,
     footprint=None,
     well_point_info=well_pt,
     transect_buffer=nep_mapping_dist
)
well_z = log_basin.well_point.elevation_dem
clipped_dem = log_basin.clipped_dem.dem

pre_depth_map = (well_z - clipped_dem) + pre_depth
post_depth_map = (well_z - clipped_dem) + post_depth

# Mask out cells where depth < -1 m
depth_mask = (pre_depth_map >= -1) & (post_depth_map >= -1)
pre_depth_map = np.where(depth_mask, pre_depth_map, np.nan)
post_depth_map = np.where(depth_mask, post_depth_map, np.nan)

pre_nep_map = (pre_depth_map * slope_m) + b
post_nep_map = (post_depth_map * slope_m) + b

# %% Quick visualizations of pre and post depth and NEP maps

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

nodata = log_basin.clipped_dem.nodata
plot_depth_and_nep_maps(pre_depth_map, post_depth_map, pre_nep_map, post_nep_map, clipped_dem, nodata)


# %%
