# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from wetland_dem_models.basin_attributes import WetlandBasin

buffer = 75

data_dir = "D:/depressional_lidar/data/bradford/"
source_dem_path = data_dir + '/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = data_dir + '/in_data/bradford_basins_assigned_wetland_ids_KG.shp'
distributions_path = data_dir + 'out_data/logging_hypothetical_distributions.csv'
wetland_pairs_path = data_dir + 'out_data/strong_ols_models.csv'
footprints = gpd.read_file(footprints_path)
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
    .rename(columns={'rtk_elevat': 'rtk_elevation'})
    .query("type in ['core_well', 'wetland_well']")
)

pairs = pd.read_csv(wetland_pairs_path)
unique_log_ids = pairs['log_id'].unique()
unique_log_ids = ['15_268']
distributions = pd.read_csv(distributions_path)

distributions = distributions[distributions['log_id'].isin(unique_log_ids)]


# %% Calculate an average hypsometric curve based on the logged_id basins

area_shifts = []

for i in unique_log_ids:

    b = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path,
        footprint=None,
        transect_buffer=buffer 
    )
    b.visualize_shape(
        show_deepest=False, 
        show_well=True, 
        show_centroid=False, 
        show_shape=False
    )
    b.plot_basin_hypsometry()

    hypsometry = b.calculate_hypsometry(method="total_cdf")
    wetland_min = b.deepest_point.elevation
    hypsometry_df = pd.DataFrame(
        {'area': hypsometry[0],
         'elevation': hypsometry[1]}
    )
    hypsometry_df['depth'] = hypsometry_df['elevation'] - wetland_min
    hypsometry_df['depth_rounded'] = hypsometry_df['depth'].round(2)
    hypsometry_df['area_scaled'] = (
        hypsometry_df['area'] / hypsometry[0].max()
    ).round(2)
    
    hypsometry_df['log_id'] = i

    distributions_clean = distributions[
        (distributions['pre'] >= -1) & (distributions['pre'] <= 1.0) &
        (distributions['post'] >= -1) & (distributions['post'] <= 1.0) &
        (distributions['log_id'] == i)
    ].copy()

    pre_data = distributions_clean['pre']
    kde_pre = stats.gaussian_kde(pre_data)
    x_pre = np.linspace(pre_data.min(), pre_data.max(), 500).round(2)
    y_pre = kde_pre(x_pre)

    pre_dist = pd.DataFrame(
        {'depth': x_pre,
        'weight': y_pre}
    )

    post_data = distributions_clean['post']
    kde_post = stats.gaussian_kde(post_data)
    x_post = np.linspace(post_data.min(), post_data.max(), 500).round(2)
    y_post = kde_post(x_post)

    post_dist = pd.DataFrame(
        {'depth': x_post,
        'weight': y_post}
    )

    pre_dist_merged = pd.merge(
        pre_dist, 
        hypsometry_df,
        how='left',
        left_on='depth',
        right_on='depth_rounded'
    )
    pre_dist_merged['area_scaled'] = pre_dist_merged['area_scaled'].fillna(0)

    post_dist_merged = pd.merge(
        post_dist, 
        hypsometry_df,
        how='left',
        left_on='depth',
        right_on='depth_rounded'
    )

    post_dist_merged['area_scaled'] = post_dist_merged['area_scaled'].fillna(0)
    
    pre_expected_area = np.average(pre_dist_merged['area_scaled'], weights=pre_dist_merged['weight'])
    post_expected_area = np.average(post_dist_merged['area_scaled'], weights=post_dist_merged['weight'])

    # Calculate proportion of time dry (area_scaled = 0)
    pre_dry_prob = pre_dist_merged[pre_dist_merged['area_scaled'] == 0]['weight'].sum()
    post_dry_prob = post_dist_merged[post_dist_merged['area_scaled'] == 0]['weight'].sum()

    # Normalize weights for proper probability
    pre_dist_merged['weight_norm'] = pre_dist_merged['weight'] / pre_dist_merged['weight'].sum()
    post_dist_merged['weight_norm'] = post_dist_merged['weight'] / post_dist_merged['weight'].sum()

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Weighted histogram of inundated area (including zeros)
    ax.hist(pre_dist_merged['area_scaled'], weights=pre_dist_merged['weight_norm'] * 100,
        bins=30, alpha=0.8, color='#333333', edgecolor='black',
        label=f'Pre-logging')
    ax.hist(post_dist_merged['area_scaled'], weights=post_dist_merged['weight_norm'] * 100,
        bins=30, alpha=0.8, color='#E69F00', edgecolor='black',
        label=f'Post-logging')
    ax.axvline(pre_expected_area, color='#333333', linestyle='--', linewidth=2,
        label=f'Pre mean: {pre_expected_area:.2f}')
    ax.axvline(post_expected_area, color='#E69F00', linestyle='--', linewidth=2,
        label=f'Post mean: {post_expected_area:.2f}')
    ax.set_xlabel('Inundated Fraction (0-1)', fontsize=14)
    ax.set_ylabel('% of Days', fontsize=14)
    ax.set_title('Example Wetland Inundation', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    summary = {
        'pre_area_mean': pre_expected_area,
        'post_area_mean': post_expected_area,
        'logged_id': i
    }
    
    area_shifts.append(summary)

# %% 

results = pd.concat([pd.DataFrame([i]) for i in area_shifts], ignore_index=True)
# %%

results['shift_nominal'] = results['post_area_mean'] -  results['pre_area_mean']
results['shift_relative'] = results['shift_nominal'] / results['pre_area_mean']

# %%
mean_relative = results['shift_relative'].mean() * 100
mean_nominal = results['shift_nominal'].mean() * 100
print(mean_nominal)
print(mean_relative)
# %%
