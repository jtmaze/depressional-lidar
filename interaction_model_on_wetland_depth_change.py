# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

from lai_wy_scripts.dmc_vis_functions import (
    remove_flagged_buffer, fit_interaction_model, plot_correlations_from_model, sample_reference_ts,
    generate_model_distributions, plot_hypothetical_distributions, summarize_depth_shift, 
    summarize_inundation_shift
)

from wetland_dem_models.basin_attributes import WetlandBasin

spatial_buffer = 40

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv"
source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'

wetland_pairs_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs.csv'
wetland_pairs = pd.read_csv(wetland_pairs_path)


# %% Run the model

model_results = []
shift_results = []

rando_plot_idxs = np.random.choice(len(wetland_pairs), size=10, replace=False)

for index, row in wetland_pairs.iterrows():
    # Designate ids and logging date
    logged_id = row['logged_id']  # Adjust column names as needed
    reference_id = row['reference_id']
    logging_date = row['logging_date']
    print(f"Processing pair: {logged_id} vs {reference_id} (logged: {logging_date})")

    # Read stage data and remove the flags
    stage_data = pd.read_csv(stage_path)
    stage_data['well_id'] = stage_data['well_id'].str.replace('/', '.')
    stage_data['day'] = pd.to_datetime(stage_data['day'])
    logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
    reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()
    # logged_ts = remove_flagged_buffer(logged_ts, buffer_days=1)
    # reference_ts = remove_flagged_buffer(reference_ts, buffer_days=1)

    # Get the well points and establish basin classes to get depth estimates
    well_point = (
        gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
        .rename(columns={'rtk_elevat': 'rtk_elevation'})
        .query("type in ['core_well', 'wetland_well']")
    )
    ref_basin = WetlandBasin(
        wetland_id=reference_id, 
        well_point_info=well_point[well_point['wetland_id'] == reference_id],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=spatial_buffer
    )
    log_basin = WetlandBasin(
        wetland_id=logged_id,
        well_point_info=well_point[well_point['wetland_id'] == logged_id],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=spatial_buffer
    )

    # Calculate wetland depth timeseries using the deepest point on the DEM
    logged_well_diff = log_basin.well_point.elevation_dem - log_basin.deepest_point.elevation
    logged_ts['wetland_depth'] = logged_ts['well_depth'] + logged_well_diff
    ref_well_diff = ref_basin.well_point.elevation_dem - ref_basin.deepest_point.elevation
    reference_ts['wetland_depth'] = reference_ts['well_depth'] + ref_well_diff

    # Changing all dry wetland days to zero
    logged_ts['wetland_depth'] = logged_ts['wetland_depth'].clip(lower=0)
    reference_ts['wetland_depth'] = reference_ts['wetland_depth'].clip(lower=0)

    comparison = pd.merge(
        reference_ts, 
        logged_ts, 
        how='inner', 
        on='day', 
        suffixes=('_ref', '_log')
    ).drop(columns=['flag_ref', 'flag_log'])

    r, m = fit_interaction_model(
        comparison,
        x_series_name='wetland_depth_ref',
        y_series_name='wetland_depth_log',
        log_date=logging_date,
        cov_type="HC3"
    )

    ref_sample = sample_reference_ts(
        df=comparison,
        only_pre_log=False,
        column_name='wetland_depth_ref',
        n=10_000
    )
    modeled_distributions = generate_model_distributions(f_dist=ref_sample, models=r)
    depth_shift = summarize_depth_shift(model_distributions=modeled_distributions)
    inundation_shift = summarize_inundation_shift(model_distributions=modeled_distributions, z_thresh=0)

    if index in rando_plot_idxs:
        plot_correlations_from_model(
            comparison,
            x_series_name='wetland_depth_ref',
            y_series_name='wetland_depth_log',
            log_date=logging_date, 
            model_results=r
        )
        plot_hypothetical_distributions(modeled_distributions, f_dist=ref_sample, bins=50)

    model_results.append(r)

    shift_result = {
        'log_id': logged_id,
        'ref_id': reference_id,
        'logging_date': logging_date,
        'pre_logging_modeled_mean': depth_shift['mean_pre'],
        'post_logging_modeled_mean': depth_shift['mean_post'], 
        'mean_depth_change': depth_shift['delta_mean'], 
        'pre_inundation': inundation_shift['pre_inundation'],
        'post_inundation': inundation_shift['post_inundation'],
        'delta_inundation': inundation_shift['delta_inundation'],
    }

    shift_results.append(shift_result)


# %% 3.0 Quick visualizations

model_results_df = pd.DataFrame([
    {
        'logged_id': wetland_pairs.iloc[i]['logged_id'],
        'reference_id': wetland_pairs.iloc[i]['reference_id'],
        'pre_slope': r['pre']['slope'],
        'post_slope': r['post']['slope'],
        'slope_change': r['post']['slope'] - r['pre']['slope'],
        'p_slope_diff': r['tests']['p_slope_diff'],
        'p_intercept_diff': r['tests']['p_intercept_diff'],
        'joint_p': r['tests']['joint_p'],
        'r2': r['model_fit']['r2'],
        'n': r['model_fit']['n']
    }
    for i, r in enumerate(model_results)
])

shift_results_df = pd.DataFrame(shift_results)

# %%


fig, ax = plt.subplots(figsize=(10, 7))

# Calculate statistics for annotation
mean_change = shift_results_df['mean_depth_change'].mean()
std_change = shift_results_df['mean_depth_change'].std()

# Create histogram with nice styling
n, bins, patches = ax.hist(
    shift_results_df['mean_depth_change'], 
    bins=20, 
    edgecolor='black', 
    alpha=0.7, 
    color='steelblue',
    linewidth=1.2
)

ax.axvline(0, color='red', linestyle='--', linewidth=5, alpha=0.8, label='No change')
ax.axvline(mean_change, color='darkgreen', linestyle='--', linewidth=5, alpha=0.8, label=f'Mean: {mean_change:.3f}m')

ax.set_xlabel('Mean Depth Change (Post - Pre Logging) [m]', fontsize=12)
ax.set_ylabel('Number of Wetland Pairs', fontsize=12)
ax.set_title('Distribution of Wetland Depth Changes After Logging', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

stats_text = f'N = {len(shift_results_df)}\n'
stats_text += f'Mean = {mean_change:.3f} m\n'
stats_text += f'Std = {std_change:.3f} m\n'
stats_text += f'Positive changes: {(shift_results_df["mean_depth_change"] > 0).sum()}/{len(shift_results_df)}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()



# %%

fig, ax = plt.subplots(figsize=(10, 8))

# Color by significance
colors = ['red' if sig else 'gray' for sig in results_df['slope_sig']]
sizes = [100 if sig else 50 for sig in results_df['slope_sig']]

ax.scatter(results_df['pre_slope'], results_df['post_slope'], 
           c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)

# Add 1:1 line (no change)
min_val = min(results_df['pre_slope'].min(), results_df['post_slope'].min())
max_val = max(results_df['pre_slope'].max(), results_df['post_slope'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No change (1:1)')

# Add reference lines
ax.axhline(1, color='blue', linestyle=':', alpha=0.5, label='Post-slope = 1')
ax.axvline(1, color='green', linestyle=':', alpha=0.5, label='Pre-slope = 1')

ax.set_xlabel('Pre-logging Slope', fontsize=12)
ax.set_ylabel('Post-logging Slope', fontsize=12)
ax.set_title('Change in Slope: Pre vs Post Logging', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, label='Significant (p<0.05)'),
    Patch(facecolor='gray', alpha=0.6, label='Not significant')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()


# %%

fig, ax = plt.subplots(figsize=(10, 8))

# Color by significance of intercept difference
colors = ['red' if p < 0.05 else 'gray' for p in results_df['p_intercept_diff']]
sizes = [100 if p < 0.05 else 50 for p in results_df['p_intercept_diff']]

# Calculate intercepts from model results
pre_intercepts = [r['pre']['intercept'] for r in model_results]
post_intercepts = [r['post']['intercept'] for r in model_results]

ax.scatter(pre_intercepts, post_intercepts, 
           c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)

# Add 1:1 line (no change)
min_val = min(min(pre_intercepts), min(post_intercepts))
max_val = max(max(pre_intercepts), max(post_intercepts))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No change (1:1)')

# Add reference lines at zero
ax.axhline(0, color='blue', linestyle=':', alpha=0.5, label='Post-intercept = 0')
ax.axvline(0, color='green', linestyle=':', alpha=0.5, label='Pre-intercept = 0')

ax.set_xlabel('Pre-logging Intercept', fontsize=12)
ax.set_ylabel('Post-logging Intercept', fontsize=12)
ax.set_title('Change in Intercept: Pre vs Post Logging', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, label='Significant (p<0.05)'),
    Patch(facecolor='gray', alpha=0.6, label='Not significant')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()

# Print summary statistics for intercepts
intercept_changes = [post - pre for pre, post in zip(pre_intercepts, post_intercepts)]
sig_intercept_changes = sum([1 for p in results_df['p_intercept_diff'] if p < 0.05])

print(f"\nIntercept Change Summary:")
print(f"  Significant intercept changes: {sig_intercept_changes} of {len(results_df)}")
print(f"  Mean intercept change: {np.mean(intercept_changes):.4f}")
print(f"  Median intercept change: {np.median(intercept_changes):.4f}")

# %%

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of slope changes
ax1.hist(results_df['slope_change'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax1.set_xlabel('Change in Slope (Post - Pre)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Slope Changes', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Bar chart showing significance
sig_counts = results_df['slope_sig'].value_counts()
ax2.bar(['Not Significant', 'Significant'], 
        [sig_counts.get(False, 0), sig_counts.get(True, 0)],
        color=['gray', 'red'], edgecolor='black', alpha=0.7)
ax2.set_ylabel('Number of Pairs', fontsize=12)
ax2.set_title('Significance of Slope Changes (p<0.05)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
# %%

pre_intercepts = np.array([r['pre']['intercept'] for r in model_results])
post_intercepts = np.array([r['post']['intercept'] for r in model_results])
intercept_changes = post_intercepts - pre_intercepts

# Ensure p-values for intercept differences exist in results_df
p_intercept = results_df['p_intercept_diff']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of intercept changes
ax1.hist(intercept_changes, bins=15, edgecolor='black', alpha=0.7, color='seagreen')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax1.set_xlabel('Change in Intercept (Post - Pre)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Intercept Changes', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Bar chart showing significance of intercept differences
sig_counts = (p_intercept < 0.05).value_counts()
ax2.bar(['Not Significant', 'Significant'],
        [sig_counts.get(False, 0), sig_counts.get(True, 0)],
        color=['gray', 'red'], edgecolor='black', alpha=0.7)
ax2.set_ylabel('Number of Pairs', fontsize=12)
ax2.set_title('Significance of Intercept Changes (p<0.05)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Optional: print quick summary
print(f"Intercept change: mean={intercept_changes.mean():.4f}, median={np.median(intercept_changes):.4f}")
print(f"Significant intercept changes (p<0.05): {(p_intercept < 0.05).sum()} of {len(p_intercept)}")

# %%








