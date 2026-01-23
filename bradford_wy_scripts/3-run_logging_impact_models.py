# %% 1.0 Libraries and directories

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bradford_wy_scripts.functions.wetland_logging_functions import (
    remove_flagged_buffer, fit_interaction_model_ols, fit_interaction_model_huber, plot_correlations_from_model, 
    sample_reference_ts, generate_model_distributions, plot_hypothetical_distributions, summarize_depth_shift, 
    flatten_model_results, compute_residuals
)

from wetland_utilities.basin_attributes import WetlandBasin

well_inc = 'all_wells'
min_depth_search_radius = 50
lai_buffer = 150

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'

wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer}m_{well_inc}.csv'
wetland_pairs = pd.read_csv(wetland_pairs_path)
wetland_pairs = wetland_pairs[wetland_pairs['logged_hydro_sufficient'] == True]

# %% 2.0 Load the stage data and well coordinates

stage_data = pd.read_csv(stage_path)
stage_data['well_id'] = stage_data['well_id'].str.replace('/', '.')
stage_data['day'] = pd.to_datetime(stage_data['date'])

well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
    .rename(columns={'rtk_elevat': 'rtk_elevation'})
    .query("type in ['core_well', 'wetland_well']")
)

# %% 3.0 Process each logging/reference wetland pair

# %% 3.1 Wrapper function to process a single wetland pair

def process_wetland_pair(
    row,
    stage_data: pd.DataFrame,
    well_point: gpd.GeoDataFrame,
    source_dem_path: str,
    min_depth_search_radius: int,
    plot: bool = False
):
    """
    Process a single logged/reference wetland pair and return model results.
    
    Returns:
        dict with keys: 'model_results', 'shift_results', 'residual_results', 'distribution_results'
    """
    logged_id = row['logged_id']
    reference_id = row['reference_id']
    logging_date = row['logging_date']
    print(f"Processing pair: {logged_id} vs {reference_id} (logged: {logging_date})")

    # Filter and clean stage data for this pair
    logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
    logged_ts = logged_ts.dropna(subset=['well_depth_m'])
    logged_ts, removed_log_days = remove_flagged_buffer(logged_ts, buffer_days=0)
    
    reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()
    reference_ts = reference_ts.dropna(subset=['well_depth_m'])
    reference_ts, removed_ref_days = remove_flagged_buffer(reference_ts, buffer_days=0)

    # Find the record length and proportion of omitted days
    common_start = max(logged_ts['day'].min(), reference_ts['day'].min())
    common_end = min(logged_ts['day'].max(), reference_ts['day'].max())
    date_range = pd.date_range(start=common_start, end=common_end, freq='D')
    removed_log_in_range = [d for d in removed_log_days if common_start <= d <= common_end]
    removed_ref_in_range = [d for d in removed_ref_days if common_start <= d <= common_end]
    all_removed_days = set(removed_log_in_range) | set(removed_ref_in_range)
    n_dry_days = len(all_removed_days)
    total_days = len(date_range)
    print(f'Bottomed Out Days={n_dry_days} | {(total_days - n_dry_days) / total_days * 100:.1f}% are valid')

    # Establish basin classes to get depth estimates
    ref_basin = WetlandBasin(
        wetland_id=reference_id, 
        well_point_info=well_point[well_point['wetland_id'] == reference_id],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=min_depth_search_radius
    )
    log_basin = WetlandBasin(
        wetland_id=logged_id,
        well_point_info=well_point[well_point['wetland_id'] == logged_id],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=min_depth_search_radius
    )

    # Calculate wetland depth timeseries using the deepest point on the DEM
    logged_well_diff = log_basin.well_point.elevation_dem - log_basin.deepest_point.elevation
    logged_ts['wetland_depth'] = logged_ts['well_depth_m'] + logged_well_diff
    ref_well_diff = ref_basin.well_point.elevation_dem - ref_basin.deepest_point.elevation
    reference_ts['wetland_depth'] = reference_ts['well_depth_m'] + ref_well_diff

    # Merge into comparison dataframe
    comparison = pd.merge(
        reference_ts, 
        logged_ts, 
        how='inner',
        on='day', 
        suffixes=('_ref', '_log')
    ).drop(columns=['flag_ref', 'flag_log'])

    # Sample reference distribution once for both models
    ref_sample = sample_reference_ts(
        df=comparison,
        only_pre_log=False,
        column_name='wetland_depth_ref',
        n=10_000
    )

    # Run both OLS and Huber models
    model_configs = [
        ('ols', 'OLS', fit_interaction_model_ols, {"cov_type": "HC3"}),
        ('huber', 'HuberRLM', fit_interaction_model_huber, {}),
    ]

    model_results = []
    shift_results = []
    residual_results = []
    distribution_results = []

    for model_type, model_label, fit_func, fit_kwargs in model_configs:
        # Fit the interaction model
        results = fit_func(
            comparison,
            x_series_name='wetland_depth_ref',
            y_series_name='wetland_depth_log',
            log_date=logging_date,
            **fit_kwargs
        )

        # Generate modeled distributions and compute residuals
        modeled_distributions = generate_model_distributions(f_dist=ref_sample, models=results)
        residuals = compute_residuals(comparison, logging_date, 'wetland_depth_ref', 'wetland_depth_log', results)
        depth_shift = summarize_depth_shift(model_distributions=modeled_distributions)

        # Flatten and store model results
        model_results.append(flatten_model_results(results, logged_id, logging_date, reference_id, "full"))

        # Store shift results
        shift_results.append({
            'log_id': logged_id,
            'ref_id': reference_id,
            'logging_date': logging_date,
            'data_set': 'full',
            'model_type': model_type,
            'total_obs': total_days,
            'n_bottomed_out': n_dry_days,
            'pre_logging_modeled_mean': depth_shift['mean_pre'],
            'post_logging_modeled_mean': depth_shift['mean_post'], 
            'mean_depth_change': depth_shift['delta_mean'], 
        })

        # Store residuals
        residuals['log_id'] = logged_id
        residuals['ref_id'] = reference_id
        residuals['model_type'] = model_label
        residual_results.append(residuals)

        # Store distributions (only for huber)
        if model_type == 'huber':
            dist_df = pd.DataFrame(modeled_distributions)
            dist_df['data_set'] = 'full'
            dist_df['model_type'] = model_type
            dist_df['log_id'] = logged_id
            dist_df['ref_id'] = reference_id
            dist_df['log_date'] = logging_date
            distribution_results.append(dist_df)

        # Plot if requested (only for OLS)
        if plot and model_type == 'ols':
            plot_correlations_from_model(
                comparison,
                x_series_name='wetland_depth_ref',
                y_series_name='wetland_depth_log',
                log_date=logging_date, 
                model_results=results
            )
            plot_hypothetical_distributions(modeled_distributions, f_dist=ref_sample, bins=50)

    return {
        'model_results': model_results,
        'shift_results': shift_results,
        'residual_results': residual_results,
        'distribution_results': distribution_results
    }

# %% 2.0 Run the models for each wetland pair

model_results = []
distribution_results = []
shift_results = []
residual_results = []

# View plots for random pairs of logged and reference wetlands
rando_plot_idxs = np.random.choice(len(wetland_pairs), size=len(wetland_pairs), replace=False)

for index, row in wetland_pairs.iterrows():
    pair_results = process_wetland_pair(
        row=row,
        stage_data=stage_data,
        well_point=well_point,
        source_dem_path=source_dem_path,
        min_depth_search_radius=min_depth_search_radius,
        plot=(index in rando_plot_idxs)
    )
    
    model_results.extend(pair_results['model_results'])
    shift_results.extend(pair_results['shift_results'])
    residual_results.extend(pair_results['residual_results'])
    distribution_results.extend(pair_results['distribution_results'])

# %% 3.0 Combine the results into a dataframe and save results

shift_results_df = pd.DataFrame(shift_results)
distribution_results_df = pd.concat(distribution_results)
residual_results_df = pd.concat(residual_results)
model_results_df = pd.DataFrame(model_results)

# %% 3.1 Save the results

out_dir = "D:/depressional_lidar/data/bradford/out_data/"
shift_path = out_dir + f'/modeled_logging_stages/{well_inc}_shift_results_LAI_{lai_buffer}m.csv'
distributions_path = out_dir + f'/modeled_logging_stages/{well_inc}_hypothetical_distributions_LAI_{lai_buffer}m.csv'
residuals_path = out_dir + f'/model_info/{well_inc}_residuals_LAI_{lai_buffer}m.csv'
models_path = out_dir + f'/model_info/{well_inc}_model_estimates_LAI_{lai_buffer}m.csv'

shift_results_df.to_csv(shift_path, index=False)
distribution_results_df.to_csv(distributions_path, index=False)
residual_results_df.to_csv(residuals_path, index=False)
model_results_df.to_csv(models_path, index=False)

# %% 4.0 Plot the shifts in depth

plot_df = shift_results_df.query("data_set == 'full' and model_type == 'huber'")
#plot_df = plot_df[~plot_df['log_id'].isin(['15_516', '3_244'])]
fig, ax = plt.subplots(figsize=(10, 7))

# Calculate statistics for annotation
mean_change = plot_df['mean_depth_change'].mean()
median_change = plot_df['mean_depth_change'].median()
std_change = plot_df['mean_depth_change'].std()

# Create histogram with nice styling
n, bins, patches = ax.hist(
    plot_df['mean_depth_change'], 
    bins=20, 
    edgecolor='black', 
    alpha=0.7, 
    color='steelblue',
    linewidth=1.2
)

ax.axvline(0, color='red', linestyle='--', linewidth=5, alpha=0.8, label='No change')
ax.axvline(mean_change, color='darkgreen', linestyle='--', linewidth=5, alpha=0.8, label=f'Mean: {mean_change:.3f}m')
ax.axvline(median_change, color='orange', linestyle='--', linewidth=5, alpha=0.8, label=f'Median: {median_change:.3f}m')

ax.set_xlabel('Mean Depth Change (Post - Pre Logging) [m]', fontsize=12)
ax.set_ylabel('Number of Wetland Pairs', fontsize=12)
ax.set_title('Distribution of Wetland Depth Changes After Logging', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

stats_text = f'N = {len(plot_df)}\n'
stats_text += f'Mean = {mean_change:.3f} m\n'
stats_text += f'Std = {std_change:.3f} m\n'
stats_text += f'Positive changes: {(plot_df["mean_depth_change"] > 0).sum()}/{len(plot_df)}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()


# %%


