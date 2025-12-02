# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

from lai_wy_scripts.dmc_vis_functions import (
    remove_flagged_buffer, fit_interaction_model_ols, fit_interaction_model_huber, plot_correlations_from_model, 
    sample_reference_ts, generate_model_distributions, plot_hypothetical_distributions, summarize_depth_shift, 
    flatten_model_results, compute_residuals, visualize_residuals
)

from wetland_dem_models.basin_attributes import WetlandBasin

spatial_buffer = 50

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Spring2025.csv"
source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'

wetland_pairs_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_all_wells.csv'
wetland_pairs = pd.read_csv(wetland_pairs_path)

# %% Run the model

model_results = []
distribution_results = []
shift_results = []
residual_results = []

rando_plot_idxs = np.random.choice(len(wetland_pairs), size=50, replace=False)

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
    logged_ts, removed_log_days = remove_flagged_buffer(logged_ts, buffer_days=0)
    reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()
    reference_ts, removed_ref_days = remove_flagged_buffer(reference_ts, buffer_days=0)

    # Find the record length and proportion of omitted days
    common_start = max(logged_ts['day'].min(), reference_ts['day'].min())
    common_end = min(logged_ts['day'].max(), reference_ts['day'].max())
    date_range = pd.date_range(start=common_start, end=common_end, freq='D')
    total_days = len(date_range)
    removed_log_in_range = [d for d in removed_log_days if common_start <= d <= common_end]
    removed_ref_in_range = [d for d in removed_ref_days if common_start <= d <= common_end]
    all_removed_days = set(removed_log_in_range) | set(removed_ref_in_range)
    n_dry_days = len(all_removed_days)

    print(f'Bottomed Out Days={n_dry_days} | {(total_days - n_dry_days) / total_days * 100}% are valid')

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

    """
    FIRST: test OLS and Huber Regression including all data (even poorly correlated below ground data)
    """

    comparison = pd.merge(
        reference_ts, 
        logged_ts, 
        how='inner', # There could b
        on='day', 
        suffixes=('_ref', '_log')
    ).drop(columns=['flag_ref', 'flag_log'])

    r_ols = fit_interaction_model_ols(
        comparison,
        x_series_name='wetland_depth_ref',
        y_series_name='wetland_depth_log',
        log_date=logging_date,
        cov_type="HC3"
    )

    r_huber = fit_interaction_model_huber(
        comparison,
        x_series_name='wetland_depth_ref',
        y_series_name='wetland_depth_log',
        log_date=logging_date
    )

    ref_sample = sample_reference_ts(
        df=comparison,
        only_pre_log=False,
        column_name='wetland_depth_ref',
        n=10_000
    )
    modeled_distributions_ols = generate_model_distributions(f_dist=ref_sample, models=r_ols)
    modeled_distributions_huber = generate_model_distributions(f_dist=ref_sample, models=r_huber)

    residuals_ols = compute_residuals(comparison, logging_date, 'wetland_depth_ref', 'wetland_depth_log', r_ols)
    residuals_huber = compute_residuals(comparison, logging_date, 'wetland_depth_ref', 'wetland_depth_log', r_huber)

    depth_shift_ols = summarize_depth_shift(model_distributions=modeled_distributions_ols)
    depth_shift_huber = summarize_depth_shift(model_distributions=modeled_distributions_huber)

    if index in rando_plot_idxs:
        plot_correlations_from_model(
            comparison,
            x_series_name='wetland_depth_ref',
            y_series_name='wetland_depth_log',
            log_date=logging_date, 
            model_results=r_ols
        )
        # plot_correlations_from_model(
        #     comparison,
        #     x_series_name='wetland_depth_ref',
        #     y_series_name='wetland_depth_log',
        #     log_date=logging_date,
        #     model_results=r_huber
        # )

        plot_hypothetical_distributions(modeled_distributions_ols, f_dist=ref_sample, bins=50)
        #plot_hypothetical_distributions(modeled_distributions_huber, f_dist=ref_sample, bins=50)

        # visualize_residuals(residuals_ols, logging_date)
        # visualize_residuals(residuals_huber, logging_date)

    r_ols_flat = flatten_model_results(r_ols, logged_id, logging_date, reference_id, "full")
    r_huber_flat = flatten_model_results(r_huber, logged_id, logging_date, reference_id, "full")
    model_results.append(r_ols_flat)
    model_results.append(r_huber_flat)

    shift_result_ols = {
        'log_id': logged_id,
        'ref_id': reference_id,
        'logging_date': logging_date,
        'data_set': 'full',
        'model_type': 'ols',
        'total_obs': total_days,
        'n_bottomed_out': n_dry_days,
        'pre_logging_modeled_mean': depth_shift_ols['mean_pre'],
        'post_logging_modeled_mean': depth_shift_ols['mean_post'], 
        'mean_depth_change': depth_shift_ols['delta_mean'], 
    }

    shift_result_huber = {
        'log_id': logged_id,
        'ref_id': reference_id,
        'logging_date': logging_date,
        'data_set': 'full',
        'model_type': 'huber',
        'total_obs': total_days,
        'n_bottomed_out': n_dry_days,
        'pre_logging_modeled_mean': depth_shift_huber['mean_pre'],
        'post_logging_modeled_mean': depth_shift_huber['mean_post'], 
        'mean_depth_change': depth_shift_huber['delta_mean'], 
    }

    shift_results.append(shift_result_ols)
    shift_results.append(shift_result_huber)

    residuals_ols['log_id'] = logged_id
    residuals_ols['ref_id'] = reference_id
    residuals_ols['model_type'] = 'OLS'
    residuals_huber['log_id'] = logged_id
    residuals_huber['ref_id'] = reference_id
    residuals_huber['model_type'] = 'HuberRLM'
    residual_results.append(residuals_ols)
    residual_results.append(residuals_huber)

    huber_distributions = pd.DataFrame(modeled_distributions_huber)
    huber_distributions['data_set'] = 'full'
    huber_distributions['model_type'] = 'huber'
    huber_distributions['log_id'] = logged_id
    huber_distributions['ref_id'] = reference_id
    huber_distributions['log_date'] = logging_date
    distribution_results.append(huber_distributions)

    """
    SECOND: run the same workflow, but using (stage >= -0.20m) 
    """

    # Filtering based on depth 
    logged_ts_trunc = logged_ts[logged_ts['wetland_depth'] >= -0.2].copy()
    reference_ts_trunc = reference_ts[reference_ts['wetland_depth'] >= -0.2].copy()

    comparison_trunc = pd.merge(
        reference_ts_trunc,
        logged_ts_trunc,
        how='inner',
        on='day',
        suffixes=('_ref', '_log')
    ).drop(columns=['flag_ref', 'flag_log'])

    r_ols = fit_interaction_model_ols(
        comparison_df=comparison_trunc,
        x_series_name='wetland_depth_ref',
        y_series_name='wetland_depth_log',
        log_date=logging_date,
        cov_type='HC3'
    )
    r_huber = fit_interaction_model_huber(
        comparison_df=comparison_trunc, 
        x_series_name='wetland_depth_ref',
        y_series_name='wetland_depth_log',
        log_date=logging_date,
    )

    ref_sample = sample_reference_ts(
        df=comparison_trunc,
        only_pre_log=False,
        column_name='wetland_depth_ref',
        n=10_000
    )

    modeled_distributions_ols = generate_model_distributions(f_dist=ref_sample, models=r_ols)
    modeled_distributions_huber = generate_model_distributions(f_dist=ref_sample, models=r_huber)

    depth_shift_ols = summarize_depth_shift(model_distributions=modeled_distributions_ols)
    depth_shift_huber = summarize_depth_shift(model_distributions=modeled_distributions_huber)

    r_ols_flat = flatten_model_results(r_ols, logged_id, logging_date, reference_id, "above_-0.2")
    r_huber_flat = flatten_model_results(r_huber, logged_id, logging_date, reference_id, "above_-0.2")
    model_results.append(r_ols_flat)
    model_results.append(r_huber_flat)

    shift_result_ols = {
        'log_id': logged_id,
        'ref_id': reference_id,
        'logging_date': logging_date,
        'data_set': 'above_-0.2',
        'model_type': 'ols',
        'total_obs': total_days,
        'n_bottomed_out': n_dry_days,
        'pre_logging_modeled_mean': depth_shift_ols['mean_pre'],
        'post_logging_modeled_mean': depth_shift_ols['mean_post'], 
        'mean_depth_change': depth_shift_ols['delta_mean']
    }

    shift_result_huber = {
        'log_id': logged_id,
        'ref_id': reference_id,
        'logging_date': logging_date,
        'data_set': 'above_-0.2',
        'model_type': 'huber',
        'total_obs': total_days,
        'n_bottomed_out': n_dry_days,
        'pre_logging_modeled_mean': depth_shift_huber['mean_pre'],
        'post_logging_modeled_mean': depth_shift_huber['mean_post'], 
        'mean_depth_change': depth_shift_huber['delta_mean']
    }

    shift_results.append(shift_result_ols)
    shift_results.append(shift_result_huber)

    """THIRD: run the same workflow, but using (stage >= 0.00m)"""

    # Filtering based on depth 
    logged_ts_trunc = logged_ts[logged_ts['wetland_depth'] >= 0.0].copy()
    reference_ts_trunc = reference_ts[reference_ts['wetland_depth'] >= 0.0].copy()

    comparison_trunc = pd.merge(
        reference_ts_trunc,
        logged_ts_trunc,
        how='inner',
        on='day',
        suffixes=('_ref', '_log')
    ).drop(columns=['flag_ref', 'flag_log'])

    r_ols = fit_interaction_model_ols(
        comparison_df=comparison_trunc,
        x_series_name='wetland_depth_ref',
        y_series_name='wetland_depth_log',
        log_date=logging_date,
        cov_type='HC3'
    )
    r_huber = fit_interaction_model_huber(
        comparison_df=comparison_trunc, 
        x_series_name='wetland_depth_ref',
        y_series_name='wetland_depth_log',
        log_date=logging_date,
    )

    ref_sample = sample_reference_ts(
        df=comparison_trunc,
        only_pre_log=False,
        column_name='wetland_depth_ref',
        n=10_000
    )

    modeled_distributions_ols = generate_model_distributions(f_dist=ref_sample, models=r_ols)
    modeled_distributions_huber = generate_model_distributions(f_dist=ref_sample, models=r_huber)

    depth_shift_ols = summarize_depth_shift(model_distributions=modeled_distributions_ols)
    depth_shift_huber = summarize_depth_shift(model_distributions=modeled_distributions_huber)

    r_ols_flat = flatten_model_results(r_ols, logged_id, logging_date, reference_id, "above_ground")
    r_huber_flat = flatten_model_results(r_huber, logged_id, logging_date, reference_id, "above_ground")
    model_results.append(r_ols_flat)
    model_results.append(r_huber_flat)

    shift_result_ols = {
        'log_id': logged_id,
        'ref_id': reference_id,
        'logging_date': logging_date,
        'data_set': 'above_ground',
        'model_type': 'ols',
        'total_obs': total_days,
        'n_bottomed_out': n_dry_days,
        'pre_logging_modeled_mean': depth_shift_ols['mean_pre'],
        'post_logging_modeled_mean': depth_shift_ols['mean_post'], 
        'mean_depth_change': depth_shift_ols['delta_mean']
    }

    shift_result_huber = {
        'log_id': logged_id,
        'ref_id': reference_id,
        'logging_date': logging_date,
        'data_set': 'above_ground',
        'model_type': 'huber',
        'total_obs': total_days,
        'n_bottomed_out': n_dry_days,
        'pre_logging_modeled_mean': depth_shift_huber['mean_pre'],
        'post_logging_modeled_mean': depth_shift_huber['mean_post'], 
        'mean_depth_change': depth_shift_huber['delta_mean']
    }

    shift_results.append(shift_result_ols)
    shift_results.append(shift_result_huber)

# %% 3.0 Combine the results into a dataframe 

shift_results_df = pd.DataFrame(shift_results)
distribution_results_df = pd.concat(distribution_results)
residual_results_df = pd.concat(residual_results)
model_results_df = pd.DataFrame(model_results)

# %% 3.1 Save the results

out_dir = "D:/depressional_lidar/data/bradford/out_data/"
shift_path = out_dir + 'logging_hypothetical_shift_results_all_wells.csv'
distributions_path = out_dir + 'logging_hypothetical_distributions_all_wells.csv'
residuals_path = out_dir + 'model_residuals_all_wells.csv'
models_path = out_dir + 'pre_post_models_all_wells.csv'

shift_results_df.to_csv(shift_path, index=False)
distribution_results_df.to_csv(distributions_path, index=False)
residual_results_df.to_csv(residuals_path, index=False)
model_results_df.to_csv(models_path, index=False)

# %% 4.0 Plot the shifts in depth

plot_df = shift_results_df.query("data_set == 'full' and model_type == 'huber'")

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


