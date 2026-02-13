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

min_depth_search_radius = 25 # NOTE: Only used if censoring low water table values. 
lai_buffer = 150
data_set = 'wtd_above0_25'

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'

wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer}m_all_wells.csv'
wetland_pairs = pd.read_csv(wetland_pairs_path)

# %% 2.0 Load the stage data and well coordinates

stage_data = pd.read_csv(stage_path)
stage_data['well_id'] = stage_data['well_id'].str.replace('/', '.')
stage_data['day'] = pd.to_datetime(stage_data['date'])

well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'geometry', 'rtk_z']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)

# %% 3.0 Process each logging/reference wetland pair

# %% 3.1 Wrapper function to process a single wetland pair

def timeseries_qaqc(df):
    """ 
    First, filters days with water level uncertianty (flag = 1)
    Then, filters bottomed out days (flag=2), tracking number of bottomed out days.
    """
    df_cleaned = df[df['flag'] != 1].copy()
    df_cleaned = df_cleaned.dropna(subset=['well_depth_m'])
    min_date = min(df_cleaned['date'][df_cleaned['flag'] == 0])
    max_date = max(df_cleaned['date'][df_cleaned['flag'] == 0])
    df_cleaned, bottomed_well_days = remove_flagged_buffer(df_cleaned, buffer_days=0)

    return {
        'clean_ts': df_cleaned,
        'bottomed_dates': bottomed_well_days,
        'min_date': min_date,
        'max_date': max_date,
    }

def check_domain_overlap(comp_df, log_date):
    
    """
    Filters the logging and reference data to ensure they have the same domain for interaction models
    Calculates the proportion of observations outside the domain. 
    """
    common_comp = comp_df.copy().dropna(subset=['depth_ref', 'depth_log'])
    initial_obs_cnt = len(common_comp)
    common_comp['pre_logging'] = common_comp['day'] <= log_date
    max_common = min(
        common_comp['depth_ref'][common_comp['pre_logging']].max(), 
        common_comp['depth_ref'][~common_comp['pre_logging']].max()
    )
    min_common = max(
        common_comp['depth_ref'][common_comp['pre_logging']].min(), 
        common_comp['depth_ref'][~common_comp['pre_logging']].min()
    )
    common_comp = common_comp[common_comp['depth_ref'].between(min_common, max_common)]
    filtered_obs_cnt = len(common_comp)
    print(f'Domain % Overlap: {(filtered_obs_cnt / initial_obs_cnt * 100):.2f}')

    return {
        'common_comp': common_comp,
        'initial_domain_days': initial_obs_cnt,
        'filtered_domain_days': filtered_obs_cnt
    }

def dem_depth_censor(
    reference_ts: pd.DataFrame,
    logged_ts: pd.DataFrame, 
    well_point: gpd.GeoDataFrame,
    source_dem_path: str,
    depth_thresh: float = 0.0,
    min_depth_search_radius = 50
):

    adj_log_ts = logged_ts.copy()
    adj_ref_ts = reference_ts.copy()

    # Establish basin classes to get depth estimates
    reference_id = reference_ts['well_id'].iloc[0]
    logged_id = logged_ts['well_id'].iloc[0]
    well_point_log = well_point[well_point['wetland_id'] == logged_id].copy()
    well_point_ref = well_point[well_point['wetland_id'] == reference_id].copy()

    ref_basin = WetlandBasin(
        wetland_id=reference_id, 
        well_point_info=well_point_ref,
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=min_depth_search_radius
    )
    log_basin = WetlandBasin(
        wetland_id=logged_id,
        well_point_info=well_point_log,
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=min_depth_search_radius
    )

    # Calculate wetland depth timeseries using the deepest point on the DEM
    logged_well_diff = log_basin.well_point.elevation_dem - log_basin.deepest_point.elevation
    adj_log_ts['wetland_depth'] = adj_log_ts['well_depth_m'] + logged_well_diff
    ref_well_diff = ref_basin.well_point.elevation_dem - ref_basin.deepest_point.elevation
    adj_ref_ts['wetland_depth'] = adj_ref_ts['well_depth_m'] + ref_well_diff

    # Eliminate date points where wetland depth is less than zero
    adj_log_ts = adj_log_ts[adj_log_ts['wetland_depth'] >= depth_thresh]
    adj_ref_ts = adj_ref_ts[adj_ref_ts['wetland_depth'] >= depth_thresh]

    # Remove the extra column
    adj_ref_ts.drop(columns=['wetland_depth'], inplace=True)
    adj_log_ts.drop(columns=['wetland_depth'], inplace=True)

    return adj_log_ts, adj_ref_ts

def process_wetland_pair(
    row,
    stage_data: pd.DataFrame,
    plot: bool = False, 
    data_set: str = None,
    depth_censor: bool = False,
    # Optional args if depth censor is True
    well_point: gpd.GeoDataFrame = None,
    source_dem_path: str = None,
    min_depth_search_radius: int = None, 
):
    """
    Process a single logged/reference wetland pair and return model results.
    Returns:
        dict with keys: 'model_results', 'shift_results', 'residual_results', 'distribution_results'
    """
    logged_id = row['logged_id']
    reference_id = row['reference_id']
    logging_date = row['planet_logging_date']
    print(f"Processing pair: {logged_id} vs {reference_id} (logged: {logging_date})")

    # Filter and clean stage data for this pair
    logged_raw_ts = stage_data[stage_data['well_id'] == logged_id].copy()
    logged_qaqc = timeseries_qaqc(logged_raw_ts)
    reference_raw_ts = stage_data[stage_data['well_id'] == reference_id].copy()
    reference_qaqc = timeseries_qaqc(reference_raw_ts)

    # Find the record length and proportion of omitted days
    common_end = min(reference_qaqc['max_date'], logged_qaqc['max_date'])
    common_start = max(reference_qaqc['min_date'], logged_qaqc['min_date'])
    date_range = pd.date_range(start=common_start, end=common_end, freq='D')
    n_dry_days = len(reference_qaqc['bottomed_dates'] | logged_qaqc['bottomed_dates'])
    total_days = len(date_range)
    print(f'proportion of bottomed-out days {(n_dry_days / total_days * 100):.2f}')

    if depth_censor:
        logged_ts, reference_ts = dem_depth_censor(
            reference_ts=reference_qaqc['clean_ts'],
            logged_ts=logged_qaqc['clean_ts'],
            well_point=well_point, 
            source_dem_path=source_dem_path, 
            depth_thresh=0.0,
            min_depth_search_radius=min_depth_search_radius
        )
    else: 
        reference_ts = reference_qaqc['clean_ts']
        logged_ts = logged_qaqc['clean_ts']
    
    reference_ts['depth'] = reference_ts['well_depth_m']
    logged_ts['depth'] = logged_ts['well_depth_m']

    comparison = pd.merge(
        reference_ts, 
        logged_ts, 
        how='inner',
        on='day', 
        suffixes=('_ref', '_log')
    ).drop(columns=['flag_ref', 'flag_log', 'well_depth_m_log', 'well_depth_m_ref'])

    # Run both OLS and Huber models
    model_configs = [
        ('ols', 'OLS', fit_interaction_model_ols, {"cov_type": "HC3"}),
        ('huber', 'HuberRLM', fit_interaction_model_huber, {}),
    ]

    model_results = []
    shift_results = []
    residual_results = []
    distribution_results = []

    domain_comparison = check_domain_overlap(
        comparison,
        log_date=logging_date
    )

    common_comparison = domain_comparison['common_comp']
    initial_domain_days = domain_comparison['initial_domain_days']
    filtered_domain_days = domain_comparison['filtered_domain_days']

    if len(common_comparison[~common_comparison['pre_logging']]) < 50 or len(common_comparison[common_comparison['pre_logging']]) < 50:
        print(f"WARNING: Insufficient Post-Logging Sample Counts")
        print(f"pre={len(common_comparison[common_comparison['pre_logging']])}")
        print(f"post={len(common_comparison[~common_comparison['pre_logging']])}")

        data_limited = {
            'log_id': logged_id,
            'ref_id': reference_id,
            'post_days': len(common_comparison[~common_comparison['pre_logging']]),
            'pre_days': len(common_comparison[common_comparison['pre_logging']])
        }

        print(data_limited)

        return {
            'model_results': [],
            'shift_results': [],
            'residual_results': [],
            'distribution_results': [],
            'data_limited_pairs': [data_limited]
        }

    for model_type, model_label, fit_func, fit_kwargs in model_configs:

        # Sample reference distribution 
        ref_sample = sample_reference_ts(
            df=common_comparison,
            column_name='depth_ref',
            n=10_000
        )

        # Fit the interaction model
        results = fit_func(
            common_comparison,
            x_series_name='depth_ref',
            y_series_name='depth_log',
            log_date=logging_date,
            **fit_kwargs
        )

        # Generate modeled distributions and compute residuals
        modeled_distributions = generate_model_distributions(f_dist=ref_sample, models=results)
        residuals = compute_residuals(common_comparison, logging_date, 'depth_ref', 'depth_log', results)
        depth_shift = summarize_depth_shift(model_distributions=modeled_distributions)

        # Flatten and store model results
        model_results.append(flatten_model_results(results, logged_id, logging_date, reference_id, data_set))

        # Store shift results
        shift_results.append({
            'log_id': logged_id,
            'ref_id': reference_id,
            'logging_date': logging_date,
            'data_set': data_set,
            'model_type': model_type,
            'total_obs': total_days,
            'n_bottomed_out': n_dry_days,
            'initial_domain_days': initial_domain_days,
            'filtered_domain_days': filtered_domain_days,
            'pre_logging_modeled_mean': depth_shift['mean_pre'],
            'post_logging_modeled_mean': depth_shift['mean_post'], 
            'mean_depth_change': depth_shift['delta_mean'], 
        })

        # Store residuals
        residuals['log_id'] = logged_id
        residuals['ref_id'] = reference_id
        residuals['model_type'] = model_label
        residuals['data_set'] = data_set
        residual_results.append(residuals)

        # Store distributions
        dist_df = pd.DataFrame(modeled_distributions)
        dist_df['data_set'] = data_set
        dist_df['model_type'] = model_type
        dist_df['log_id'] = logged_id
        dist_df['ref_id'] = reference_id
        distribution_results.append(dist_df)

        # Plot if requested (only for OLS)
        if plot:
            plot_correlations_from_model(
                common_comparison,
                x_series_name='depth_ref',
                y_series_name='depth_log',
                log_date=logging_date, 
                model_results=results
            )
            plot_hypothetical_distributions(modeled_distributions, f_dist=ref_sample, bins=50)

    return {
        'model_results': model_results,
        'shift_results': shift_results,
        'residual_results': residual_results,
        'distribution_results': distribution_results,
        'data_limited_pairs': []
    }

# %% 2.0 Run the models for each wetland pair

model_results = []
distribution_results = []
shift_results = []
residual_results = []
data_limited_pairs = []

# View plots for random pairs of logged and reference wetlands
rando_plot_idxs = np.random.choice(len(wetland_pairs), size=0, replace=False)

for index, row in wetland_pairs.iterrows():
    pair_results = process_wetland_pair(
        row=row,
        stage_data=stage_data,
        plot=(index in rando_plot_idxs),
        data_set=data_set,
        depth_censor=True,
        # Optional params for depth censoring
        well_point=well_point,
        source_dem_path=source_dem_path,
        min_depth_search_radius=min_depth_search_radius
    )
    
    model_results.extend(pair_results['model_results'])
    shift_results.extend(pair_results['shift_results'])
    residual_results.extend(pair_results['residual_results'])
    distribution_results.extend(pair_results['distribution_results'])
    data_limited_pairs.extend(pair_results['data_limited_pairs'])

# %% 3.0 Combine the results into a dataframe and save results

shift_results_df = pd.DataFrame(shift_results)
distribution_results_df = pd.concat(distribution_results)
residual_results_df = pd.concat(residual_results)
model_results_df = pd.DataFrame(model_results)
data_limited_pairs_df = pd.DataFrame(data_limited_pairs)

# %% 3.1 Save the results

out_dir = "D:/depressional_lidar/data/bradford/out_data/"

shift_path = out_dir + f'/modeled_logging_stages/all_wells_shift_results_LAI{lai_buffer}m_domain_{data_set}.csv'
distributions_path = out_dir + f'/modeled_logging_stages/all_wells_hypothetical_distributions_LAI{lai_buffer}m_domain_{data_set}.csv'
residuals_path = out_dir + f'/model_info/all_wells_residuals_LAI{lai_buffer}m_domain_{data_set}.csv'
models_path = out_dir + f'/model_info/all_wells_model_estimates_LAI{lai_buffer}m_domain_{data_set}.csv'
data_limited_path = out_dir + f'/model_info/data_limitted_pairs_LAI{lai_buffer}m_domain_{data_set}.csv'

shift_results_df.to_csv(shift_path, index=False)
# distribution_results_df.to_csv(distributions_path, index=False)
# residual_results_df.to_csv(residuals_path, index=False)
model_results_df.to_csv(models_path, index=False)
data_limited_pairs_df.to_csv(data_limited_path, index=False)

# %% 4.0 Plot the shifts in depth

plot_df = shift_results_df.query("data_set == 'wtd_above0_25' and model_type == 'ols'")

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


