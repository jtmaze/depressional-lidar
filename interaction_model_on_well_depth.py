# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

from lai_wy_scripts.dmc_vis_functions import (
    remove_flagged_buffer, fit_interaction_model, plot_correlations_from_model, sample_reference_ts,
    generate_model_distributions, plot_hypothetical_distributions, summarize_depth_shift
)

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv"
wetland_pairs_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs.csv'
wetland_pairs = pd.read_csv(wetland_pairs_path)

# %% Run the model

model_results = []

rando_plot_idxs = np.random.choice(len(wetland_pairs), size=1, replace=False)

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
    logged_ts = remove_flagged_buffer(logged_ts, buffer_days=1)
    reference_ts = remove_flagged_buffer(reference_ts, buffer_days=1)

    # Inner join on the logged and reference ts, only keeps common recods. 
    comparison = pd.merge(
        reference_ts, 
        logged_ts, 
        how='inner', 
        on='day', 
        suffixes=('_ref', '_log')
    ).drop(columns=['flag_ref', 'flag_log'])

    # Fit a model based on correlations. 
    r, m = fit_interaction_model(
        comparison,
        x_series_name='well_depth_ref',
        y_series_name='well_depth_log',
        log_date=logging_date,
        cov_type="HC3"
    )

    ref_sample = sample_reference_ts(
        df=comparison, 
        only_pre_log=False, 
        column_name='well_depth_ref',
        n=10_000
    )
    
    modeled_distributions = generate_model_distributions(f_dist=ref_sample, models=r)
    depth_shift = summarize_depth_shift(modeled_distributions)

    if index in rando_plot_idxs:
        plot_correlations_from_model(
            comparison,
            x_series_name='well_depth_ref',
            y_series_name='well_depth_log',
            log_date=logging_date,
            model_results=r
        )
        plot_hypothetical_distributions(modeled_distributions, f_dist=ref_sample, bins=50)


    result = {
        'log_id': logged_id,
        'ref_id': reference_id,
        'logging_date': logging_date,
        'pre_logging_modeled_mean': depth_shift['mean_pre'],
        'post_logging_modeled_mean': depth_shift['mean_post'], 
        'mean_depth_change': depth_shift['delta_mean']
    }

    model_results.append(result)
# %% 3.0 Quick visualizations

result_df = pd.DataFrame(model_results)


# %%

fig, ax = plt.subplots(figsize=(10, 7))

# Calculate statistics for annotation
mean_change = result_df['mean_depth_change'].mean()
std_change = result_df['mean_depth_change'].std()

# Create histogram with nice styling
n, bins, patches = ax.hist(
    result_df['mean_depth_change'], 
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

stats_text = f'N = {len(result_df)}\n'
stats_text += f'Mean = {mean_change:.3f} m\n'
stats_text += f'Std = {std_change:.3f} m\n'
stats_text += f'Positive changes: {(result_df["mean_depth_change"] > 0).sum()}/{len(result_df)}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# %%
log_summary = result_df.groupby('log_id').agg({
    'mean_depth_change': ['mean', 'std', 'count']
}).round(2)
log_summary.columns = ['mean_change', 'std_change', 'n_pairs']

depth_increases = result_df.groupby('log_id')['mean_depth_change'].apply(
    lambda x: (x > 0).sum()
).rename('n_increases')

# Combine into final summary
log_summary = log_summary.join(depth_increases)

log_summary = log_summary.sort_values('mean_change', ascending=False)


# %%
ref_summary = result_df.groupby('ref_id').agg({
    'mean_depth_change': ['mean', 'std', 'count']
}).round(2)

ref_summary.columns = ['mean_change', 'std_change', 'n_pairs']

depth_increases = result_df.groupby('ref_id')['mean_depth_change'].apply(
    lambda x: (x > 0).sum()
).rename('n_increases')

ref_summary = ref_summary.join(depth_increases)

ref_summary = ref_summary.sort_values('mean_change', ascending=False)



# %%
