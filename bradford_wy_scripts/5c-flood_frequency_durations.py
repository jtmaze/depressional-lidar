# %% 1.0 Libraries function imports and file paths
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wetland_utilities.basin_attributes import WetlandBasin

buffer = 100
data_set = 'no_dry_days'
lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/"
source_dem_path = f'{data_dir}/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
connectivity_key_path = f'{data_dir}bradford_wetland_connect_logging_key.xlsx'
distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
strong_wetland_pairs_path = f'{data_dir}/out_data/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv'
agg_shift_data_path = f'{data_dir}/out_data/modeled_logging_stages/shift_results_LAI{lai_buffer_dist}m_domain_{data_set}.csv'

# %% 2.0 Read and merge data 
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)

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
dry_days = dry_days[['log_id', 'ref_id', 'total_obs', 'n_bottomed_out', 'initial_domain_days', 'filtered_domain_days']] 
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

# %% 3.0 Calculate the flood frequency curve for each basin

unique_log_ids = distributions['log_id'].unique()
fd_results = []

for i in unique_log_ids:

    # Generate hypsometric curve for the logged basin
    well_dist = distributions[distributions['log_id'] == i].copy()
    basin = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path,
        footprint=None,
        transect_buffer=buffer 
    )
    hyp = basin.calculate_hypsometry(method='total_cdf')
    hypsometry = pd.DataFrame({
        'area': hyp[0],
        'elevation': hyp[1]
    })
    hypsometry['depth_rel_well'] = hypsometry['elevation'] - basin.well_point.elevation_dem
    hypsometry['rel_area'] = (hypsometry['area'] / hypsometry['area'].max()).round(3)


    def calc_exceedance_curve(model_well_depths, hypsometry):

        hypsometry_sorted = hypsometry.sort_values('depth_rel_well')
        hyp_depths = hypsometry_sorted['depth_rel_well'].values
        hyp_areas = hypsometry_sorted['rel_area'].values

        model_well_depths = np.round(model_well_depths, decimals=3)
        sorted_depths = np.sort(model_well_depths)[::-1]
        n = len(sorted_depths)

        # Weibull plotting postion rank / (n + 1)
        probs = np.arange(1, n + 1) / (n + 1)

        inundated_fracs = np.interp(
            sorted_depths,
            hyp_depths,
            hyp_areas,
            left=0,
            right=hyp_areas.max()
        )

        scaled_depths = sorted_depths / (sorted_depths.max() - sorted_depths.min())

        return pd.DataFrame(
            {'probability': probs,
             'depth': sorted_depths,
             'depth_scaled': scaled_depths,
             'inundated_fraction': inundated_fracs}
        )
    
    def swap_dry_days(depths, not_modeled_pct):
        """Replace random values with -1.5 based on proportion of dry days"""
        swap_depths = depths.copy().to_numpy()
        proportion = not_modeled_pct / 100

        n_to_swap = int(len(depths) * proportion)

        if n_to_swap > 0:
            swap_idx = np.random.choice(len(depths), size=n_to_swap, replace=False)
            swap_depths[swap_idx] = -1.5 # NOTE this values is arbitrary, but far below DEM elevations

        return swap_depths

    log_valid_refs = well_dist['ref_id'].unique()

    # Get the bottomed dry (un-modeled) fraction for the specific pair. 

    for r in log_valid_refs:
        
        ref_well_dist = well_dist[well_dist['ref_id'] == r]
        modeled_pct = ref_well_dist['modeled_pct'].iloc[0]
        not_modeled = 100 - modeled_pct

        pre_depths = ref_well_dist['pre']
        pre_depths_with_dry = swap_dry_days(pre_depths, not_modeled)
        pre_curve = calc_exceedance_curve(pre_depths_with_dry, hypsometry)
        pre_curve['pre_post'] = 'pre'

        post_depths = ref_well_dist['post']
        post_depths_with_dry = swap_dry_days(post_depths, not_modeled)
        post_curve = calc_exceedance_curve(post_depths_with_dry, hypsometry) 
        post_curve['pre_post'] = 'post'

        """
        # Quick plot to observe depth curves
        plt.figure(figsize=(8, 6))
        plt.plot(pre_curve['probability'], pre_curve['depth'], label='Pre-logging', color='blue')
        plt.plot(post_curve['probability'], post_curve['depth'], label='Post-logging', color='red')
        plt.xlabel('Exceedance Probability')
        plt.ylabel('Water Depth (m)')
        plt.title(f'Depth Duration Curves for {i}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
        # Quick plot to observe area curves
        plt.figure(figsize=(8, 6))
        plt.plot(pre_curve['probability'], pre_curve['inundated_fraction'], label='Pre-logging', color='blue')
        plt.plot(post_curve['probability'], post_curve['inundated_fraction'], label='Post-logging', color='red')
        plt.xlabel('Exceedance Probability')
        plt.ylabel('Inundated Fraction')
        plt.title(f'Flood Duration Curves for {i}')
        plt.legend()
        plt.grid(True)
        plt.show()
        """
  
        result = pd.concat([post_curve, pre_curve])
        result['log_id'] = i
        result['ref_id'] = r

        fd_results.append(result)
        print(f'computed curves log={i} ref={r}')

# %% 3.0 Combine results

inundate_freq = pd.concat(fd_results)

# %% 4.0 Make pre and post curves on a single plot. Combine the pre and post curves for each log_id

unique_log_ids = strong_pairs['log_id'].unique()
unique_ref_ids = strong_pairs['ref_id'].unique()
# Color scheme
post_color = '#E69F00'  # Orange
pre_color = '#333333'  # Dark gray

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Define common probability bins for interpolation
prob_bins = np.linspace(0, 1, 51)

# Aggregate pre-logging curves across all log_ids
pre_data = inundate_freq[inundate_freq['pre_post'] == 'pre']
pre_interp = []
for l in unique_log_ids:
    for r in unique_ref_ids:
        data = pre_data[
            (pre_data['log_id'] == l) & (pre_data['ref_id'] == r)
        ].sort_values('probability')
        if data.empty:
            continue
        interp_vals = np.interp(prob_bins, data['probability'], data['inundated_fraction'])
        pre_interp.append(interp_vals)

pre_interp = np.array(pre_interp)
pre_mean = np.mean(pre_interp, axis=0)
pre_std = np.std(pre_interp, axis=0)
pre_se = pre_std / np.sqrt(pre_interp.shape[0])

# Aggregate post-logging curves across all log_ids
post_data = inundate_freq[inundate_freq['pre_post'] == 'post']
post_interp = []
for l in unique_log_ids:
    for r in unique_ref_ids:
        data = post_data[
            (post_data['log_id'] == l) & (post_data['ref_id'] == r)
        ].sort_values('probability')
        if data.empty:
            continue
        interp_vals = np.interp(prob_bins, data['probability'], data['inundated_fraction'])
        post_interp.append(interp_vals)

post_interp = np.array(post_interp)
post_mean = np.mean(post_interp, axis=0)
post_std = np.std(post_interp, axis=0)
post_se = post_std / np.sqrt(post_interp.shape[0])

# Plot pre-logging mean and standard error lines
ax.plot(prob_bins, pre_mean * 100, color=pre_color, linewidth=5, label='Pre-logging Mean')
ax.plot(prob_bins, (pre_mean - pre_se) * 100, color=pre_color, linewidth=2.5, linestyle='--', alpha=0.35, label='Pre-logging ±1 SE')
ax.plot(prob_bins, (pre_mean + pre_se) * 100, color=pre_color, linewidth=2.5, linestyle='--', alpha=0.35)

# Plot post-logging mean and standard error lines
ax.plot(prob_bins, post_mean * 100, color=post_color, linewidth=5, label='Post-logging Mean')
ax.plot(prob_bins, (post_mean - post_se)* 100, color=post_color, linewidth=2.5, linestyle='--', alpha=0.35, label='Post-logging ±1 SE')
ax.plot(prob_bins, (post_mean + post_se) * 100, color=post_color, linewidth=2.5, linestyle='--', alpha=0.35)

ax.axvline(x=0.5, color='maroon', alpha=0.7, linewidth=4, label="P=0.5")
ax.axvline(x=0.25, color='navy', alpha=0.7, linewidth=4, label="P=0.25")

ax.set_xlabel('Exceedance Probability', fontsize=16)
ax.set_ylabel('Inundated Fraction (%)', fontsize=16)
ax.set_title('Flood Duration Curves All Wetlands', fontsize=18)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 75)
ax.legend(fontsize=16)

ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.show()


# %% 5.0 Bar showing graph of inundated fraction at P=0.5

def get_inundated_at_prob(curve_df, target_prob):
    """Get the inundated fraction closest to a target exceedance probability"""
    idx = (curve_df['probability'] - target_prob).abs().idxmin()
    return curve_df.loc[idx, 'inundated_fraction']

# Extract values at target probabilities for each log_id
bar_data = []

for log_id in unique_log_ids:
    log_curves = inundate_freq[inundate_freq['log_id'] == log_id].copy()
    for ref_id in unique_ref_ids:
        ref_curves = log_curves[log_curves['ref_id'] == ref_id]

        pre_curve = ref_curves[ref_curves['pre_post'] == 'pre']
        post_curve = ref_curves[ref_curves['pre_post'] == 'post']

        if len(pre_curve) == 0 or len(post_curve) == 0:
            continue
        
        for p_val in [0.25, 0.5]:
            pre_val = get_inundated_at_prob(pre_curve, p_val)
            post_val = get_inundated_at_prob(post_curve, p_val)
            
            bar_data.append({
                'log_id': log_id,
                'ref_id': ref_id,
                'probability': p_val,
                'pre_inundated': pre_val,
                'post_inundated': post_val
            })

bar_df = pd.DataFrame(bar_data)
# Average across reference wetlands for each log_id
bar_means = bar_df.groupby(['log_id', 'probability']).agg(
    pre_mean=('pre_inundated', 'mean'),
    post_mean=('post_inundated', 'mean'),
    pre_se=('pre_inundated', 'sem'),
    post_se=('post_inundated', 'sem')
).reset_index()

bar_means['nominal_diff'] = bar_means['post_mean'] * 100 - bar_means['pre_mean'] * 100
bar_means['rel_diff'] = (bar_means['nominal_diff'] / (bar_means['pre_mean'] * 100)) * 100
# %% 5.1 Plot

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
bar_width = 0.35

for ax, p_val in zip(axes, [0.25, 0.5]):
    subset = bar_means[bar_means['probability'] == p_val].sort_values('log_id')
    x = np.arange(len(subset))
    
    ax.bar(x - bar_width/2, subset['pre_mean'], bar_width,
           yerr=subset['pre_se'], capsize=3,
           color='#333333', alpha=0.85, label='Pre-logging')
    ax.bar(x + bar_width/2, subset['post_mean'], bar_width,
           yerr=subset['post_se'], capsize=3,
           color='#E69F00', alpha=0.85, label='Post-logging')
    
    ax.set_xticks(x)
    ax.set_xticklabels(subset['log_id'], rotation=45, ha='right')
    #ax.set_ylabel('Mean Inundated Fraction (%)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

axes[-1].set_xlabel('Logged Wetland ID', fontsize=16)
plt.tight_layout()
plt.show()


# %%
