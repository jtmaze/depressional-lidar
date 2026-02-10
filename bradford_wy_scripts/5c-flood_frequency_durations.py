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
lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/"
source_dem_path = data_dir + '/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
connectivity_key_path = connectivity_key_path = data_dir + 'bradford_wetland_connect_logging_key.xlsx'
distributions_path = data_dir + f'/out_data/modeled_logging_stages/all_wells_hypothetical_distributions_LAI_{lai_buffer_dist}m.csv'
wetland_pairs_path = data_dir + f'out_data/strong_ols_models_{lai_buffer_dist}m_all_wells.csv'


well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry']]
    .query("type in ['core_well', 'wetland_well']")
)

pairs = pd.read_csv(wetland_pairs_path)
unique_log_ids = pairs['log_id'].unique().tolist()

distributions = pd.read_csv(distributions_path)
# Filter distributions for log and ref ids in the strong data
distributions = distributions.merge(
    pairs[['log_id', 'ref_id']],
    on=['log_id', 'ref_id'],
    how='inner'
)

# %% 3.0 Calculate the flood frequency curve for each basin

fd_results = []

for i in unique_log_ids:

    well_dist = distributions[distributions['log_id'] == i]
    b_f = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path,
        footprint=None,
        transect_buffer=buffer 
    )
    h = b_f.calculate_hypsometry(method='total_cdf')
    hypsometry = pd.DataFrame({
        'area': h[0],
        'elevation': h[1]
    })
    hypsometry['depth'] = hypsometry['elevation'] - b_f.well_point.elevation_dem
    hypsometry['rel_area'] = (hypsometry['area'] / hypsometry['area'].max()).round(2)

    # ----------------------------------------------------------------------
    # NOTE: This step will go away in revised version of analysis
    # b50_min = WetlandBasin(
    #     wetland_id=i,
    #     well_point_info=well_point[well_point['wetland_id'] == i],
    #     source_dem_path=source_dem_path,
    #     footprint=None,
    #     transect_buffer=50 
    # ).deepest_point.elevation

    # diff = b50_min - b_f.deepest_point.elevation
    # print(diff)
    # well_dist = distributions[distributions['log_id'] == i].copy()
    # well_dist['pre'] = well_dist['pre'] + diff
    # well_dist['post'] = well_dist['post'] + diff
    # ----------------------------------------------------------------------

    def calc_exceedance_curve(depths, hypsometry):

        hypsometry_sorted = hypsometry.sort_values('depth')
        hyp_depths = hypsometry_sorted['depth'].values
        hyp_areas = hypsometry_sorted['rel_area'].values

        depths = np.round(depths, decimals=3)
        sorted_depths = np.sort(depths)[::-1]
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
    
    pre_curve = calc_exceedance_curve(well_dist['pre'], hypsometry)
    pre_curve['pre_post'] = 'pre'
    post_curve = calc_exceedance_curve(well_dist['post'], hypsometry) 
    post_curve['pre_post'] = 'post'

  
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
  
    result = pd.concat([post_curve, pre_curve])
    result['log_id'] = i

    fd_results.append(result)

# %% 3.0 Combine results and join the connectivity key

inundate_freq = pd.concat(fd_results)
connect = pd.read_excel(connectivity_key_path)

inundate_freq = inundate_freq.merge(
    connect[['well_id', 'connectivity']],
    left_on='log_id',
    right_on='well_id',
)

# %% 4.0 Two panel plot with curves for each log_id. Panel A is pre, and Panel B is post

connectivity_config = {
    'flow-through': {'color': 'red', 'label': 'Flow-through'},
    'first order': {'color': 'orange', 'label': '1st Order Ditched'},
    'giw': {'color': 'blue', 'label': 'GIW'}
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Panel A: Pre-logging curves
pre_data = inundate_freq[inundate_freq['pre_post'] == 'pre']
for log_id in unique_log_ids:
    log_data = pre_data[pre_data['log_id'] == log_id]
    connectivity = log_data['connectivity'].iloc[0]
    color = connectivity_config[connectivity]['color']
    ax1.plot(log_data['probability'], log_data['inundated_fraction'], 
             color=color, alpha=0.7, linewidth=2)

ax1.set_xlabel('Exceedance Probability')
ax1.set_ylabel('Inundated Fraction')
ax1.set_title('Panel A: Pre-logging Flood Duration Curves')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Panel B: Post-logging curves
post_data = inundate_freq[inundate_freq['pre_post'] == 'post']
for log_id in unique_log_ids:
    log_data = post_data[post_data['log_id'] == log_id]
    connectivity = log_data['connectivity'].iloc[0]
    color = connectivity_config[connectivity]['color']
    ax2.plot(log_data['probability'], log_data['inundated_fraction'], 
             color=color, alpha=0.7, linewidth=2)

ax2.set_xlabel('Exceedance Probability')
ax2.set_ylabel('Inundated Fraction')
ax2.set_title('Panel B: Post-logging Flood Duration Curves')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Create a single legend at the bottom
handles = [plt.Line2D([0], [0], color=config['color'], linewidth=2) for config in connectivity_config.values()]
labels = [config['label'] for config in connectivity_config.values()]
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=14)

plt.show()


# %% 5.0 Make pre and post curves on a single plot. Combine the pre and post curves for each log_id

# Color scheme
post_color = '#E69F00'  # Orange
pre_color = '#333333'  # Dark gray

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Define common probability bins for interpolation
prob_bins = np.linspace(0, 1, 101)

# Aggregate pre-logging curves across all log_ids
pre_data = inundate_freq[inundate_freq['pre_post'] == 'pre']
pre_interp = []
for log_id in unique_log_ids:
    log_data = pre_data[pre_data['log_id'] == log_id].sort_values('probability')
    interp_vals = np.interp(prob_bins, log_data['probability'], log_data['inundated_fraction'])
    pre_interp.append(interp_vals)

pre_interp = np.array(pre_interp)
pre_mean = np.mean(pre_interp, axis=0)
pre_se = np.std(pre_interp, axis=0) / np.sqrt(pre_interp.shape[0])

# Aggregate post-logging curves across all log_ids
post_data = inundate_freq[inundate_freq['pre_post'] == 'post']
post_interp = []
for log_id in unique_log_ids:
    log_data = post_data[post_data['log_id'] == log_id].sort_values('probability')
    interp_vals = np.interp(prob_bins, log_data['probability'], log_data['inundated_fraction'])
    post_interp.append(interp_vals)

post_interp = np.array(post_interp)
post_mean = np.mean(post_interp, axis=0)
post_se = np.std(post_interp, axis=0) / np.sqrt(post_interp.shape[0])

# Plot pre-logging mean and standard error lines
ax.plot(prob_bins, pre_mean * 100, color=pre_color, linewidth=5, label='Pre-logging Mean')
ax.plot(prob_bins, (pre_mean - pre_se) * 100, color=pre_color, linewidth=2.5, linestyle='--', alpha=0.35, label='Pre-logging ±1 SE')
ax.plot(prob_bins, (pre_mean + pre_se) * 100, color=pre_color, linewidth=2.5, linestyle='--', alpha=0.35)

# Plot post-logging mean and standard error lines
ax.plot(prob_bins, post_mean * 100, color=post_color, linewidth=5, label='Post-logging Mean')
ax.plot(prob_bins, (post_mean - post_se)* 100, color=post_color, linewidth=2.5, linestyle='--', alpha=0.35, label='Post-logging ±1 SE')
ax.plot(prob_bins, (post_mean + post_se) * 100, color=post_color, linewidth=2.5, linestyle='--', alpha=0.35)

ax.axvline(x=0.5, color='maroon', alpha=0.7, linewidth=2.5, label="P=0.5")

ax.set_xlabel('Exceedance Probability', fontsize=16)
ax.set_ylabel('Inundated Fraction (%)', fontsize=16)
ax.set_title('Flood Duration Curves All Wetlands', fontsize=18)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
ax.legend(fontsize=16)

ax.tick_params(axis='both', which='major', labelsize=14)


plt.tight_layout()
plt.show()


# %% 6.0 Print the median (p50) inundated fraction shift for each well_id

p50_results = []

for log_id in unique_log_ids:
    well_data = inundate_freq[inundate_freq['log_id'] == log_id]
    connectivity = well_data['connectivity'].iloc[0]

    pre = well_data[well_data['pre_post'] == 'pre'].sort_values('probability')
    post = well_data[well_data['pre_post'] == 'post'].sort_values('probability')

    pre_p50 = np.interp(0.5, pre['probability'], pre['inundated_fraction'])
    post_p50 = np.interp(0.5, post['probability'], post['inundated_fraction'])
    shift = post_p50 - pre_p50

    p50_results.append({
        'log_id': log_id,
        'connectivity': connectivity,
        'pre_p50': round(pre_p50 * 100, 2),
        'post_p50': round(post_p50 * 100, 2),
        'shift_pct': round(shift * 100, 2)
    })

p50_df = pd.DataFrame(p50_results).sort_values('shift_pct', ascending=False)
print(p50_df.to_string(index=False))

# Aggregate summary
print(f"\nAggregate median inundated fraction shift:")
print(f"  Pre P50 Mean:   {p50_df['pre_p50'].mean():.2f}%")
print(f"  Post P50 Mean:   {p50_df['post_p50'].mean():.2f}%")
print(f"  Mean shift:   {p50_df['shift_pct'].mean():.2f}%")
print(f"  Median shift: {p50_df['shift_pct'].median():.2f}%")
print(f"  Std shift:    {p50_df['shift_pct'].std():.2f}%")

# By connectivity class
print("\nBy connectivity class:")
print(p50_df.groupby('connectivity')['shift_pct'].agg(['mean', 'median', 'std', 'count']).to_string())

# %%
