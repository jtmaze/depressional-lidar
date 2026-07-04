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

data_set = 'no_dry_days'
lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/"

source_dem_path = f'{data_dir}/in_data/bradford_DEM_cleaned_USGS.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
wetland_shapes_path = f'{data_dir}/out_data/bradford_tgt_wetlands.shp'

connectivity_key_path = f'{data_dir}bradford_wetland_connect_logging_key.xlsx'
well_dry_days_path = f'{data_dir}/in_data/stage_data/bradford_wells_proportion_dry_days.csv'

distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_wetlandLAI{lai_buffer_dist}m_domain_{data_set}.csv'
strong_wetland_pairs_path = f'{data_dir}/out_data/strong_ols_models_wetland{lai_buffer_dist}m_domain_{data_set}.csv'
agg_shift_data_path = f'{data_dir}/out_data/modeled_logging_stages/shift_results_wetlandLAI{lai_buffer_dist}m_domain_{data_set}.csv'

# %% 2.0 Read and merge data 
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)

wetland_shapes = gpd.read_file(wetland_shapes_path)
connect = pd.read_excel(connectivity_key_path)[['wetland_id', 'connectivity']]

distributions = pd.read_csv(distributions_path)
# Only keep strong models
strong_pairs = pd.read_csv(strong_wetland_pairs_path)
distributions = distributions.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id'],
    right_on=['log_id', 'ref_id'],
    how='inner'
)

# %% 3.0 Inspect dry day counts from models and well data

# dry_days = pd.read_csv(agg_shift_data_path)
# dry_days = dry_days[['log_id', 'ref_id', 'total_obs', 'n_bottomed_out', 'initial_domain_days', 'filtered_domain_days']] 
# dry_days['modeled_pct'] = (1 - (dry_days['n_bottomed_out'] / dry_days['total_obs'])) * 100

# plt.figure(figsize=(6, 4))
# vals = distributions['modeled_pct'].unique()
# plt.hist(vals, bins=20, color='gray', edgecolor='black', alpha=0.8)
# plt.axvline(vals.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean = {vals.mean():.1f}%')
# plt.xlabel('Modeled portion of days (%)')
# plt.ylabel('Pair Count')
# plt.legend(framealpha=0.8)
# plt.grid(alpha=0.25)
# plt.xlim(0, 100)
# plt.tight_layout()
# plt.show()

well_dry_days = pd.read_csv(well_dry_days_path)

# %% 3.0 Calculate the flood frequency curve for each basin

# %% 3.1 Helper functions

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
    print(n_to_swap)

    if n_to_swap > 0:
        swap_idx = np.random.choice(len(depths), size=n_to_swap, replace=False)
        swap_depths[swap_idx] = -1.5 # NOTE this values is arbitrary, but far below DEM elevations

    return swap_depths

# %% 3.2 Run the calculations. 

unique_log_ids = distributions['log_id'].unique()
fd_results = []
hypsometry_data = []

for i in unique_log_ids:

    # Generate hypsometric curve for the logged basin
    well_dist = distributions[distributions['log_id'] == i].copy()
    basin = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path,
        footprint=wetland_shapes[wetland_shapes['wetland_id'] == i],
        transect_buffer=0
    )
    hyp = basin.calculate_hypsometry(method='total_cdf')
    hypsometry = pd.DataFrame({
        'area': hyp[0],
        'elevation': hyp[1]
    })
    hypsometry['depth_rel_well'] = hypsometry['elevation'] - basin.well_point.elevation_dem
    hypsometry['rel_area'] = (hypsometry['area'] / hypsometry['area'].max()).round(3)
    hypsometry_data.append(hypsometry.assign(wetland_id=i))

    not_modeled = (well_dry_days[well_dry_days['wetland_id'] == i].iloc[0]['proportion_flag2']) * 100
    print(not_modeled)

    log_valid_refs = well_dist['ref_id'].unique()
    for r in log_valid_refs:
        
        ref_well_dist = well_dist[well_dist['ref_id'] == r]

        pre_depths = ref_well_dist['pre']
        pre_depths_with_dry = swap_dry_days(pre_depths, not_modeled)
        pre_curve = calc_exceedance_curve(pre_depths_with_dry, hypsometry)
        pre_curve['pre_post'] = 'pre'

        post_depths = ref_well_dist['post']
        post_depths_with_dry = swap_dry_days(post_depths, not_modeled)
        post_curve = calc_exceedance_curve(post_depths_with_dry, hypsometry) 
        post_curve['pre_post'] = 'post'

        """
        # Plots used to check work durring developement

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
inundate_freq = inundate_freq.merge(
    connect,
    left_on='log_id',
    right_on='wetland_id',
    how='left'
)
hypsometry_data = pd.concat(hypsometry_data, ignore_index=True)


# %% %% 4.0 Q-Q plot pre versus post by connectivity

unique_log_ids = strong_pairs['log_id'].unique()
unique_ref_ids = strong_pairs['ref_id'].unique()


def mean_pre_post_curves(df):
    prob_bins = np.linspace(0.02, 0.98, 49) #NOTE need to teak these
    pre_curves = []
    post_curves = []

    for _, pair_data in df.groupby(['log_id', 'ref_id']):
        pre_data = pair_data[pair_data['pre_post'] == 'pre'].sort_values('probability')
        post_data = pair_data[pair_data['pre_post'] == 'post'].sort_values('probability')
        pre_curves.append(np.interp(prob_bins, pre_data['probability'], pre_data['inundated_fraction']))
        post_curves.append(np.interp(prob_bins, post_data['probability'], post_data['inundated_fraction']))

    pre_mean = np.mean(np.vstack(pre_curves), axis=0)
    post_mean = np.mean(np.vstack(post_curves), axis=0)

    return pd.DataFrame({
        'probability': prob_bins,
        'pre_mean': pre_mean,
        'post_mean': post_mean,
        'delta': post_mean - pre_mean
    })

# %% 4.1 Render the plot

connectivity_config = {
    'first order': {'color': '#6C5B7B', 'label': 'Ditch connected'},
    'giw': {'color': '#1B7F79', 'label': 'Unditched'}, 
    'flow-through': {'color': '#C46A1A', 'label': 'Flow-through connected'}
}

draw_order = ['flow-through', 'first order', 'giw']
legend_conn_order = ['giw', 'first order', 'flow-through']

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

class_handles = {}

for connectivity_class in draw_order:
    cfg = connectivity_config[connectivity_class]
    class_data = inundate_freq[inundate_freq['connectivity'] == connectivity_class]

    # Add one faint curve per logged wetland (averaged across valid reference pairs).
    for _, wetland_data in class_data.groupby('log_id'):
        wetland_curves = mean_pre_post_curves(wetland_data)
        wetland_pre_curve = wetland_curves['pre_mean'].to_numpy()
        wetland_post_curve = wetland_curves['post_mean'].to_numpy()
        ax.plot(
            wetland_pre_curve * 100,
            wetland_post_curve * 100,
            color=cfg['color'],
            linewidth=2,
            alpha=0.8,
            zorder=1
        )

    curve_df = mean_pre_post_curves(class_data)
    pre_curve = curve_df['pre_mean'].to_numpy()
    post_curve = curve_df['post_mean'].to_numpy()
    line = ax.plot(
        pre_curve * 100,
        post_curve * 100,
        color=cfg['color'],
        linewidth=5,
        marker='o',
        markersize=15,
        zorder=3
    )[0]
    class_handles[connectivity_class] = line

diag_line = ax.plot([0, 100], [0, 100], 'k--', linewidth=7.5, zorder=4)[0]

legend_handles = [class_handles[c] for c in legend_conn_order]
legend_labels = [connectivity_config[c]['label'] for c in legend_conn_order]
legend_handles.append(diag_line)
legend_labels.append('1:1')

ax.set_xlabel('Pre-logging inundated fraction (%)', fontsize=22)
ax.set_ylabel('Post-logging inundated fraction (%)', fontsize=22)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.legend(legend_handles, legend_labels, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.show()

# %% 5.0 Generate curves for descriptive statistics

full_q_q = mean_pre_post_curves(inundate_freq)

log_wetland_curves = []

for i in unique_log_ids:
    subset = inundate_freq[inundate_freq['log_id'] == i].copy()
    log_wetland_curves.append(mean_pre_post_curves(subset).assign(log_id=i))

log_wetland_curves = pd.concat(log_wetland_curves, ignore_index=True)

# %% 6.0 Print some statistics for general dataset

print(full_q_q[full_q_q['probability'] == 0.5])

p50_log_ids = log_wetland_curves.loc[log_wetland_curves['probability'] == 0.5].copy()

p25 = np.percentile(p50_log_ids['delta'], 25)
p75 = np.percentile(p50_log_ids['delta'], 75)
print(f"25th percentile of delta: {p25:.4f}")
print(f"75th percentile of delta: {p75:.4f}")
print(f"IQR of delta at p=0.5: {p75-p25:.4f}")
print(f"Mean delta: {p50_log_ids['delta'].mean():.4f}")
print(f"SD of delta: {p50_log_ids['delta'].std():.4f}")

log_wetland_curves = log_wetland_curves.merge(
    connect.rename(columns={'wetland_id': 'log_id'}),
    on='log_id',
    how='left'
) 

# %% 7.0 Print statistics for the different wetland connectivity classes

for conn in log_wetland_curves['connectivity'].dropna().unique():
    subset = log_wetland_curves[
        (log_wetland_curves['probability'] == 0.5) &
        (log_wetland_curves['connectivity'] == conn)
    ]
    print(
        f"{conn}: n={len(subset)}, "
        f"mean pre={subset['pre_mean'].mean():.4f}, "
        f"mean post={subset['post_mean'].mean():.4f}, "
        f"mean delta={subset['delta'].mean():.4f}"
    )

bins = {
    "wet (p=0.1–0.3)": (0.1, 0.3),
    "intermediate (p=0.3–0.6)": (0.3, 0.6),
    "dry (p=0.6–0.9)": (0.6, 0.9)
}

for conn in log_wetland_curves['connectivity'].dropna().unique():
    print(f"\nConnectivity: {conn}")
    print(f"{'Condition':<22} {'pre_mean':>10} {'post_mean':>10} {'delta':>10}")
    for label, (pmin, pmax) in bins.items():
        subset = log_wetland_curves[
            (log_wetland_curves['connectivity'] == conn) &
            (log_wetland_curves['probability'] >= pmin) &
            (log_wetland_curves['probability'] < pmax)
        ]
        pre = subset['pre_mean'].mean() * 100
        post = subset['post_mean'].mean() * 100
        delta = subset['delta'].mean() * 100
        print(f"{label:<22} {pre:10.0f} {post:10.0f} {delta:10.0f}")



# %%
