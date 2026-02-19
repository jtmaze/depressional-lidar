# %% 1.0 Libraries, function imports and filepaths

import sys

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from bradford_wy_scripts.functions.lai_vis_functions import read_concatonate_lai

lai_buffer_dist = 150
model_type = 'OLS'  
data_set = 'no_dry_days'  

data_dir = "D:/depressional_lidar/data/bradford/"
lai_dir = data_dir + f'/in_data/hydro_forcings_and_LAI/well_buffer_{lai_buffer_dist}m_nomasking/'

# Path to results
shift_path = data_dir + f'out_data/modeled_logging_stages/shift_results_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
#distributions_path = data_dir + f'out_data/modeled_logging_stages/all_wells_hypothetical_distributions_LAI_{lai_buffer_dist}m.csv'
models_path = data_dir + f'out_data/model_info/model_estimates_LAI_{lai_buffer_dist}m.csv'

# Path to wetland pairs, connnectivity key and strong model fits
wetland_pairs_path = data_dir + f'/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_all_wells.csv'
connectivity_key_path = data_dir + 'bradford_wetland_connect_logging_key.xlsx'
strong_pairs = data_dir + f'out_data/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv'

wetland_pairs = pd.read_csv(wetland_pairs_path)
strong_pairs = pd.read_csv(strong_pairs)
shift_data = pd.read_csv(shift_path)

# NOTE: Filters shift data where model fits were worse than threshold
shift_data = shift_data.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id', 'logging_date'],
    right_on=['log_id', 'ref_id', 'log_date'],
    how='inner'
)
shift_data = shift_data[shift_data['mean_depth_change'] < 1]

print(len(shift_data))
print(len(strong_pairs))

# %% 2.0 Histogram of inundation shift data for a specific model and dataset

plot_df = shift_data[
    (shift_data['model_type'] == model_type) & (shift_data['data_set'] == data_set)
]

mean_change = plot_df['mean_depth_change'].mean()
median_change = plot_df['mean_depth_change'].median()
std_change = plot_df['mean_depth_change'].std()

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    plot_df['mean_depth_change'], 
    bins=20, 
    edgecolor='black', 
    alpha=0.7, 
    color='steelblue',
    linewidth=1.2
)

ax.axvline(0, color='grey', linestyle='--', linewidth=2, alpha=0.8)
ax.axvline(mean_change, color='navy', linestyle='--', linewidth=4, alpha=1)

ax.set_title('Modeled Stage Increase (Unlogged - Logged)', fontsize=18, fontweight='bold')
ax.set_xlabel('Mean Depth Difference [m]', fontsize=16)
ax.set_ylabel('Number of Pairs', fontsize=18)
ax.tick_params(axis='both', labelsize=14)  # Add this line to increase tick label size

# Add stats text
stats_text = f'Mean = {mean_change:.3f}m\nStd = {std_change:.3f}m'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=18,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# %% 3.0 LAI change biplot (unclassified by connectivity)

# %% 3.1 Calculate the LAI differences for every pair

lai_roll_diffs = []

for idx, p in wetland_pairs.iterrows():

    date = p['planet_logging_date']
    log_id = p['logged_id']
    logged_lai = read_concatonate_lai(
        lai_dir,
        log_id,
        "include_wetlands_250m",
        upper_bound=5.5,
        lower_bound=0.5
    )
    logged_lai = logged_lai[logged_lai['date'] >= '2019-01-01']

    ref_id = p['reference_id']
    reference_lai = read_concatonate_lai(
        lai_dir, 
        ref_id,
        "includes_wetlands_250m",
        upper_bound=5.5,
        lower_bound=0.5
    )
    reference_lai = reference_lai[reference_lai['date'] >= '2019-01-01']

    merged_lai = logged_lai[['date', 'roll_yr']].merge(
        reference_lai[['date', 'roll_yr']],
        on='date', 
        suffixes=('_log', '_ref'),
        how='outer'
    )
    merged_lai['roll_yr_diff'] = merged_lai['roll_yr_log'] - merged_lai['roll_yr_ref']

    pre_mean = merged_lai[merged_lai['date'] <= date]['roll_yr_diff'].mean()
    post_mean = merged_lai[merged_lai['date'] > date]['roll_yr_diff'].mean()

    roll_diff_change = pre_mean - post_mean
    print(roll_diff_change)

    lai_roll_diffs.append(roll_diff_change)

wetland_pairs['roll_diff_change'] = lai_roll_diffs

# %% 3.2 Merge the water level shift data with the LAI data

shift_data = shift_data.merge(
    wetland_pairs[['logged_id', 'reference_id', 'roll_diff_change']],
    how='left',
    left_on=['log_id','ref_id'],
    right_on=['logged_id', 'reference_id'],
)

# %% 3.3 Visualize LAI ~ Depth change without factoring connectivity

plot_df = shift_data[
    (shift_data['model_type'] == model_type) & 
    (shift_data['data_set'] == data_set)
].copy()


slope, intercept, r_value, p_value, std_err = stats.linregress(
    plot_df['roll_diff_change'], 
    plot_df['mean_depth_change']
)

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    plot_df['roll_diff_change'], 
    plot_df['mean_depth_change'],
    alpha=0.6, 
    s=80, 
    edgecolors='black', 
    linewidth=0.5,
    color='steelblue'
)

# Add text labels for each point
# for idx, row in plot_df.iterrows():
#     ax.text(row['roll_diff_change'], row['mean_depth_change'], 
#             str(row['log_id']), fontsize=8, alpha=0.7)

x_range = np.linspace(plot_df['roll_diff_change'].min(), plot_df['roll_diff_change'].max(), 100)
y_pred = slope * x_range + intercept
ax.plot(x_range, y_pred, 'r--', linewidth=2, 
        label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.3f}')

ax.axhline(0, color='gray', linestyle=':', alpha=1)
ax.axhline(plot_df['mean_depth_change'].mean(), color='blue', linestyle=':', linewidth=2, alpha=0.8, label=f'Mean Stage Increase {plot_df['mean_depth_change'].mean():.2f} [m]')

ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=12)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=12)
ax.set_title(f'LAI Loss vs Wetland Depth Change (n={len(strong_pairs)} pairs)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)


plt.tight_layout()
plt.show()

# %% 4.0 LAI biplot factored by connectivity

connectivity_key = pd.read_excel(connectivity_key_path)

plot_df['log_connected'] = plot_df.apply(
    lambda row: connectivity_key.loc[
        connectivity_key['wetland_id'] == row['log_id'], 'connectivity'
    ].values[0],
    axis=1
)

print(plot_df['log_connected'].value_counts())

# %% 4.1 Make the connectivity biplot

connectivity_config = {
    'first order': {'color': 'green', 'label': '1st Order Ditched', 'marker': 's'},
    'giw': {'color': 'blue', 'label': 'GIW', 'marker': '^'}, 
    'flow-through': {'color': 'red', 'label': 'flow-through', 'marker': 'X'}
}

fig, ax = plt.subplots(figsize=(10, 8))

for connectivity_level, config in connectivity_config.items():
    subset = plot_df[plot_df['log_connected'] == connectivity_level].copy()

    subset_clean = subset.dropna(subset=['roll_diff_change', 'mean_depth_change', 'log_id'])

    ax.scatter(
        subset_clean['roll_diff_change'],
        subset_clean['mean_depth_change'],
        alpha=0.6,
        s=32,
        edgecolors='black',
        linewidth=0.5,
        color=config['color'],
        marker=config['marker']
    )

    if connectivity_level == 'flow-through':
        for _, row in subset_clean.iterrows():
            ax.annotate(
                str(row['log_id']),
                xy=(row['roll_diff_change'], row['mean_depth_change']),
                xytext=(6, 4),
                textcoords='offset points',
                fontsize=9,
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6)
            )

    # only compute / plot regression if there are >=2 valid points
    if len(subset_clean) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            subset_clean['roll_diff_change'],
            subset_clean['mean_depth_change']
        )

        x_min = subset_clean['roll_diff_change'].min()
        x_max = subset_clean['roll_diff_change'].max()
        if x_min == x_max:
            x_range = np.linspace(x_min - 0.1, x_max + 0.1, 100)
        else:
            x_range = np.linspace(x_min, x_max, 100)

        y_pred = slope * x_range + intercept

        ax.plot(x_range, y_pred, '--', linewidth=2, color=config['color'],
                label=f"{config['label']}: slope={slope:.3f}, R²={r_value**2:.3f}")

# Formatting
ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=18)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=18)
ax.set_title(f'LAI Loss vs Wetland Depth Change by Connectivity', 
            fontsize=20, fontweight='bold')
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='best', fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% 5.2 Print the connectivity stats

print("\n" + "="*70)
print("STATISTICS BY CONNECTIVITY")
print("="*70)

for connectivity_level in sorted(connectivity_config.keys()):
    subset = plot_df[plot_df['log_connected'] == connectivity_level]
    

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        subset['roll_diff_change'], 
        subset['mean_depth_change']
    )
        
    print(f"\nConnectivity Level: {connectivity_config[connectivity_level]['label']}")
    print(f"  N = {len(subset)}")
    print(f"  Slope = {slope:.4f}")
    print(f"  Intercept = {intercept:.4f}")
    print(f"  R² = {r_value**2:.3f}")
    print(f"  p-value = {p_value:.4f}")

print("="*70)

# %% 7.0 Linear regression lines by connectivity with confidence intervals

from scipy.stats import t

fig, ax = plt.subplots(figsize=(10, 8))

connectivity_levels = ['giw', 'first order', 'flow-through'] 
connectivity_labels = {'giw': 'GIW', 'first order': '1st Order Ditched', 'flow-through': 'flow-through'}
connectivity_colors = {'giw': 'green', 'first order': 'navy', 'flow-through': 'red'}

for conn_level in connectivity_levels:
    subset = plot_df[plot_df['log_connected'] == conn_level]
    filter_ids = ['7_341', '9_77']
    subset = subset[~subset['log_id'].isin(filter_ids)]
    x = subset['roll_diff_change'].values
    y = subset['mean_depth_change'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_pred = slope * x_range + intercept
    
    n = len(x)
    dof = n - 2 
    t_val = t.ppf(0.975, dof) 

    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)
    residuals = y - (slope * x + intercept)
    s_res = np.sqrt(np.sum(residuals**2) / dof)
    
    se = s_res * np.sqrt(1/n + (x_range - x_mean)**2 / sxx)
    ci = t_val * se
    
    ax.plot(x_range, y_pred, '--', linewidth=2, 
            color=connectivity_colors[conn_level],
            label=f"{connectivity_labels[conn_level]}: slope={slope:.3f}, R²={r_value**2:.3f}")
    
    # Plot confidence interval
    ax.fill_between(x_range, y_pred - ci, y_pred + ci, 
                    alpha=0.2, color=connectivity_colors[conn_level])

ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=18)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=18)
ax.set_title(f'LAI Loss vs Wetland Depth Change by Connectivity', 
            fontsize=20, fontweight='bold')
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='best', fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% 8.0 Simple histogram
