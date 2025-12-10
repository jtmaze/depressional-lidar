# %% 1.0 Libraries, function imports and filepaths
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bradford_wy_scripts.functions.lai_vis_functions import read_concatonate_lai

data_dir = "D:/depressional_lidar/data/bradford/out_data/"
lai_dir = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/well_buffer_250m_includes_wetlands/'
shift_path = data_dir + 'logging_hypothetical_shift_results.csv'
distributions_path = data_dir + 'logging_hypothetical_distributions.csv'
models_path = data_dir + 'pre_post_models.csv'
wetland_pairs_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs.csv'
strong_pairs = data_dir + 'strong_ols_models.csv'

wetland_pairs = pd.read_csv(wetland_pairs_path)
strong_pairs = pd.read_csv(strong_pairs)
shift_data = pd.read_csv(shift_path)

# NOTE: Filters shift data where model fits were above 0.35
# shift_data = shift_data.merge(
#     strong_pairs[['log_id', 'ref_id', 'log_date']],
#     left_on=['log_id', 'ref_id', 'logging_date'],
#     right_on=['log_id', 'ref_id', 'log_date'],
#     how='inner'
# )
shift_data = shift_data[shift_data['mean_depth_change'] >= -0.4]

print(len(shift_data))

# %% Show pre vs. post logging distributions for all models.

models = ['huber', 'ols']
datasets = ['full', 'above_ground', 'above_-0.2']

# Create subplots
fig, axes = plt.subplots(len(models), len(datasets), figsize=(14, 10), sharex=True)
summary_data = []

for i, m in enumerate(models):
    for j, d in enumerate(datasets):
        ax = axes[i, j]
        plot_df = shift_data[(shift_data['model_type'] == m) & 
                             (shift_data['data_set'] == d)]
        
        # Calculate statistics
        mean_change = plot_df['mean_depth_change'].mean()
        median_change = plot_df['mean_depth_change'].median()
        std_change = plot_df['mean_depth_change'].std()
        
        # Store for summary table
        summary_data.append({
            'Model': m,
            'Dataset': d,
            'Mean': mean_change,
            'Median': median_change,
            'Std': std_change
        })
        
        # Create histogram
        ax.hist(
            plot_df['mean_depth_change'], 
            bins=20, 
            edgecolor='black', 
            alpha=0.7, 
            color='steelblue',
            linewidth=1.2
        )
        
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(mean_change, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(median_change, color='orange', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_title(f'{m.upper()} - {d.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add stats text
        stats_text = f'Mean = {mean_change:.3f}m\nStd = {std_change:.3f}m'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

# Set x-label only on bottom row
for j in range(len(datasets)):
    axes[-1, j].set_xlabel('Mean Depth Change (Post - Pre Logging) [m]', fontsize=11)

plt.tight_layout()
plt.show()

# Print summary table
summary_df = pd.DataFrame(summary_data)
print("\nSummary Statistics:")
print(summary_df.to_string(index=False))

# %% Only plot shift data for a single model and dataset

plot_df = shift_data[
    (shift_data['model_type'] == 'ols') & (shift_data['data_set'] == 'full')
]

mean_change = plot_df['mean_depth_change'].mean()
median_change = plot_df['mean_depth_change'].median()
std_change = plot_df['mean_depth_change'].std()


fig, ax = plt.subplots(figsize=(12, 5))

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


# %% LAI change biplot

lai_diffs = []
for idx, p in wetland_pairs.iterrows():

    date = p['logging_date']
    log_id = p['logged_id']
    logged_lai = read_concatonate_lai(
        lai_dir,
        log_id,
        "include_wetlands_250m",
        upper_bound=5.5,
        lower_bound=0.5
    )
    logged_lai = logged_lai[logged_lai['date'] >= '2019-01-01']
    # log_pre_lai = logged_lai[logged_lai['date'] <= date]
    # log_post_lai = logged_lai[logged_lai['date'] > date]


    ref_id = p['reference_id']
    reference_lai = read_concatonate_lai(
        lai_dir, 
        ref_id,
        "includes_wetlands_250m",
        upper_bound=5.5,
        lower_bound=0.5
    )
    reference_lai = reference_lai[reference_lai['date'] >= '2019-01-01']
    # ref_pre_lai = reference_lai[reference_lai['date'] <= date]
    # ref_post_lai = reference_lai[reference_lai['date'] > date]

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

    lai_diffs.append(roll_diff_change)

wetland_pairs['roll_diff_change'] = lai_diffs


# %%

shift_data = shift_data.merge(
    wetland_pairs[['logged_id', 'reference_id', 'roll_diff_change']],
    how='left',
    left_on=['log_id','ref_id'],
    right_on=['logged_id', 'reference_id'],
)

# %%

model_type = 'ols'  # or 'ols'
data_set = 'full'  # or 'full'

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

x_range = np.linspace(plot_df['roll_diff_change'].min(), plot_df['roll_diff_change'].max(), 100)
y_pred = slope * x_range + intercept
ax.plot(x_range, y_pred, 'r--', linewidth=2, 
        label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.3e}')

ax.axhline(0, color='gray', linestyle=':', alpha=1)

ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=12)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=12)
ax.set_title(f'LAI Loss vs Wetland Depth Change', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)


plt.tight_layout()
plt.show()


# %%

connected_dict = {
    '15_268': 2,
    '14_418': 0,
    '14_500': 2,
    '14_610': 1,
    '5_597': 1,
    '5_510': 2,
    '9_439': 0,
    '9_508': 0,
    '13_271': 0,
    '7_243': 1,
    '3_311': 0,
    '3_173': 0,
    '3_244': 0, 
    '14.9_601': 2,
    '9_77': 2,
    '3_23': 2,
    '13_267': 1,
}

# %%

plot_df['connectivity'] = plot_df['log_id'].map(connected_dict)

# %%

connectivity_config = {
    0: {'color': 'red', 'label': 'Unconnected', 'marker': 'o'},
    1: {'color': 'orange', 'label': 'Uncertian Connected', 'marker': 's'},
    2: {'color': 'blue', 'label': 'Stongly Connected', 'marker': '^'}
}

fig, ax = plt.subplots(figsize=(12, 8))

for connectivity_level, config in connectivity_config.items():
    subset = plot_df[plot_df['connectivity'] == connectivity_level]

    ax.scatter(
            subset['roll_diff_change'], 
            subset['mean_depth_change'],
            alpha=0.6, 
            s=32, 
            edgecolors='black', 
            linewidth=0.5,
            color=config['color'],
            marker=config['marker']
        )
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        subset['roll_diff_change'], 
        subset['mean_depth_change']
    )

    x_range = np.linspace(subset['roll_diff_change'].min(), 
                                 subset['roll_diff_change'].max(), 100)
    
    y_pred = slope * x_range + intercept
    
    ax.plot(x_range, y_pred, '--', linewidth=2, color=config['color'],
            label=f"{config['label']}")

# Formatting
ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=12)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=12)
ax.set_title(f'LAI Loss vs Wetland Depth Change by Connectivity', 
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%

print("\n" + "="*70)
print("STATISTICS BY CONNECTIVITY")
print("="*70)

for connectivity_level in sorted(connectivity_config.keys()):
    subset = plot_df[plot_df['connectivity'] == connectivity_level]
    

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        subset['roll_diff_change'], 
        subset['mean_depth_change']
    )
        
    print(f"\nConnectivity Level {connectivity_level} ({connectivity_config[connectivity_level]['label']}):")
    print(f"  N = {len(subset)}")
    print(f"  Slope = {slope:.4f}")
    print(f"  Intercept = {intercept:.4f}")
    print(f"  R² = {r_value**2:.3f}")
    print(f"  p-value = {p_value:.4f}")

print("="*70)


# %%




