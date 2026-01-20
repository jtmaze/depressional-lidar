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

lai_buffer_dist = 150

data_dir = "D:/depressional_lidar/data/bradford/out_data/"
lai_dir = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/well_buffer_{lai_buffer_dist}m_nomasking/'

# Path to results
shift_path = data_dir + f'/modeled_logging_stages/shift_results_LAI_{lai_buffer_dist}m.csv'
distributions_path = data_dir + f'/modeled_logging_stages/hypothetical_distributions_LAI_{lai_buffer_dist}m.csv'
models_path = data_dir + f'/model_info/model_estimates_LAI_{lai_buffer_dist}m.csv'

wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_limited.csv'
strong_pairs = data_dir + f'strong_ols_models_{lai_buffer_dist}m.csv'

wetland_pairs = pd.read_csv(wetland_pairs_path)
strong_pairs = pd.read_csv(strong_pairs)
shift_data = pd.read_csv(shift_path)

# NOTE: Filters shift data where model fits were above 0.35
shift_data = shift_data.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id', 'logging_date'],
    right_on=['log_id', 'ref_id', 'log_date'],
    how='inner'
)

print(len(shift_data))

# %% 2.0 Show pre vs. post logging distributions for all models and datasets

# models = ['huber', 'ols']
# datasets = ['full', 'above_ground', 'above_-0.2']

# # Create subplots
# fig, axes = plt.subplots(len(models), len(datasets), figsize=(14, 10), sharex=True)
# summary_data = []

# for i, m in enumerate(models):
#     for j, d in enumerate(datasets):
#         ax = axes[i, j]
#         plot_df = shift_data[(shift_data['model_type'] == m) & 
#                              (shift_data['data_set'] == d)]
        
#         # Calculate statistics
#         mean_change = plot_df['mean_depth_change'].mean()
#         median_change = plot_df['mean_depth_change'].median()
#         std_change = plot_df['mean_depth_change'].std()
        
#         # Store for summary table
#         summary_data.append({
#             'Model': m,
#             'Dataset': d,
#             'Mean': mean_change,
#             'Median': median_change,
#             'Std': std_change
#         })
        
#         # Create histogram
#         ax.hist(
#             plot_df['mean_depth_change'], 
#             bins=20, 
#             edgecolor='black', 
#             alpha=0.7, 
#             color='steelblue',
#             linewidth=1.2
#         )
        
#         ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
#         ax.axvline(mean_change, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
#         ax.axvline(median_change, color='orange', linestyle='--', linewidth=2, alpha=0.8)
        
#         ax.set_title(f'{m.upper()} - {d.replace("_", " ").title()}', fontsize=14, fontweight='bold')
#         ax.set_ylabel('Count', fontsize=10)
#         ax.grid(True, alpha=0.3, axis='y')
        
#         # Add stats text
#         stats_text = f'Mean = {mean_change:.3f}m\nStd = {std_change:.3f}m'
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
#         ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
#                 verticalalignment='top', bbox=props)

# # Set x-label only on bottom row
# for j in range(len(datasets)):
#     axes[-1, j].set_xlabel('Mean Depth Change (Post - Pre Logging) [m]', fontsize=11)

# plt.tight_layout()
# plt.show()

# # Print summary table
# summary_df = pd.DataFrame(summary_data)
# print("\nSummary Statistics:")
# print(summary_df.to_string(index=False))

# %% 3.0 Only plot shift data for a single model and dataset

plot_df = shift_data[
    (shift_data['model_type'] == 'ols') & (shift_data['data_set'] == 'full')
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


# %% 4.0 LAI change biplot

# %% 4.1 Calculate the LAI differences for every pair

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


# %% 4.2 Merge the water level shift data with the LAI data

shift_data = shift_data.merge(
    wetland_pairs[['logged_id', 'reference_id', 'roll_diff_change']],
    how='left',
    left_on=['log_id','ref_id'],
    right_on=['logged_id', 'reference_id'],
)

# %% 4.3 Visualize LAI ~ Depth change without factoring connectivity

model_type = 'ols'  # or 'ols'
data_set = 'full'  # or 'full'

plot_df = shift_data[
    (shift_data['model_type'] == model_type) & 
    (shift_data['data_set'] == data_set)
].copy()

#plot_df = plot_df[plot_df['mean_depth_change'] >= -0.15]

print(plot_df)

# %%
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
        label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.3f}')

ax.axhline(0, color='gray', linestyle=':', alpha=1)

ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=12)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=12)
ax.set_title(f'LAI Loss vs Wetland Depth Change', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)


plt.tight_layout()
plt.show()


# %% 5.0 LAI biplot factored by connectivity

connected_dict = {
    '15_268': 2,
    '15_516': 0, # NOTE Omitted for bad well data
    '14_418': 0,
    '14_500': 2,
    '14_610': 2,
    '5_597': 2,
    '5_510': 2,
    '9_439': 0,
    '9_508': 0,
    '13_263': 0,
    '13_271': 0,
    '6a_17': 2,
    '6_300': 0,
    '7_626': 2,
    '7_243': 2,
    '3_311': 0,
    '3_173': 0,
    '3_244': 0, # NOTE Omitted for bad well data
    '6_93': 0,
    # Wetlands with inflows
    #'14.9_601': 2,
    #'9_77': 2,
    #'3_23': 2,
    #'13_267': 1,
}

plot_df['connectivity'] = plot_df['log_id'].map(connected_dict)

# %% 5.1 Make the connectivity biplot

connectivity_config = {
    0: {'color': 'red', 'label': 'Unconnected', 'marker': 'o'},
    #1: {'color': 'orange', 'label': 'Uncertian Connected', 'marker': 's'},
    2: {'color': 'blue', 'label': 'Connected', 'marker': '^'}
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

# %% 5.2 Print the connectivity stats

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

# %% 6.0 Make a boxplot with connectivity

lai_change_map = {
    (0, 1.25): 'low',
    (1.25, np.inf): 'high'
}

# Apply the mapping to the roll_diff_change column
def categorize_lai_change(value):
    for (lower, upper), label in lai_change_map.items():
        if lower <= value < upper:
            return label
    return None

plot_df['lai_change_category'] = plot_df['roll_diff_change'].apply(categorize_lai_change)

print("\nCategory Counts:")
print(plot_df['lai_change_category'].value_counts())

# %% 6.1 Boxplot of depth change by LAI category and connectivity

# Define category order for plotting
category_order = ['low', 'high']

# Prepare data for boxplot
connectivity_levels = [0, 2]
connectivity_labels = {0: 'Unconnected', 2: 'Connected'}
connectivity_colors = {0: 'red', 2: 'blue'}
category_labels = [
    'small LAI decrease\n(<1.1 units)',
    'large LAI decrease\n(>1.1 units)'
]

fig, ax = plt.subplots(figsize=(10, 6))

# Create positions for boxes
positions = []
box_data = []
colors = []

for i, category in enumerate(category_order):
    for j, conn_level in enumerate(connectivity_levels):
        subset = plot_df[
            (plot_df['lai_change_category'] == category) & 
            (plot_df['connectivity'] == conn_level)
        ]

# Create boxplot
bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                showmeans=False,
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=2, color='black'))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

ax.set_xticks([i * (len(connectivity_levels) + 1) + 0.5 for i in range(len(category_order))])
ax.set_xticklabels(category_labels, fontsize=14)

ax.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

ax.set_xlabel('LAI Change Categorical', fontsize=18)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=18)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=connectivity_colors[0], alpha=0.6, label='Unconnected'),
    Patch(facecolor=connectivity_colors[2], alpha=0.6, label='Strongly Connected')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=0.9)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS BY LAI CATEGORY AND CONNECTIVITY")
print("="*80)

for category in category_order:
    print(f"\n{category.upper()}:")
    for conn_level in connectivity_levels:
        subset = plot_df[
            (plot_df['lai_change_category'] == category) & 
            (plot_df['connectivity'] == conn_level)
        ]
        if len(subset) > 0:
            print(f"  {connectivity_labels[conn_level]}:")
            print(f"    N = {len(subset)}")
            print(f"    Mean depth change = {subset['mean_depth_change'].mean():.4f} m")
            print(f"    Median depth change = {subset['mean_depth_change'].median():.4f} m")
            print(f"    Std = {subset['mean_depth_change'].std():.4f} m")

print("="*80)

# %% 7.0 Linear regression lines by connectivity with confidence intervals

from scipy.stats import t

fig, ax = plt.subplots(figsize=(10, 6))

connectivity_levels = [0, 2]
connectivity_labels = {0: 'Unconnected', 2: 'Connected'}
connectivity_colors = {0: 'red', 2: 'blue'}

for conn_level in connectivity_levels:
    subset = plot_df[plot_df['connectivity'] == conn_level]

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
    
    ax.plot(x_range, y_pred, '-', linewidth=3, 
            color=connectivity_colors[conn_level],
            label=f"{connectivity_labels[conn_level]}: slope={slope:.2f}, R²={r_value**2:.2f}, p={p_value:.4f}")
    
    # Plot confidence interval
    ax.fill_between(x_range, y_pred - ci, y_pred + ci, 
                    alpha=0.2, color=connectivity_colors[conn_level])

ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=18)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=18)
ax.tick_params(axis='both', labelsize=14)

ax.legend(loc='best', fontsize=14, framealpha=0.9)
plt.tight_layout()
plt.show()

# %%
