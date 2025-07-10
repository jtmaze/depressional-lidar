# %%
import os
import sys
import pandas as pd

import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradient_timeseries import gradient_ts_viz

os.chdir('/Users/jmaze/Documents/projects/depressional_lidar/')
catchment = 'jl'
gradients_path = f'./delmarva/out_data/{catchment}_gradient_timeseries.csv'

# %% 
gradient_ts = pd.read_csv(gradients_path)
gradient_ts['Date'] = pd.to_datetime(gradient_ts['Date'])

# %%

well_pairs_to_plot = gradient_ts['well_pair'].unique()

gradient_ts_viz.gradient_ts_plot(
    gradient_ts, 
    well_pairs_to_plot, 
    y_var='head_gradient_cm_m', 
    abs_vals=False, 
    y_lim=None)

# %% 

max_gradients = gradient_ts.groupby('well_pair')['head_gradient_cm_m'].max()
high_gradient_pairs = max_gradients[max_gradients >= 3].index.tolist()

high_gradient_pairs = high_gradient_pairs + [
    "ND-UW1__to__ND-SW",
    "DK-UW2__to__DK-SW",
    "DK-UW1__to__DK-SW",
    "ND-UW2__to__ND-SW",
    "TS-UW1__to__TS-SW",
    "DK-CH__to__DK-SW",
    "BD-CH__to__BD-SW",
    "TS-SW__to__BD-CH",
    "DK-CH__to__DK-UW2",
    "TS-CH__to__DK-SW"
]

#high_gradient_pairs = high_gradient_pairs + ['XB-SW__to__HB-CH']

gradient_ts_viz.gradient_ts_plot(
    gradient_ts, 
    high_gradient_pairs, 
    y_var='head_gradient_cm_m', 
    abs_vals=True, 
    y_lim=None)


# %%

well_pairs_to_plot = list(gradient_ts['well_pair'].unique())


gradient_ts_viz.gradient_ts_plot(
    gradient_ts, 
    well_pairs_to_plot, 
    y_var='adj_gradient', 
    abs_vals=True, 
    y_lim=None)


# %%

max_gradients = gradient_ts.groupby('well_pair')['adj_gradient'].max()
high_gradient_pairs = max_gradients[max_gradients >= 2].index.tolist()

#high_gradient_pairs = high_gradient_pairs + ['XB-SW__to__HB-CH']

gradient_ts_viz.gradient_ts_plot(
    gradient_ts, 
    high_gradient_pairs, 
    y_var='adj_gradient', 
    abs_vals=True, 
    y_lim=None)

# %%
def cv(x):
    return x.std() / x.mean()

head_gradient_site_summary = gradient_ts.groupby(['well_pair']).agg(
    mean_head_gradient=('head_gradient_cm_m', 'mean'), 
    std_head_gradient=('head_gradient_cm_m', 'std'), 
    cv_head_gradient=('head_gradient_cm_m', lambda x: cv(x)),
    mean_adj_gradient=('adj_gradient', 'mean'),
    elevation_gradient=('elevation_gradient_cm_m', 'first'),
).reset_index()

# %%

# Create a figure with four subplots in a 2x2 grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Histogram for mean_head_gradient
ax1.hist(head_gradient_site_summary['mean_head_gradient'].abs(), bins=15, 
         color='blue', alpha=0.7, edgecolor='black')
ax1.set_title('Distribution of Mean Head Gradient', fontsize=14)
ax1.set_xlabel('Mean Head Gradient (cm/m)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
# Add mean and median lines
mean_head = head_gradient_site_summary['mean_head_gradient'].abs().mean()
median_head = head_gradient_site_summary['mean_head_gradient'].abs().median()
ax1.axvline(mean_head, color='red', linestyle='dashed', linewidth=3.5, label=f'Mean: {mean_head:.2f}')
ax1.axvline(median_head, color='orange', linestyle='dashed', linewidth=3.5, label=f'Median: {median_head:.2f}')
ax1.legend()

# Histogram for mean_adj_gradient
ax2.hist(head_gradient_site_summary['mean_adj_gradient'].abs(), bins=15, 
         color='blue', alpha=0.7, edgecolor='black')
ax2.set_title('Distribution of Mean Adjusted Gradient', fontsize=14)
ax2.set_xlabel('Mean Adjusted Gradient', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
# Add mean and median lines
mean_adj = head_gradient_site_summary['mean_adj_gradient'].abs().mean()
median_adj = head_gradient_site_summary['mean_adj_gradient'].abs().median()
ax2.axvline(mean_adj, color='red', linestyle='dashed', linewidth=3.5, label=f'Mean: {mean_adj:.2f}')
ax2.axvline(median_adj, color='orange', linestyle='dashed', linewidth=3.5, label=f'Median: {median_adj:.2f}')
ax2.legend()

# Histogram for std_head_gradient
ax3.hist(head_gradient_site_summary['std_head_gradient'], bins=15, 
         color='blue', alpha=0.7, edgecolor='black')
ax3.set_title('Distribution of Head Gradient Standard Deviation', fontsize=14)
ax3.set_xlabel('Head Gradient Std Dev (cm/m)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
# Add mean and median lines
mean_std = head_gradient_site_summary['std_head_gradient'].mean()
median_std = head_gradient_site_summary['std_head_gradient'].median()
ax3.axvline(mean_std, color='red', linestyle='dashed', linewidth=3.5, label=f'Mean: {mean_std:.2f}')
ax3.axvline(median_std, color='orange', linestyle='dashed', linewidth=3.5, label=f'Median: {median_std:.2f}')
ax3.legend()

# Histogram for cv_head_gradient
ax4.hist(head_gradient_site_summary['cv_head_gradient'], bins=15, 
         color='blue', alpha=0.7, edgecolor='black')
ax4.set_title('Distribution of Head Gradient Coefficient of Variation', fontsize=14)
ax4.set_xlabel('Head Gradient CV', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
# Add mean and median lines
mean_cv = head_gradient_site_summary['cv_head_gradient'].mean()
median_cv = head_gradient_site_summary['cv_head_gradient'].median()
ax4.axvline(mean_cv, color='red', linestyle='dashed', linewidth=3.5, label=f'Mean: {mean_cv:.2f}')
ax4.axvline(median_cv, color='orange', linestyle='dashed', linewidth=3.5, label=f'Median: {median_cv:.2f}')
ax4.legend()

plt.tight_layout()
plt.show()

# %%

# Remove any rows with NaN values for clean correlation
clean_data = head_gradient_site_summary.dropna(subset=['elevation_gradient', 'mean_head_gradient'])

# Take absolute values of gradients if needed
# Uncomment the line below if you want to use absolute values
# clean_data['mean_head_gradient'] = clean_data['mean_head_gradient'].abs()

# Calculate Pearson correlation (linear relationship)
pearson_corr, p_value = stats.pearsonr(
    clean_data['elevation_gradient'], 
    clean_data['mean_head_gradient']
)

# Calculate Spearman rank correlation (monotonic relationship)
spearman_corr, sp_p_value = stats.spearmanr(
    clean_data['elevation_gradient'],
    clean_data['mean_head_gradient']
)

# Create a scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='elevation_gradient', y='mean_head_gradient', data=clean_data, 
           scatter_kws={'alpha':0.6}, line_kws={'color':'red'})

plt.title(f'Elevation Gradient vs. Head Gradient\nPearson r: {pearson_corr:.3f} (p={p_value:.4f})\n' +
          f'Spearman ρ: {spearman_corr:.3f} (p={sp_p_value:.4f})', 
          fontsize=14)
plt.xlabel('Elevation Gradient (cm/m)', fontsize=12)
plt.ylabel('Mean Head Gradient (cm/m)', fontsize=12)
plt.grid(alpha=0.3)

# Add a horizontal line at y=0
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print(f"Pearson correlation coefficient: {pearson_corr:.3f}")
print(f"Pearson p-value: {p_value:.4f}")
print(f"Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
print("\n")
print(f"Spearman rank correlation coefficient: {spearman_corr:.3f}")
print(f"Spearman p-value: {sp_p_value:.4f}")
print(f"Interpretation: {'Significant' if sp_p_value < 0.05 else 'Not significant'} at α=0.05")



#%%

"""
Scratch function -- actually pretty useful for plotting
"""

def plot_well_timeseries(well_name, data):
    test_plot = data[data['Site_Name'] == well_name]
    
    # Define a custom color map for flags
    flag_colors = {0: 'blue', 1: 'pink', 2: 'green', 3: 'orange', 4: 'red'}
    colors = [flag_colors.get(flag, 'gray') for flag in test_plot['Flag']]
    
    plt.figure(figsize=(12,6))
    plt.scatter(test_plot['Date'], 
                test_plot['waterLevel'],
                c=colors,
                alpha=0.7)
    

    plt.title(f'Water Level Time Series for {well_name}')
    plt.ylabel('Water Level')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    
    # Create a custom legend for the flag colors
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                      label=f'Flag {flag}', markerfacecolor=color, markersize=10)
                      for flag, color in flag_colors.items()]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.show()



# %%
