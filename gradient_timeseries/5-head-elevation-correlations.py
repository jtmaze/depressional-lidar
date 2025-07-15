# %%

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradient_timeseries import gradient_viz

os.chdir('/Users/jmaze/Documents/projects/depressional_lidar/')

bc_path = f'./delmarva/out_data/bc_gradient_timeseries.csv'
bc = pd.read_csv(bc_path)
jl_path = f'./delmarva/out_data/jl_gradient_timeseries.csv'
jl = pd.read_csv(jl_path)

gradient_ts = pd.concat([bc, jl], ignore_index=True)

gradient_ts['Date'] = pd.to_datetime(gradient_ts['Date'])

# %%

def cv(x):
    return x.std() / x.mean()

head_gradient_site_summary = gradient_ts.groupby(['well_pair']).agg(
    pair_type=('pair_type', 'first'),
    mean_head_gradient=('head_gradient_cm_m', 'mean'), 
    std_head_gradient=('head_gradient_cm_m', 'std'), 
    cv_head_gradient=('head_gradient_cm_m', lambda x: cv(x)),
    mean_adj_gradient=('adj_gradient', 'mean'),
    elevation_gradient=('elevation_gradient_cm_m', 'first'),
).reset_index()


# %%

gradient_viz.summary_correlations_plot(
    summary_df=head_gradient_site_summary,
    pair_type_filter=['CH__to__CH']
)

# %%

gradient_viz.summary_correlations_plot(
    summary_df=head_gradient_site_summary,
    pair_type_filter=['SW__to__SW']
)

# %%

gradient_viz.summary_correlations_plot(
    summary_df=head_gradient_site_summary,
    pair_type_filter=['UW__to__UW']
)

# %%

gradient_viz.summary_correlations_plot(
    summary_df=head_gradient_site_summary,
    pair_type_filter=['CH__to__SW', 'SW__to__CH']
)

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
