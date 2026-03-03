# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lai_buffer_dist = 150
data_set = 'no_dry_days'

data_dir = "D:/depressional_lidar/data/bradford/"
connect_path = data_dir + '/bradford_wetland_connect_logging_key.xlsx'
connect_data = pd.read_excel(connect_path)

model_path = f'{data_dir}/out_data/model_info/model_estimates_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
model_df = pd.read_csv(model_path)


# %% 2.0 Merge connectivity to model data

model_df = model_df.merge(
    connect_data[['wetland_id', 'connectivity']],
    left_on='log_id',
    right_on='wetland_id',
    how='left'
).rename(columns={'connectivity': 'logged_connect'}).drop(columns=['wetland_id'])

model_df = model_df.merge(
    connect_data[['wetland_id', 'connectivity']],
    left_on='ref_id',
    right_on='wetland_id',
    how='left'
).rename(columns={'connectivity': 'ref_connect'}).drop(columns=['wetland_id'])

print(model_df.columns)

# %% 3.0 Interaction plot

connectivity_config = {
    'flow-through': {'color': 'red', 'label': 'Flow-through'},
    'first order': {'color': 'green', 'label': '1st Order Ditched'},
    'giw': {'color': 'blue', 'label': 'GIW'}
}

connect_types = list(connectivity_config.keys())

# Calculate mean and SEM for each combination
interaction_stats = model_df[model_df['model_type'] == 'OLS'].groupby(
    ['logged_connect', 'ref_connect']
)['r2_joint'].agg(['mean', 'sem', 'count']).reset_index()

fig, ax = plt.subplots(figsize=(6, 6))

# Set x-axis order: Flow-through, 1st Order Ditched, GIW
plot_connectivity_order = ['flow-through', 'first order', 'giw']
x_positions = {conn: i for i, conn in enumerate(plot_connectivity_order)}

for ref_conn in plot_connectivity_order:
    subset = interaction_stats[interaction_stats['ref_connect'] == ref_conn]
    
    x_vals = [x_positions[log_conn] for log_conn in subset['logged_connect']]
    y_vals = subset['mean'].values
    yerr_vals = subset['sem'].values
    
    sorted_indices = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
    x_sorted = [x_vals[i] for i in sorted_indices]
    y_sorted = [y_vals[i] for i in sorted_indices]
    yerr_sorted = [yerr_vals[i] for i in sorted_indices]
    
    color = connectivity_config[ref_conn]['color']
    label = connectivity_config[ref_conn]['label']
    
    ax.errorbar(x_sorted, y_sorted, yerr=yerr_sorted, 
                fmt='o-', color=color, label=f'Reference: {label}',
                capsize=4, capthick=1.5, markersize=8, linewidth=2, alpha=0.5)

# Calculate and plot marginal means for logged connectivity as black X's
logged_marginal_means = model_df[model_df['model_type'] == 'OLS'].groupby(
    'logged_connect'
)['r2_joint'].agg(['mean', 'sem']).reset_index()

for i, conn in enumerate(plot_connectivity_order):
    if conn in logged_marginal_means['logged_connect'].values:
        mean_val = logged_marginal_means[logged_marginal_means['logged_connect'] == conn]['mean'].iloc[0]
        ax.plot(i, mean_val, 'x', color='black', markersize=18, markeredgewidth=3,
               label='Logged Mean' if i == 0 else "")

ax.set_xticks(range(len(plot_connectivity_order)))
ax.set_xticklabels([connectivity_config[conn]['label'] for conn in plot_connectivity_order], fontsize=12)
ax.set_xlabel("Logged Wetland Connectivity", fontsize=18, labelpad=20)
ax.set_ylabel("Mean R² (Joint Model)", fontsize=14)
ax.tick_params(axis='y', labelsize=12)

#ax.axhline(y=0.3, color='black', linestyle='--', linewidth=2.5)
ax.legend(loc='best', fontsize=12, title_fontsize=14)

plt.ylim(0, 1)

plt.show()

# %%
