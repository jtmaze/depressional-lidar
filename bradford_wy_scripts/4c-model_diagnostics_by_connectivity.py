# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lai_buffer_dist = 150
data_set = 'no_dry_days'

data_dir = "D:/depressional_lidar/data/bradford/"
connect_path = data_dir + '/bradford_wetland_connect_logging_key.xlsx'
connect_data = pd.read_excel(connect_path)

model_path = f'{data_dir}/out_data/model_info/all_wells_model_estimates_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
model_df = pd.read_csv(model_path)


# %% 2.0 Merge connectivity to model data

model_df = model_df.merge(
    connect_data[['well_id', 'connectivity']],
    left_on='log_id',
    right_on='well_id',
    how='left'
).rename(columns={'connectivity': 'logged_connect'}).drop(columns=['well_id'])

model_df = model_df.merge(
    connect_data[['well_id', 'connectivity']],
    left_on='ref_id',
    right_on='well_id',
    how='left'
).rename(columns={'connectivity': 'ref_connect'}).drop(columns=['well_id'])

print(model_df.columns)

# %% 3.0 Boxplots

connectivity_config = {
    'flow-through': {'color': 'red', 'label': 'Flow-through'},
    'first order': {'color': 'green', 'label': '1st Order Ditched'},
    'giw': {'color': 'blue', 'label': 'GIW'}
}

connect_types = list(connectivity_config.keys())

plt.rcParams['hatch.linewidth'] = 2.0
fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharex=True, sharey=True)

for row_i, log_conn in enumerate(connect_types):
    for col_j, ref_conn in enumerate(connect_types):
        ax = axes[row_i, col_j]
        
        subset = model_df[
            (model_df['logged_connect'] == log_conn) & 
            (model_df['ref_connect'] == ref_conn) &
            (model_df['model_type'] == 'OLS')
        ]['r2_joint'].dropna()
        
        log_color = connectivity_config[log_conn]['color']
        ref_color = connectivity_config[ref_conn]['color']
        
        if len(subset) > 0:
            bp = ax.boxplot(subset, patch_artist=True, widths=0.3)
            
            for patch in bp['boxes']:
                if log_conn == ref_conn:
                    patch.set_facecolor(log_color)
                    patch.set_edgecolor('black')
                    patch.set_alpha(0.5)
                else:
                    patch.set_facecolor(log_color)
                    patch.set_edgecolor(ref_color)
                    patch.set_hatch('+')
                    patch.set_linewidth(5)
                    patch.set_alpha(0.5)
            
            # Hide median lines
            for median in bp['medians']:
                median.set_visible(False)
                
            # Print mean on panel
            mean_val = subset.mean()
            ax.text(0.98, 0.95, f'Mean: {mean_val:.3f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=9, color='black', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
        
        if row_i == 0:
            ax.set_title(connectivity_config[ref_conn]['label'])
        if col_j == 0:
            ax.set_ylabel(connectivity_config[log_conn]['label'])

        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.4, color='black', linestyle=':', linewidth=2)

fig.supxlabel('Reference Connectivity')
fig.supylabel('Logged Connectivity')
plt.tight_layout()
plt.show()
# %%
