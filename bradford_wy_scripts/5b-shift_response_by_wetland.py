# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

lai_buffer_dist = 150
data_dir = "D:/depressional_lidar/data/bradford/"
shift_path = data_dir + f'/out_data/modeled_logging_stages/all_wells_shift_results_LAI_{lai_buffer_dist}m.csv'
connectivity_key_path = data_dir + '/bradford_wetland_connect_key.xlsx'
strong_pairs_path = data_dir + f'out_data/strong_ols_models_{lai_buffer_dist}m_all_wells.csv'

shift_data = pd.read_csv(shift_path)
connect_data = pd.read_excel(connectivity_key_path)
strong_pairs = pd.read_csv(strong_pairs_path)

# Add connectivity class info
shift_data = shift_data.merge(
    connect_data[['well_id', 'connectivity']],
    left_on='log_id',
    right_on='well_id',
    how='left'
).rename(columns={'connectivity': 'logged_connect'})
shift_data = shift_data.merge(
    connect_data[['well_id', 'connectivity']],
    left_on='ref_id',
    right_on='well_id',
    how='left'
).rename(columns={'connectivity': 'ref_connect'})

# Filter to only strong model
shift_data = shift_data.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id', 'logging_date'],
    right_on=['log_id', 'ref_id', 'log_date'],
    how='inner'
)
plot_data = shift_data[
    (shift_data['model_type'] == 'ols') &
    (shift_data['data_set'] == 'full')
].copy()

# Define connectivity color palette
connectivity_config = {
    'flow-through': {'color': 'red', 'label': 'Flow-through'},
    'first order': {'color': 'orange', 'label': '1st Order Ditched'},
    'giw': {'color': 'blue', 'label': 'GIW'}
}

print(shift_data.head())

# %% 2.0 Boxplot showing depth shifts by log_id

# Sort plot_data by connectivity to cluster connectivity classes together
connectivity_order = ['flow-through', 'first order', 'giw']
plot_data['connectivity_cat'] = pd.Categorical(
    plot_data['logged_connect'], 
    categories=connectivity_order, 
    ordered=True
)
plot_data_sorted = plot_data.sort_values('connectivity_cat')

unique_log_ids = plot_data_sorted.drop_duplicates(subset='log_id').sort_values('connectivity_cat')['log_id'].values

fig, ax = plt.subplots(figsize=(10, 5))

groups = [plot_data.loc[plot_data["log_id"] == id, "mean_depth_change"] 
          for id in unique_log_ids]

bp = ax.boxplot(groups, tick_labels=unique_log_ids, patch_artist=True)

for i, log_id in enumerate(unique_log_ids):
    connectivity = plot_data.loc[plot_data["log_id"] == log_id, "logged_connect"].iloc[0]
    color = connectivity_config[connectivity]['color']
    bp['boxes'][i].set_facecolor(color)
    bp['boxes'][i].set_alpha(0.6)

for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

ax.set_ylabel("Depth Change (m)")
ax.set_title("Depth Change by Logged Wetland")
plt.xticks(rotation=90)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

legend_elements = [Patch(facecolor=connectivity_config[conn]['color'], 
                         label=connectivity_config[conn]['label'], 
                         alpha=0.6) 
                   for conn in connectivity_order]
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.show()

# %% 3.0 Boxplot showing depth_shifts by ref_id

# Sort plot_data by reference connectivity to cluster connectivity classes together
plot_data['ref_connectivity_cat'] = pd.Categorical(
    plot_data['ref_connect'], 
    categories=connectivity_order, 
    ordered=True
)
plot_data_sorted_ref = plot_data.sort_values('ref_connectivity_cat')

# Get ref_ids in sorted order
unique_ref_ids = plot_data_sorted_ref.drop_duplicates(subset='ref_id').sort_values('ref_connectivity_cat')['ref_id'].values

fig, ax = plt.subplots(figsize=(10, 5))

groups = [plot_data.loc[plot_data['ref_id'] == id, "mean_depth_change"]
          for id in unique_ref_ids]

bp = ax.boxplot(groups, tick_labels=unique_ref_ids, patch_artist=True)

for i, ref_id in enumerate(unique_ref_ids):
    connectivity = plot_data.loc[plot_data["ref_id"] == ref_id, "ref_connect"].iloc[0]
    color = connectivity_config[connectivity]['color']
    bp['boxes'][i].set_facecolor(color)
    bp['boxes'][i].set_alpha(0.6)

for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

ax.set_ylabel("Depth Change (m)")
ax.set_title("Depth Change by Reference Wetland")
plt.xticks(rotation=90)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

legend_elements = [Patch(facecolor=connectivity_config[conn]['color'], 
                         label=connectivity_config[conn]['label'], 
                         alpha=0.6) 
                   for conn in connectivity_order]
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.show()


# %% 4.0 Three-series boxplot showing depth shifts by log_id's connectivity class

fig, ax = plt.subplots(figsize=(6, 5))

# Create boxplot data grouped by connectivity
groups = [plot_data.loc[plot_data['logged_connect'] == conn, "mean_depth_change"] 
          for conn in connectivity_order]

bp = ax.boxplot(groups, tick_labels=[connectivity_config[conn]['label'] for conn in connectivity_order], 
                patch_artist=True)

# Color boxes by connectivity
for i, conn in enumerate(connectivity_order):
    color = connectivity_config[conn]['color']
    bp['boxes'][i].set_facecolor(color)
    bp['boxes'][i].set_alpha(0.6)

for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

ax.set_ylabel("Depth Change (m)")
ax.set_title("Aggregate Depth Change by Logged Wetland Connectivity")
plt.xticks(rotation=0)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()

# %%
