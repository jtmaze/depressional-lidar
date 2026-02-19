# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

lai_buffer_dist = 150
data_set = 'no_dry_days'
data_dir = "D:/depressional_lidar/data/bradford/"
shift_path = data_dir + f'/out_data/modeled_logging_stages/shift_results_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
connectivity_key_path = data_dir + '/bradford_wetland_connect_logging_key.xlsx'
strong_pairs_path = data_dir + f'out_data/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv'

shift_data = pd.read_csv(shift_path)
connect_data = pd.read_excel(connectivity_key_path)
strong_pairs = pd.read_csv(strong_pairs_path)

# %% 2.1 Merge connectivity data and select for strong models

# Add connectivity class info
shift_data = shift_data.merge(
    connect_data[['wetland_id', 'connectivity']],
    left_on='log_id',
    right_on='wetland_id',
    how='left'
).rename(columns={'connectivity': 'logged_connect'}).drop(columns=['wetland_id'])

shift_data = shift_data.merge(
    connect_data[['wetland_id', 'connectivity']],
    left_on='ref_id',
    right_on='wetland_id',
    how='left'
).rename(columns={'connectivity': 'ref_connect'}).drop(columns=['wetland_id'])

# Filter to only strong model
shift_data = shift_data.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id', 'logging_date'],
    right_on=['log_id', 'ref_id', 'log_date'],
    how='inner'
)
plot_data = shift_data[
    (shift_data['model_type'] == 'OLS') &
    (shift_data['data_set'] == data_set)
].copy()

# %% 2.2 Generate a new column that specifies both log and ref connectivity status

def classify_pair(df_row):
    """
    Assigns all nine possible connectivity pairings between logged and reference wetlands
    """

    if (df_row['logged_connect'] == 'giw') & (df_row['ref_connect'] == 'giw'): #1
        return 'giw-giw'
    elif (df_row['logged_connect'] == 'flow-through') & (df_row['ref_connect'] == 'flow-through'): #2
        return 'flow-flow'
    elif (df_row['logged_connect'] == 'first order') & (df_row['ref_connect'] == 'first order'): #3
        return 'first-first'
    elif (df_row['logged_connect'] == 'giw') & (df_row['ref_connect'] == 'flow-through'): #4 
        return 'giw-flow'
    elif (df_row['logged_connect'] == 'flow-through') & (df_row['ref_connect'] == 'giw'): #5
        return 'flow-giw'
    elif (df_row['logged_connect'] == 'giw') & (df_row['ref_connect'] == 'first order'): #6
        return 'giw-first'
    elif (df_row['logged_connect'] == 'first order') & (df_row['ref_connect'] == 'giw'): #7
        return 'first-giw'
    elif (df_row['logged_connect'] == 'flow-through') & (df_row['ref_connect'] == 'first order'): #8
        return 'flow-first'
    elif (df_row['logged_connect'] == 'first order') & (df_row['ref_connect'] == 'flow-through'): #9
        return 'first-flow'
    else:
        raise ValueError("Unrecognized connectivity pair")
    
plot_data['pair_connect'] = plot_data.apply(classify_pair, axis=1)

print(plot_data.head(2))

# Define connectivity color palette
connectivity_config = {
    'flow-through': {'color': 'red', 'label': 'Flow-through'},
    'first order': {'color': 'green', 'label': '1st Order Ditched'},
    'giw': {'color': 'blue', 'label': 'GIW'}
}

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

# %% 5.0 Three-series boxplot showing depth shifts by ref_id's connectivity class

fig, ax = plt.subplots(figsize=(6, 5))

# Create boxplot data grouped by connectivity
groups = [plot_data.loc[plot_data['ref_connect'] == conn, "mean_depth_change"] 
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
ax.set_title("Aggregate Depth Change by Reference Wetland Connectivity")
plt.xticks(rotation=0)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()

# %% 6.0 Interaction plot


# %% 7.0 Heat map grid with t-stats by class