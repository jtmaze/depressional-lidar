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
).rename(columns={'connectivity': 'logged_connect'}).drop(columns=['well_id'])

shift_data = shift_data.merge(
    connect_data[['well_id', 'connectivity']],
    left_on='ref_id',
    right_on='well_id',
    how='left'
).rename(columns={'connectivity': 'ref_connect'}).drop(columns=['well_id'])

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
    'first order': {'color': 'orange', 'label': '1st Order Ditched'},
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


# %% 6.0 A nine-series boxplot showing depth shifts by connectivity pairings

# Define the order of connectivity pairings (logged-reference)
# pair_order = [
#     'giw-giw', 'giw-flow', 'giw-first',
#     'flow-giw', 'flow-flow', 'flow-first',
#     'first-giw', 'first-flow', 'first-first'
# ]

# # Create labels for the x-axis
# pair_labels = [
#     'GIW→GIW', 'GIW→Flow', 'GIW→1st',
#     'Flow→GIW', 'Flow→Flow', 'Flow→1st',
#     '1st→GIW', '1st→Flow', '1st→1st'
# ]

# # Color mapping based on the logged wetland's connectivity
# pair_colors = {
#     'giw-giw': 'blue', 'giw-flow': 'blue', 'giw-first': 'blue',
#     'flow-giw': 'red', 'flow-flow': 'red', 'flow-first': 'red',
#     'first-giw': 'orange', 'first-flow': 'orange', 'first-first': 'orange'
# }

# fig, ax = plt.subplots(figsize=(12, 6))

# # Create boxplot data grouped by connectivity pairing
# groups = [plot_data.loc[plot_data['pair_connect'] == pair, "mean_depth_change"] 
#           for pair in pair_order]

# # Filter out empty groups and track which pairs have data
# valid_pairs = [(pair, label, groups[i]) for i, (pair, label) in enumerate(zip(pair_order, pair_labels)) 
#                if len(groups[i]) > 0]

# if valid_pairs:
#     valid_pair_names, valid_labels, valid_groups = zip(*valid_pairs)
    
#     bp = ax.boxplot(valid_groups, tick_labels=valid_labels, patch_artist=True)
    
#     # Color boxes based on logged wetland connectivity
#     for i, pair in enumerate(valid_pair_names):
#         color = pair_colors[pair]
#         bp['boxes'][i].set_facecolor(color)
#         bp['boxes'][i].set_alpha(0.6)
    
#     for median in bp['medians']:
#         median.set_color('black')
#         median.set_linewidth(1.5)

# ax.set_ylabel("Depth Change (m)")
# ax.set_xlabel("Connectivity Pairing (Logged → Reference)")
# ax.set_title("Depth Change by Logged-Reference Connectivity Pairing")
# plt.xticks(rotation=45, ha='right')

# ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

# # Legend showing logged wetland connectivity
# legend_elements = [Patch(facecolor=connectivity_config[conn]['color'], 
#                          label=f"Logged: {connectivity_config[conn]['label']}", 
#                          alpha=0.6) 
#                    for conn in connectivity_order]
# ax.legend(handles=legend_elements, loc='best')

# plt.tight_layout()
# plt.show()

# %% 7.0 Nine-panel histogram of depth shifts by connectivity pairing (same series as 5.0)
# Rows = logged wetland connectivity (flow-through, first order, giw)
# Columns = reference wetland connectivity (flow-through, first order, giw)
# Each panel shows histogram of mean_depth_change for that pair_connect category

# Compute global axis limits for common axes
all_depth_changes = plot_data['mean_depth_change'].dropna()
x_min, x_max = all_depth_changes.min(), all_depth_changes.max()
x_padding = (x_max - x_min) * 0.1
x_lim = (x_min - x_padding, x_max + x_padding)

# Create 3x3 grid: rows = logged connectivity, cols = reference connectivity
fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)

bins = 15

# Map connectivity to row/column indices
conn_to_idx = {'flow-through': 0, 'first order': 1, 'giw': 2}
conn_labels = ['Flow-through', '1st Order', 'GIW']

for i, log_conn in enumerate(connectivity_order):
    for j, ref_conn in enumerate(connectivity_order):
        ax = axes[i, j]
        
        # Filter data for this connectivity pairing
        pair_data = plot_data[
            (plot_data['logged_connect'] == log_conn) & 
            (plot_data['ref_connect'] == ref_conn)
        ]['mean_depth_change']
        
        ax.hist(pair_data, bins=bins, range=x_lim, color='grey', 
                alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='maroon', linestyle='--', linewidth=1.5)
        # Add count annotation
        ax.text(0.95, 0.95, f'n={len(pair_data)} pairs', ha='right', va='top',
                transform=ax.transAxes, fontsize=9)
        
        # Add row labels on left column
        if j == 0:
            ax.set_ylabel(conn_labels[i], fontsize=10)
        
        # Add column labels on top row
        if i == 0:
            ax.set_title(conn_labels[j], fontsize=10)

# Add overall axis labels
fig.supxlabel('Depth Change (m)', fontsize=14)
fig.supylabel('Logged Wetland Connectivity', fontsize=14, x=0.02)
fig.suptitle('Reference Wetland Connectivity', fontsize=14, y=0.96)

plt.tight_layout(rect=[0.04, 0.02, 1, 0.96])
plt.show()
# %% 6.0 Test 9-panel histogram of depth shifts by connectivity pairing
