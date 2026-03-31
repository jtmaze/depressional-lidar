# %% 1.0 Libraries and packages
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

data_dir = 'D:/depressional_lidar/data/delmarva/'
parcel_path = f'{data_dir}/in_data/sites_and_boundaries/mine_boundary.shp'
wells_path = f'{data_dir}/delmarva_well_points.shp'

ft_to_meter = 0.3048
mine_thresh = 3_000 #ft

# Output paths 
results_path = f'{data_dir}/out_data/well_gw_impacts.csv'
mine_extent_path = f'{data_dir}/in_data/mine_pond.shp'

# %% 2.0 Read the parcel boundary and shrink by 200ft for pond shape

parcel = gpd.read_file(parcel_path) 
parcel = parcel.to_crs('EPSG:26918')
parcel = parcel[['ACRES', 'ACCTID', 'geometry']]

wells = gpd.read_file(wells_path)
wells = wells[~wells['catchment'].isin(['Jackson Lane', 'Tiger Paw'])]

mine_buffer = 200 * ft_to_meter
mine_extent = parcel.buffer(-mine_buffer)

# %% 3.0 Scenario fits

# Neumann 1972: y = -287.9 / (x + 48.57)^0.9675
# Papadopulos-Cooper 1967 (105,633 gpd, 20 days): y = -3.933 / (x + 17.35)^0.4473
# Papadopulos-Cooper 1967 (116,196 gpd, 30 days): y = -5.171 / (x + 16.53)^0.4101
# Thiem (time-averaged equivalent): y = -1e+06 / (x + 524.1)^1.828
# Thiem (long-term adjustec values): y = -1e+06 / (x + 512.9)^1.971


# Extract parameters from each scenario: y = numerator / (x + constant)^exponent
scenario_params = {
    'n72': {'numerator': -287.9, 'constant': 48.57, 'exponent': 0.9675},
    'pc_20d': {'numerator': -3.933, 'constant': 17.35, 'exponent': 0.4473},
    'pc_30d': {'numerator': -5.171, 'constant': 16.53, 'exponent': 0.4101},
    'thiem_avgd': {'numerator': -1e6, 'constant': 524.1, 'exponent': 1.828},
    'thiem_lt': {'numerator': -1e6, 'constant': 512.9, 'exponent': 1.971},
}

scenario_labels = {
    'n72': 'Neumann 1972',
    'pc_20d': 'Papadopulos-Cooper 1967 (105,633 gpd, 20 days)',
    'pc_30d': 'Papadopulos-Cooper 1967 (116,196 gpd, 30 days)',
    'thiem_avgd': 'Thiem (time-averaged equivalent)',
    'thiem_lt': 'Thiem (long-term adjustec values)',
}

scenarios_df = pd.DataFrame(scenario_params).T
scenarios_df['label'] = scenarios_df.index.map(scenario_labels)

print(scenarios_df)

# %% 4.0 Calculate each well's distance from mine_extent and calculate impact for each

wells['mine_dist_m'] = wells.geometry.distance(mine_extent.union_all())
wells['mine_dist_ft'] = wells['mine_dist_m'] / ft_to_meter
impacted_wells = wells[wells['mine_dist_ft'] < mine_thresh]

calc = []

for idx, row in scenarios_df.iterrows(): 

    temp = impacted_wells.copy()
    a = row['numerator']
    p = row['exponent']
    x_0 = row['constant']

    temp['gw_draw_ft'] = a / ((temp['mine_dist_ft'] + x_0)**p)
    temp['scenario'] = idx

    calc.append(temp)

results = pd.concat(calc)

# %% 5.0 View distance for impacted wells

import matplotlib.pyplot as plt

# Define markers for each well type
marker_map = {
    'SW': 'o',      # circle
    'CH': 's',      # square
    'UW': '^',      # triangle
}

plt.figure(figsize=(10, 2))

# Plot each well type with its own marker
for well_type in impacted_wells['type'].unique():
    mask = impacted_wells['type'] == well_type
    wells_subset = impacted_wells[mask]
    marker = marker_map.get(well_type, 'o')  # default to circle if type not in map
    
    if well_type == 'SW':
        color='red'
    else:
        color='black'
    plt.scatter(wells_subset['mine_dist_ft'], 
                np.random.normal(0, 0.02, size=len(wells_subset)),  # jitter y-position
                marker=marker, 
                alpha=0.6, 
                c=color,
                s=100,
                label=well_type)

plt.title('Distance to Mine for Impacted Wells')
plt.xlabel('Distance to Mine Boundary (ft)')
plt.ylim(-0.15, 0.15)
plt.yticks([])  
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Well Type', loc='upper right')
plt.xlim(0, 3_000)
plt.tight_layout()
plt.show()

# %% 6.0 Plot the range of impacts for each wetland (SW) well

wetland_results = results[results['type'] == 'SW'].copy()

# Calculate mean drawdown per wetland and sort 
# (Since values are negative, sorting ascending puts the largest magnitude drawdown first)
mean_drawdown = wetland_results.groupby('wetland_id')['gw_draw_ft'].mean().sort_values()

# Use the sorted index for our x-axis ordering
unique_wells = mean_drawdown.index.tolist()
well_positions = {well_id: i for i, well_id in enumerate(unique_wells)}

# Color map for scenarios
scenario_colors = {
    'n72': '#1f77b4',           # blue
    'pc_20d': '#ff7f0e',        # orange
    'pc_30d': '#2ca02c',        # green
    'thiem_avgd': '#d62728',    # red
    'thiem_lt': '#9467bd',      # purple
}

fig, ax = plt.subplots(figsize=(10, 8))

for scenario in wetland_results['scenario'].unique():
    scenario_data = wetland_results[wetland_results['scenario'] == scenario]
    

    x_pos = [well_positions[well_id] for well_id in scenario_data['wetland_id']]
    
    ax.scatter(x_pos, 
               scenario_data['gw_draw_ft'],
               c=scenario_colors.get(scenario, '#888888'),
               label=scenario,
               alpha=0.75,
               s=200,
               edgecolor='k',
               linewidth=0.5)

ax.scatter(range(len(unique_wells)), 
           mean_drawdown, 
           marker='x', 
           color='black', 
           s=100, 
           zorder=5, 
           linewidths=2,
           label='Mean Drawdown')

# Format x-axis
ax.set_xticks(range(len(unique_wells)))
ax.set_xticklabels(unique_wells, rotation=45, ha='right')

ax.set_xlabel('Wetland ID')
ax.set_ylabel('Groundwater Drawdown (ft)')
ax.set_title('Predicted Aquifer Drawdown by Scenario and Wetland')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(title='Scenario', loc='best')

plt.tight_layout()
plt.show()

# %% 7.0 Write the results file and the mine boundar

results.to_csv(results_path, index=False)
mine_extent.to_file(mine_extent_path)


# %%
