# %% 1.0 Libraries and packages
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'D:/depressional_lidar/data/delmarva/'
parcel_path = f'{data_dir}/in_data/sites_and_boundaries/mine_boundary.shp'
example_wetlands_path = f'{data_dir}/out_data/mdnr_example_wetlands.shp'
full_basins_path = f'{data_dir}/out_data/full_impact_wetlands.shp'

ft_to_meter = 0.3048
mine_thresh = 3_000 #ft

# Output paths 
results_path = f'{data_dir}/out_data/basin_gw_impacts.csv'
mine_extent_path = f'{data_dir}/in_data/mine_pond.shp'

# %% 2.0 Read the parcel boundary and shrink by 200ft for pond shape

parcel = gpd.read_file(parcel_path) 
parcel = parcel.to_crs('EPSG:26918')
parcel = parcel[['ACRES', 'ACCTID', 'geometry']]

basins = gpd.read_file(example_wetlands_path)

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

# %% 4.0 Calculate each example wetlands edge-to-edge distance from mine_extent apply each scenario

basins['mine_dist_m'] = basins.geometry.distance(mine_extent.union_all())
basins['mine_dist_ft'] = basins['mine_dist_m'] / ft_to_meter
impacted_basins = basins[basins['mine_dist_ft'] < mine_thresh]

calc = []

for idx, row in scenarios_df.iterrows(): 

    temp = impacted_basins.copy()
    a = row['numerator']
    p = row['exponent']
    x_0 = row['constant']

    temp['gw_draw_ft'] = a / ((temp['mine_dist_ft'] + x_0)**p)
    temp['scenario'] = idx

    calc.append(temp)

results = pd.concat(calc)


# %% 5.0 Plot the range of impacts for each wetland 

wetland_results = results.copy()
# Calculate mean drawdown per basin and sort 
# (Since values are negative, sorting ascending puts the largest magnitude drawdown first)
mean_drawdown = wetland_results.groupby('wetland_id')['gw_draw_ft'].mean().sort_values()

# Use the sorted index for our x-axis ordering
unique_basins = mean_drawdown.index.tolist()
basin_positions = {basin_id: i for i, basin_id in enumerate(unique_basins)}

# Color map for scenarios
scenario_colors = {
    'n72': '#1f77b4',           # blue
    'pc_20d': '#ff7f0e',        # orange
    'pc_30d': '#2ca02c',        # green
    'thiem_avgd': '#d62728',    # red
    'thiem_lt': '#9467bd',      # purple
}

fig, ax = plt.subplots(figsize=(12, 8))

for scenario in wetland_results['scenario'].unique():
    scenario_data = wetland_results[wetland_results['scenario'] == scenario]
    

    x_pos = [basin_positions[basin_id] for basin_id in scenario_data['wetland_id']]
    
    ax.scatter(x_pos, 
               scenario_data['gw_draw_ft'],
               c=scenario_colors.get(scenario, '#888888'),
               label=scenario,
               alpha=0.75,
               s=200,
               edgecolor='k',
               linewidth=0.5)

ax.scatter(range(len(unique_basins)), 
           mean_drawdown, 
           marker='x', 
           color='black', 
           s=100, 
           zorder=5, 
           linewidths=2,
           label='Mean Drawdown')

# Format x-axis with basin ID and distance
basin_distances = impacted_basins.set_index('wetland_id')['mine_dist_ft']
xlabels = [f"{basin_id}\n({int(round(basin_distances[basin_id] / 10) * 10)} ft)" 
           for basin_id in unique_basins]

ax.set_xticks(range(len(unique_basins)))
ax.set_xticklabels(xlabels, rotation=0, fontsize=12)

ax.set_xlabel('Wetland ID (Distance to Mine)', fontsize=14, labelpad=20)
ax.set_ylabel('Groundwater Drawdown (ft)', fontsize=14)
ax.set_title('Predicted Aquifer Drawdown by Scenario and Wetland', fontsize=14)
ax.tick_params(axis='y', labelsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(title='Scenario', loc='best', fontsize=12, title_fontsize=15)

plt.tight_layout()
plt.show()

# %% 6.0 Write the results file and the mine boundar

results.to_csv(results_path, index=False)
#mine_extent.to_file(mine_extent_path)

# %% 7.0 Apply the same logic to the full set of basins

full_basins = gpd.read_file(full_basins_path)
full_basins = full_basins.to_crs('EPSG:26918')

# Compute edge-to-edge distance and filter to mine threshold
full_basins['mine_dist_m'] = full_basins.geometry.distance(mine_extent.union_all())
full_basins['mine_dist_ft'] = full_basins['mine_dist_m'] / ft_to_meter
full_impacted = full_basins[full_basins['mine_dist_ft'] < mine_thresh].copy()

# Apply each GW scenario
full_calc = []

for idx, row in scenarios_df.iterrows():
    temp = full_impacted.copy()
    a = row['numerator']
    p = row['exponent']
    x_0 = row['constant']
    temp['gw_draw_ft'] = a / ((temp['mine_dist_ft'] + x_0)**p)
    temp['scenario'] = idx
    full_calc.append(temp)

full_results = pd.concat(full_calc)

# Build summary dataframe: mean drawdown across scenarios per basin
mean_draw = full_results.groupby(level=0)['gw_draw_ft'].mean().rename('mean_gw_draw_ft')

summary_df = full_impacted[['geometry']].copy()
summary_df['area_m2'] = summary_df.geometry.area
summary_df['centroid_x'] = summary_df.geometry.centroid.x
summary_df['centroid_y'] = summary_df.geometry.centroid.y
summary_df = summary_df.join(mean_draw)
summary_df = summary_df.drop(columns='geometry')

print(summary_df)

# %% 8.0 Plot of wetland counts as function of pit distance

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(full_impacted['mine_dist_ft'], bins=np.arange(0, mine_thresh + 100, 200),
        color='steelblue', edgecolor='white', linewidth=0.5)

ax.set_xlabel('Distance to Mine Pit (ft)', fontsize=15)
ax.set_ylabel('Number of Wetlands', fontsize=15)
ax.set_title('Wetland Counts by Distance', fontsize=16)
ax.tick_params(labelsize=13)
plt.tight_layout()
plt.show()

# %% 9.0 Plot of wetland counts as a function of water level loss bins

wtr_loss_bins = np.arange(summary_df['mean_gw_draw_ft'].min(), 0 + 0.1, step=0.1)

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(summary_df['mean_gw_draw_ft'], bins=wtr_loss_bins,
        color='indianred', edgecolor='white', linewidth=0.5)

ax.set_xlabel('Mean Groundwater Drawdown (ft)', fontsize=15)
ax.set_ylabel('Number of Wetlands', fontsize=15)
ax.set_title('Wetland Counts by Predicted Water Level Loss', fontsize=16)
ax.tick_params(labelsize=13)
plt.tight_layout()
plt.show()

# %% 10.0 Make a CDF of impacted wetland area by drawdown threshold

# Sort ascending (most negative = most impacted first), cumsum builds from most to least impacted.
# Result: at x=-2ft (left), only the most severely impacted area; at x=0 (right), all wetland area.
cdf_df = summary_df[['area_m2', 'mean_gw_draw_ft']].copy()
cdf_df = cdf_df.sort_values('mean_gw_draw_ft', ascending=True)
cdf_df['cumulative_area_acres'] = cdf_df['area_m2'].cumsum() / 4046.86

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(cdf_df['mean_gw_draw_ft'], cdf_df['cumulative_area_acres'],
        color='steelblue', linewidth=2)
ax.fill_between(cdf_df['mean_gw_draw_ft'], cdf_df['cumulative_area_acres'],
                alpha=0.2, color='steelblue')

ax.set_xlim(-2, 0)
ax.set_ylim(0, None)

ax.set_xlabel('Mean Groundwater Drawdown (ft)', fontsize=15)
ax.set_ylabel('Cumulative Wetland Area (acres)', fontsize=15)
ax.set_title('Cumulative Wetland Area by Water Level Loss', fontsize=16)
ax.tick_params(labelsize=13)

plt.tight_layout()
plt.show()
# %%
