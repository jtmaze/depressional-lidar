# %% 1.0 Libraries and File Paths
import geopandas as gpd
from basin_attributes import WetlandBasin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

source_dem = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
basins_path = 'D:/depressional_lidar/data/bradford/in_data/basins_assigned_wetland_ids.shp'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'

footprints = gpd.read_file(basins_path)

well_points = gpd.read_file(well_points_path)
well_points = well_points[['wetland_id', 'type', 'rtk_elevat', 'geometry']]

# %% 2.0 Clean the well points df

well_points.rename(
    columns={
        'rtk_elevat': 'rtk_elevation'
    },
    inplace=True
)
well_points = well_points[
    (well_points['type'] == 'core_well') | (well_points['type'] == 'wetland_well')
]

# %% 3.0 Make a list with basins of interest

basin_ids = [
    '14_500', '5_597', '13_267', '5_546', 
    '15_409', '14_418', '5a_550', '5_573',
    '13_263', '15_268', '15_4', '15_516',
    '5_161', '5_510', '14_115', '13_263',
    '13_410', '13_274', '13_271', '5_560'
]

transect_buffers = [0, 20, 30, 40]

aggregated_transects = []

for i in basin_ids:
    for j in transect_buffers:
        well_point = well_points[well_points['wetland_id'] == i]
        basin_footprint = footprints[footprints['wetland_id'] == i]
        basin = WetlandBasin(
            wetland_id=i,
            source_dem_path=source_dem,
            footprint=basin_footprint,
            well_point_info=well_point,
            transect_method='deepest',
            transect_n=12,
            transect_buffer=j
        )

        basin.radial_transects_map()
        aggregated_transect = basin.aggregate_radial_transects()
        lowest_elevation = aggregated_transect['mean'].min()
        aggregated_transect['mean_relative_to_low'] = (
            aggregated_transect['mean'] - lowest_elevation
        )
        aggregated_transect['transect_buffer'] = j
        aggregated_transect['basin_id'] = i
        basin.plot_aggregated_radial_transects()
        aggregated_transects.append(aggregated_transect)
        del basin

# %% 4.1 Visualize each basin's aggregated transects

# Concatenate all transects into a single DataFrame with basin_id
combined_transects = pd.concat(aggregated_transects)

def compute_hayashi_p(df, r0, r1):

    temp = df.copy()
    z0 = temp[temp['distance_m'] == r0]['mean_relative_to_low'].values[0]
    z1 = temp[temp['distance_m'] == r1]['mean_relative_to_low'].values[0]

    if np.isnan(z0) or np.isnan(z1):
        return np.nan
    else:
        p = np.log(z1/z0) / np.log(r1/r0)
        return p

hayashi_p_constants = []

subset = combined_transects[combined_transects['transect_buffer'] == 0]
basin_ids = subset['basin_id'].unique()
print(basin_ids)

# Create the plot
plt.figure(figsize=(10, 6))
for id in basin_ids:
    temp = subset[subset['basin_id'] == id]
    if temp.empty:
        continue
    p_value = compute_hayashi_p(temp, r0=2, r1=20)

    hayashi_p_constants.append({
        'basin_id': id,
        'p_value': p_value
    })
    temp = temp.sort_values(by='distance_m')
    plt.plot(temp['distance_m'], temp['mean_relative_to_low'], label=id)

plt.xlabel('Distance from radial reference (m)')
plt.ylabel('Elevation from Wetland Bottom (m)')
plt.xlim(0, 150)
plt.title('Transect Profiles by Basin')
plt.legend()
plt.grid(True)
plt.show()

# %%
hayashi_p_constants_df = pd.DataFrame(hayashi_p_constants)

# Now make a barplot for each basin's mean p-values
plt.figure(figsize=(10, 10))
for id in basin_ids:
    temp = hayashi_p_constants_df[hayashi_p_constants_df['basin_id'] == id]
    if temp.empty:
        continue
    plt.bar(temp['basin_id'], temp['p_value'], label=id)

plt.xlabel('Basin ID')
plt.ylabel('Hayashi P-value') 
plt.title('Hayashi P-values by Basin')
plt.xticks(rotation=45)
plt.show()

# %% Histogram of Hayashi p values

plt.figure(figsize=(10, 10))
sns.histplot(data=hayashi_p_constants_df, x='p_value', kde=True)
plt.xlabel('Hayashi P-value')
plt.ylabel('Frequency')
plt.title('KDE of Hayashi P-values')
plt.axvline(hayashi_p_constants_df['p_value'].mean(), color='red', linewidth=3)
plt.show()

# %% 4.2 Take the mean distance profile grouped by transect buffer

landscape_aggregated = combined_transects.groupby(['distance_m', 'transect_buffer']).agg({
    'mean_relative_to_low': ['mean', 'std', lambda x: x.quantile(0.75) - x.quantile(0.25)]
}).reset_index()

landscape_aggregated.columns = ['distance_m', 'transect_buffer', 'mean_relative_to_low', 'std', 'iqr']

# Create the landscape mean plot with different colors by transect buffer
plt.figure(figsize=(10, 6))

# Define a colormap for different transect buffers
colors = plt.cm.viridis(np.linspace(0, 1, len(transect_buffers)))

# Plot each transect buffer as a separate line with its own color
for i, buffer in enumerate(transect_buffers):
    buffer_data = landscape_aggregated[landscape_aggregated['transect_buffer'] == buffer].sort_values('distance_m')

    plt.plot(buffer_data['distance_m'], buffer_data['mean_relative_to_low'], 
             linewidth=2, label=f'Buffer {buffer}m', 
             color=colors[i])
    
    plt.fill_between(
        buffer_data['distance_m'],
        buffer_data['mean_relative_to_low'] - buffer_data['std']/2,
        buffer_data['mean_relative_to_low'] + buffer_data['std']/2,
        alpha=0.2, color=colors[i]
    )

plt.xlabel('Distance from radial reference (m)')
plt.ylabel('Elevation from Wetland Bottom (m)')
plt.xlim(0, 150)
plt.title('Landscape-Level Basin Profiles by Transect Buffer')
plt.legend()
plt.grid(True)
plt.show()

# %% Plot Hayashi p values by landscape buffer

r1_vals = [10, 20, 30, 40, 50]
landscape_hayashi_ps = []

# Compute Hayashi P-values for each transect buffer and distance
for buffer in transect_buffers:
    buffer_data = landscape_aggregated[landscape_aggregated['transect_buffer'] == buffer]
    for r1 in r1_vals:
        p = compute_hayashi_p(buffer_data, r0=2, r1=r1)
        landscape_hayashi_ps.append({
            'transect_buffer': buffer,
            'r1': r1,
            'hayashi_p': p
        })

# Create a DataFrame for the Hayashi P-values
landscape_hayashi_ps_df = pd.DataFrame(landscape_hayashi_ps)

# Plot the Hayashi P-values
plt.figure(figsize=(10, 6))
for buffer in transect_buffers:
    buffer_data = landscape_hayashi_ps_df[landscape_hayashi_ps_df['transect_buffer'] == buffer]
    plt.plot(buffer_data['r1'], buffer_data['hayashi_p'], label=f'Buffer {buffer}m')

plt.xlabel('Distance (m)')
plt.ylabel('Hayashi P-value')
plt.title('Hayashi P-values by Landscape Buffer')
plt.legend()
plt.grid(True)
plt.show()

# %%
