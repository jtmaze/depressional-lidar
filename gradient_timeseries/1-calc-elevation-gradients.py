# %% 1.0 Libraries and file paths

import os
from itertools import combinations
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

os.chdir('D:/depressional_lidar/delmarva/')

local_projection = 'EPSG:26917' # NAD83 / UTM zone 17N
wl_path = './waterlevel_data/output_JM_2019_2022.csv'
meta_path = './delmarva_well_metadata_for_EDI_updated.xlsx'
points_path = './sites_and_boundaries/well_points.shp'

wl = pd.read_csv(wl_path)
elevations = pd.read_excel(meta_path, sheet_name='Updated')
well_pts = gpd.read_file(points_path)
# %% 2.0  Generate well combinations and prep elevation data

# Prep elevation data
elevations = elevations[elevations['elevation_to_catchment_bottom_cm'] != 'not linked']
catchment_bottoms = elevations['elevation_to_catchment_bottom_cm'] == 'catchment_low_datum'
elevations.loc[catchment_bottoms, 'elevation_to_catchment_bottom_cm'] = 0.0

# Prep the points data
well_pts = well_pts[['well_id', 'geometry']]
well_pts = well_pts.to_crs(local_projection)
well_pts = pd.merge(well_pts, elevations[['well_id', 'elevation_to_catchment_bottom_cm']], on='well_id', how='left')

# %% 2.1 Generate sets with each possible combination of site pairs

jl_wells = (
    elevations[elevations['catchment'] == 'Jackson Lane'][['well_id', 'elevation_to_catchment_bottom_cm']]
).sort_values(
    by='elevation_to_catchment_bottom_cm',
    ascending=False
)
jl_pairs = list(combinations(jl_wells['well_id'], 2))
jl_elevation_gradients = pd.DataFrame({
    'well_pair': [f'{pair[0]}__to__{pair[1]}' for pair in jl_pairs],
    'well0': [pair[0] for pair in jl_pairs],
    'well1': [pair[1] for pair in jl_pairs]
})

bc_wells = (
    elevations[elevations['catchment'] == 'Baltimore Corner'][['well_id', 'elevation_to_catchment_bottom_cm']]
).sort_values(
    by='elevation_to_catchment_bottom_cm',
    ascending=False
)
bc_pairs = list(combinations(bc_wells['well_id'], 2))
bc_elevation_gradients = pd.DataFrame({
    'well_pair': [f'{pair[0]}__to__{pair[1]}' for pair in bc_pairs],
    'well0': [pair[0] for pair in bc_pairs],
    'well1': [pair[1] for pair in bc_pairs]
})


# %% 3.0 Calculate elevation gradients for each well pair

def assign_elevation_gradients(
        gradient_df: pd.DataFrame, 
        points_gdf: pd.DataFrame
):
    
    def find_z_values(row):
        """Fetches elevation values from points df for gradients df"""
        z0 = points_gdf[points_gdf['well_id'] == row['well0']]['elevation_to_catchment_bottom_cm'].values[0]
        z1 = points_gdf[points_gdf['well_id'] == row['well1']]['elevation_to_catchment_bottom_cm'].values[0]
        return pd.Series({'z0': z0, 'z1': z1})

    
    z_wells = gradient_df.apply(find_z_values, axis=1)
    gradient_df['z0'] = z_wells['z0']
    gradient_df['z1'] = z_wells['z1']
    gradient_df['dz'] = gradient_df['z0'] - gradient_df['z1']

    def calc_well_dist(row):
        """Calculates distance (m) between wells for gradients df"""
        well0 = row['well0']
        well0_geom = points_gdf[points_gdf['well_id'] == well0].geometry.iloc[0]
        well1 = row['well1']
        well1_geom = points_gdf[points_gdf['well_id'] == well1].geometry.iloc[0]

        distance_m = well0_geom.distance(well1_geom)
        return distance_m
    
    gradient_df['well_dist_m'] = gradient_df.apply(calc_well_dist, axis=1) 
    gradient_df['elevation_gradient_cm_m'] = gradient_df['dz'] / gradient_df['well_dist_m']

    return gradient_df

jl_elevation_gradients = assign_elevation_gradients(jl_elevation_gradients, well_pts)
bc_elevation_gradients = assign_elevation_gradients(bc_elevation_gradients, well_pts)

# %% 5.0 Write the output

bc_elevation_gradients.to_csv('./out_data/bc_elevation_gradients.csv', index=False)
jl_elevation_gradients.to_csv('./out_data/jl_elevation_gradients.csv', index=False)

# %% 6.0 Plot the elevation gradients

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

# Use the original elevation gradients without taking absolute values
jl_gradients = jl_elevation_gradients['elevation_gradient_cm_m']
bc_gradients = bc_elevation_gradients['elevation_gradient_cm_m']

# Calculate means
jl_mean = jl_gradients.mean()
bc_mean = bc_gradients.mean()

# Find the common range for both histograms
min_gradient = min(jl_gradients.min(), bc_gradients.min())
max_gradient = max(jl_gradients.max(), bc_gradients.max())
bins = 20

# Plot histograms
ax1.hist(jl_gradients, bins=bins, color='blue', edgecolor='black', alpha=0.7, range=(min_gradient, max_gradient))
ax2.hist(bc_gradients, bins=bins, color='green', edgecolor='black', alpha=0.7, range=(min_gradient, max_gradient))

# Add mean lines
ax1.axvline(jl_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {jl_mean:.3f} cm/m')
ax2.axvline(bc_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {bc_mean:.3f} cm/m')

# Add legends to show mean values
ax1.legend()
ax2.legend()

# Add titles and labels
ax1.set_title('Jackson Lane Elevation Gradients')
ax2.set_title('Baltimore Corner Elevation Gradients')
ax1.set_xlabel('Elevation Gradient (cm/m)')
ax2.set_xlabel('Elevation Gradient (cm/m)')
ax1.set_ylabel('Frequency (n well pairs)')

plt.tight_layout()

