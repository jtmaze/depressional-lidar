# %% 1.0 Libraries and file paths

import os
from itertools import combinations
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

os.chdir('/Users/jmaze/Documents/projects/depressional_lidar/')

wl_path = './delmarva/waterlevel_data/output_JM_2019_2022.csv'
survey_path = './delmarva/survey_elevations.xlsx'
well_pts_path = './delmarva/trimble_well_pts.shp'

wl = pd.read_csv(wl_path)
elevations = pd.read_excel(survey_path, sheet_name='Sheet1')
elevations.rename(
    columns={'Site': 'Site_Name'}, inplace=True
)
print(elevations.columns)
well_pts = gpd.read_file(well_pts_path)
well_pts.rename(
    columns={'Descriptio': 'Site_Name'}, inplace=True
)
print(well_pts.columns)
well_pts['Site_Name'] = well_pts['Site_Name'].str.replace('well', '', case=False)
well_pts['Site_Name'] = well_pts['Site_Name'].str.replace(r'\s+', '', regex=True)

# %% 2.0  Generate well combinations and prep elevation data

# %% 2.1 Generate sets with each possible combination of site pairs
elevations = elevations[~elevations['Site_Name'].str.contains('high')]
jl_wells = elevations[elevations['Catchment'] == 'Jackson']['Site_Name']
bc_wells = elevations[elevations['Catchment'] == 'Baltimore']['Site_Name']

jl_pairs = list(combinations(jl_wells, 2))
jl_elevation_gradients = pd.DataFrame({
    'well_pair': [f'{pair[0]}__to__{pair[1]}' for pair in jl_pairs],
    'well0': [pair[0] for pair in jl_pairs],
    'well1': [pair[1] for pair in jl_pairs]
})

bc_pairs = list(combinations(bc_wells, 2))
bc_elevation_gradients = pd.DataFrame({
    'well_pair': [f'{pair[0]}__to__{pair[1]}' for pair in bc_pairs],
    'well0': [pair[0] for pair in bc_pairs],
    'well1': [pair[1] for pair in bc_pairs]
})


# %% 2.2 Join the elevation data to the well point geometries

# Merge the dataframes
z_points_df = pd.merge(elevations, well_pts, on='Site_Name', how='left').drop(
    columns=['Elevation_local_Catchment', 'Flag_Label', 'Vert_Prec', 
             'Horz_Prec', 'Point_ID', 'Comment'
    ]
)
# Convert back to GeoDataFrame with the correct CRS
z_points = gpd.GeoDataFrame(z_points_df, geometry='geometry', crs=well_pts.crs)
print(z_points.crs)

# %% 3.0

def assign_elevation_gradients(
        gradient_df: pd.DataFrame, 
        points_gdf: pd.DataFrame
):
    
    def find_z_values(row):
        """Fetches elevation values from points df for gradients df"""
        z0 = points_gdf[points_gdf['Site_Name'] == row['well0']]['Elevation'].values[0]
        z1 = points_gdf[points_gdf['Site_Name'] == row['well1']]['Elevation'].values[0]
        return pd.Series({'z0': z0, 'z1': z1})

    
    z_wells = gradient_df.apply(find_z_values, axis=1)
    gradient_df['z0'] = z_wells['z0']
    gradient_df['z1'] = z_wells['z1']
    gradient_df['dz'] = gradient_df['z0'] - gradient_df['z1']

    def calc_well_dist(row):
        """Calculates distance (m) between wells for gradients df"""
        well0 = row['well0']
        well0_geom = points_gdf[points_gdf['Site_Name'] == well0].geometry.iloc[0]
        well1 = row['well1']
        well1_geom = points_gdf[points_gdf['Site_Name'] == well1].geometry.iloc[0]

        distance_m = well0_geom.distance(well1_geom)
        return distance_m
    
    gradient_df['well_dist_m'] = gradient_df.apply(calc_well_dist, axis=1) 
    gradient_df['elevation_gradient_cm_m'] = gradient_df['dz'] / gradient_df['well_dist_m']

    return gradient_df
# %%

jl_elevation_gradients = assign_elevation_gradients(jl_elevation_gradients, z_points)
bc_elevation_gradients = assign_elevation_gradients(bc_elevation_gradients, z_points)

# %%

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

# Get the absolute values of elevation gradients
jl_abs_gradients = abs(jl_elevation_gradients['elevation_gradient_cm_m'])
bc_abs_gradients = abs(bc_elevation_gradients['elevation_gradient_cm_m'])

# Calculate means
jl_mean = jl_abs_gradients.mean()
bc_mean = bc_abs_gradients.mean()

# Find the maximum value for consistent bin ranges
max_gradient = max(jl_abs_gradients.max(), bc_abs_gradients.max())
bins = 20

# Plot histograms
ax1.hist(jl_abs_gradients, bins=bins, color='blue', edgecolor='black', alpha=0.7, range=(0, max_gradient))
ax2.hist(bc_abs_gradients, bins=bins, color='green', edgecolor='black', alpha=0.7, range=(0, max_gradient))

# Add mean lines
ax1.axvline(jl_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {jl_mean:.3f} cm/m')
ax2.axvline(bc_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {bc_mean:.3f} cm/m')

# Add legends to show mean values
ax1.legend()
ax2.legend()

# Add titles and labels
ax1.set_title('Jackson Lane Elevation Gradients')
ax2.set_title('Baltimore Corner Elevation Gradients')
ax1.set_xlabel('Absolute Elevation Gradient (cm/m)')
ax2.set_xlabel('Absolute Elevation Gradient (cm/m)')
ax1.set_ylabel('Frequency (n well pairs)')

plt.tight_layout()

# %%

bc_elevation_gradients.to_csv('./delmarva/out_data/bc_elevation_gradients.csv', index=False)
jl_elevation_gradients.to_csv('./delmarva/out_data/jl_elevation_gradients.csv', index=False)
z_points.to_file('./delmarva/out_data/well_pts_clean.shp')
# %%
