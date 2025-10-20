# %% Libraries and file paths

import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from basin_attributes import WetlandBasin


wetlands_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'

wetlands = gpd.read_file(wetlands_path)
well_points = gpd.read_file(well_points_path)
well_points.rename(
    columns={'rtk_elevat': 'rtk_elevation'},
    inplace=True,
)
well_points = well_points[(well_points['type'] == 'core_well') | (well_points['type'] == 'wetland_well')]

unique_ids = wetlands['wetland_id'].unique()


# %% 

curves = []

for i in unique_ids:

    well_pt = well_points[well_points['wetland_id'] == i]
    footprint = wetlands[wetlands['wetland_id'] == i]

    basin = WetlandBasin(
        wetland_id=i,
        source_dem_path=dem_path,
        footprint=None,
        well_point_info=well_pt,
        transect_method=None,
        transect_n=None,
        transect_buffer=300
    )

    cum_area_m2, bin_centers = basin.calculate_hypsometry(method='pct_trim')

    df_curve = pd.DataFrame({
        'wetland_id': i,
        'elevation': bin_centers,
        'cum_area_m2': cum_area_m2
    })

    curves.append(df_curve)

out_df = pd.concat(curves, ignore_index=True)

# %% Rescale the elevation values to min and max for each wetland

out_df['depth'] = out_df.groupby('wetland_id')['elevation'].transform(
    lambda x: (x - x.min())
)

out_df['area_rescaled'] = out_df.groupby('wetland_id')['cum_area_m2'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# %% Plot the hypsometric curves for all wetlands

# Assuming your data is in a DataFrame called 'df'
plt.figure(figsize=(6, 6))

# Create a color palette with enough colors for all unique wetland_ids
unique_wetlands = out_df['wetland_id'].unique()


# Group by wetland_id and plot each group
for i, (wetland, group) in enumerate(out_df.groupby('wetland_id')):
    # Sort by depth to ensure smooth curves
    group = group.sort_values('depth')
    plt.plot(group['depth'], group['area_rescaled'], 
             label=wetland, linewidth=2)

plt.xlabel('Depth (m)', fontsize=12)
plt.ylabel('Rescaled Area', fontsize=12)
plt.title('Wetland Hypsometric Curves (No Shape, 300m buffer)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Plot the average hypsometric curve
