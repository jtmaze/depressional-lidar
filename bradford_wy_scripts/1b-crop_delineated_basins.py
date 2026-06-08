# %% 1.0 Libraries and file paths

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

bradford_boundary_path = "D:/depressional_lidar/data/bradford/bradford_boundary.shp"
depressions_path = "D:/depressional_lidar/data/bradford/out_data/bradford_wetland_basins_vf.shp"
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'


# %% 2.0 Read data and print bradford area

bradford_boundary = gpd.read_file(bradford_boundary_path)
depressions = gpd.read_file(depressions_path)
print(depressions.crs)

bradford_boundary.to_crs(depressions.crs, inplace=True)
print(bradford_boundary.area)

# %% 3.0 Clip the depressions to the bradford boundary area

clipped_depressions = gpd.clip(depressions, bradford_boundary)

# %% 4.0 General info about clipped depressions

print(len(depressions))
print(len(clipped_depressions))
print(clipped_depressions.area_m2.max())
print(clipped_depressions.area_m2.min())

print(clipped_depressions.area.sum() / bradford_boundary.area * 100)

# %% 5.0 Write the clipped depressions as a shapefile

#clipped_depressions.to_file("D:/depressional_lidar/data/bradford/out_data/bradford_wetland_basins_vf_clipped.shp")

# %% 6.0 Write sepperate well points just for visualization

# well_points = (
#     gpd.read_file(well_points_path)[['wetland_id', 'type', 'geometry', 'site']]
#     .rename(columns={'rtk_z': 'rtk_z'})
#     .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
# )

# well_points.to_file('D:/depressional_lidar/data/bradford/out_data/bradford_wy_viz_wells.shp')


# %%
