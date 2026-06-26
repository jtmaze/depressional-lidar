# %% 1.0 Libraries and file paths

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

bradford_boundary_path = "D:/depressional_lidar/data/bradford/bradford_boundary.shp"
depressions_path = "D:/depressional_lidar/data/bradford/out_data/bradford_wetland_basins_vf.shp"
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
bradford_nwi_path = 'D:/depressional_lidar/data/bradford/in_data/original_basins/bradford_nwi_polygons.shp'

# %% 2.0 Read data and print bradford area

bradford_boundary = gpd.read_file(bradford_boundary_path)
depressions = gpd.read_file(depressions_path)
nwi = gpd.read_file(bradford_nwi_path)

nwi.to_crs(depressions.crs, inplace=True)
bradford_boundary.to_crs(depressions.crs, inplace=True)
print(bradford_boundary.area)

# %% 3.0 Clip the depressions to the bradford boundary area

clipped_depressions = gpd.clip(depressions, bradford_boundary)
clipped_nwi = gpd.clip(nwi, bradford_boundary)

print(clipped_nwi)

# %% 4.0 General info about clipped depressions

nwi_area = clipped_nwi.geometry.area
boundary_area = bradford_boundary.area.values[0]

print("Metric, Depressions, NWI")
print("-" * 30)
print(f"N (clipped): {len(clipped_depressions)} vs {len(clipped_nwi)}")
print(f"Max area (m²): {clipped_depressions.area_m2.max():.1f} vs {nwi_area.max():.1f}")
print(f"Min area (m²): {clipped_depressions.area_m2.min():.1f} vs {nwi_area.min():.1f}")
print(f"Mean area (m²): {clipped_depressions.area_m2.mean():.1f} vs {nwi_area.mean():.1f}")
print(f"Median area (m²): {clipped_depressions.area_m2.median():.1f} vs {nwi_area.median():.1f}")
print(f"Std area (m²): {clipped_depressions.area_m2.std():.1f} vs {nwi_area.std():.1f}")
print(f"Total area (m²): {clipped_depressions.area_m2.sum():.1f} vs {nwi_area.sum():.1f}")

dep_pct = clipped_depressions.area_m2.sum() / boundary_area * 100
nwi_pct = nwi_area.sum() / boundary_area * 100
print(f"% of boundary: {dep_pct:.2f}% vs {nwi_pct:.2f}%")

# %% 5.0 Compare the clipped depressions power function to NWI

def ccdf_and_fit(area_values):
    areas = np.sort(area_values)
    n = len(areas)
    ccdf = 1 - np.arange(n) / n
    mask = ccdf > 0
    log_x = np.log10(areas[mask])
    log_y = np.log10(ccdf[mask])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fit_y = 10 ** (slope * log_x + intercept)
    return areas, ccdf, areas[mask], fit_y, slope

dep_areas, dep_ccdf, dep_fit_x, dep_fit_y, dep_slope = ccdf_and_fit(clipped_depressions.area_m2.values)
all_nwi_areas = clipped_nwi.geometry.area.values
filtered_nwi_areas = all_nwi_areas[all_nwi_areas > 1000]
nwi_areas, nwi_ccdf, nwi_fit_x, nwi_fit_y, nwi_slope = ccdf_and_fit(filtered_nwi_areas)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(dep_areas, dep_ccdf, s=10, alpha=0.5, color="steelblue", label="Depressions")
ax.plot(dep_fit_x, dep_fit_y, color="steelblue", linestyle="-", label=f"Dep. slope = {dep_slope:.2f}")
ax.scatter(nwi_areas, nwi_ccdf, s=10, alpha=0.5, color="darkorange", label="NWI")
ax.plot(nwi_fit_x, nwi_fit_y, color="darkorange", linestyle="-", label=f"NWI slope = {nwi_slope:.2f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Wetland area (m²)")
ax.set_ylabel("P(Area ≥ x)")
ax.legend()
plt.show()

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
