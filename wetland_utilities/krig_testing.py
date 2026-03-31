# %% 1.0 Libraries and file paths
import sys
# NOTE: This shim facilites imports by bringing the root directory higher
PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import geopandas as gpd

from wetland_utilities.wtd_wetlandscape_krig import WellArray, WTDSurface

well_ts_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'

# %% 2.0 Read data

well_ts = pd.read_csv(well_ts_path)
#print(well_ts.head(10))

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'geometry', 'z_dem', 'site']]
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)

basin_13_ids = ['13_263', '13_267', '13_271', '13_410', '13_274', 'Donor_wetland', 'Receiver_wetland']
well_points = well_points[~well_points['wetland_id'].isin(basin_13_ids)]
print(len(well_points))

boundary = well_points.geometry.unary_union.convex_hull.buffer(500)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
well_points.plot(ax=ax, color='red', markersize=20)
gpd.GeoSeries([boundary]).plot(ax=ax, facecolor='none', edgecolor='blue')
plt.show()

# %% 3.0 

well_array_early = WellArray(
    well_pts=well_points,
    well_ts=well_ts,
    begin='2023-03-01',
    end='2023-03-28'
)

wtd_surface_early = WTDSurface(
    well_array=well_array_early,
    krig_params={'variogram_model': 'linear', 'n_lags': 6},
    coarse_grid_dims=(1000, 1000), 
    boundary=boundary
)

wtd_surface_early.plot_interpolation_result()
wtd_surface_early.plot_sigma_squared()

well_array_late = WellArray(
    well_pts=well_points,
    well_ts=well_ts,
    begin='2024-04-25',
    end='2024-04-26'
)

wtd_surface_late = WTDSurface(
    well_array=well_array_late,
    krig_params={'variogram_model': 'linear', 'n_lags': 6},
    coarse_grid_dims=(1000, 1000), 
    boundary=boundary
)




# %% 4.0 Plot to compare differences as a map. 

array_early = wtd_surface_early.okr_result['z']
array_late = wtd_surface_late.okr_result['z']
diff = array_late - array_early  # positive = water table rose, negative = fell

extent = [
    wtd_surface_early._x_grid.min(), wtd_surface_early._x_grid.max(),
    wtd_surface_early._y_grid.min(), wtd_surface_early._y_grid.max(),
]
vmax = np.nanmax(np.abs(diff))

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(diff, extent=extent, origin='lower', cmap='RdBu', aspect='auto',
               vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label='WSE change (m): late - early')
gpd.GeoSeries([boundary]).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
ax.set_title('Water Surface Elevation Change')
plt.show()







# %%
