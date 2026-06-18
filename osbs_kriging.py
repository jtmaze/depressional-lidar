# %% 1.0 Libraries and file paths

import sys
# NOTE: This shim facilites lateral imports by bringing the root directory higher
PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from wetland_utilities.wtd_wetlandscape_krig import WellArray, WTDSurface

well_ts_path = "D:/depressional_lidar/data/osbs/in_data/stage_data/osbs_daily_well_depth_Fall2025.csv"
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
pond_conditioning_pts_path = "D:/depressional_lidar/data/osbs/in_data/osbs_kriging_conditioning_pts.shp"


# %% 2.0 Read the data

well_ts = pd.read_csv(well_ts_path)
print(well_ts.head())
well_ts = well_ts[['date', 'wetland_id', 'indexed_well_depth_m', 'flag']]
well_ts.rename(
    columns={
        'indexed_well_depth_m': 'well_depth_m'
    }, 
    inplace=True
)

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'geometry', 'z_dem', 'site']]
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'OSBS'")
)

# %% 3.0 Quick timeseries plot of wetland_well_depths

# Convert date to datetime
well_ts['date'] = pd.to_datetime(well_ts['date'])

daily_avg = well_ts.groupby('date')['well_depth_m'].mean()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each individual wetland in faint color
for wetland_id in well_ts['wetland_id'].unique():
    wetland_data = well_ts[well_ts['wetland_id'] == wetland_id].sort_values('date')
    ax.plot(wetland_data['date'], wetland_data['well_depth_m'], 
            alpha=1, linewidth=1, label=f'Wetland {wetland_id}')

# Plot daily average in bold
ax.plot(daily_avg.index, daily_avg.values, 
        color='black', linewidth=2.5, label='Daily Average', zorder=10)

#ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Well Depth (m)')
ax.set_title('Wetland Well Depths Time Series')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% 4.0 Determine the kriging domain boundary

boundary = well_points.geometry.union_all().convex_hull.buffer(500)
boundary2 = well_points.geometry.union_all().buffer(1000).buffer(2000).buffer(-2000)

boundary2_gdf = gpd.GeoDataFrame(geometry=[boundary2])

#boundary2_gdf.to_file("D:/depressional_lidar/data/osbs/test_krig_domain.shp")

fig, ax = plt.subplots()
well_points.plot(ax=ax, color='red', markersize=20)

gpd.GeoSeries([boundary]).plot(ax=ax, facecolor='none', edgecolor='blue')
gpd.GeoSeries([boundary2]).plot(ax=ax, facecolor='none', edgecolor='violet')
plt.show()

# %% 5.0 Generate synthetic "conditioning points" based on pond surfaces

pond_points = gpd.read_file(pond_conditioning_pts_path)
pond_points.to_crs(crs=well_points.crs, inplace=True)
pond_ts = pd.DataFrame({
    'wetland_id': pond_points['wetland_id'].to_numpy(),
    'date': '2023-04-16',
    'well_depth_m': 0.1,
    'flag': 0
})

full_points = pd.concat([well_points, pond_points], axis=0)
full_ts = pd.concat([well_ts, pond_ts], axis=0)

# %% 6.0 Run Kriging on Depth Scenarios

well_array_variogram_fit = WellArray(
    well_pts=well_points,
    well_ts=well_ts,
    begin='2021-01-01',
    end='2027-01-01',
    percentile=50
)

wtd_variogram_fit = WTDSurface(
    well_array=well_array_variogram_fit,
    krig_params={
        'variogram_model': 'linear',
        'variogram_parameters': None,
        'n_lags': 7,
    },
    coarse_grid_dims=(1_000, 1_000),
    boundary=boundary2,
    plot_variogram=True
)

variogram_params = wtd_variogram_fit.okr_result['variogram_model_parameters'].tolist()
print(variogram_params)

well_array_med = WellArray(
    well_pts=full_points,
    well_ts=full_ts,
    begin='2021-01-01',
    end='2027-01-01', 
    percentile=50,
)

wtd_surface_med = WTDSurface(
    well_array=well_array_med,
    krig_params={
        'variogram_model': 'linear',
        'variogram_parameters': variogram_params,
        'n_lags': 7,
    },
    coarse_grid_dims=(1_000, 1_000),
    boundary=boundary2,
    plot_variogram=True
)

wtd_surface_med.plot_masked_result(sigma_threshold=2)

weights = wtd_variogram_fit.okr_result['weights'] #BUG ????

# %% 7.0 Write the results

x_flat = wtd_surface_med._x_grid.ravel().astype(np.float32)
y_flat = wtd_surface_med._y_grid.ravel().astype(np.float32)

coords_df = pd.DataFrame({'x': x_flat, 'y': y_flat})

wells = wtd_surface_med.well_array.well_pts.copy()
wells['x'] = wells.geometry.x
wells['y'] = wells.geometry.y

wells_df = wells[['wetland_id', 'z_dem', 'x', 'y']]
print(wells_df)

wetland_ids = wells_df['wetland_id']

out_dir = f"D:/depressional_lidar/data/osbs/out_data/well_wse_interpolations"
file_suffix = f"optimized_model_pond_conditioned"

# %% 7.1 Simple raster of median surface

wtd_surface_med.write_masked_tif(
    out_path=f'{out_dir}/interpolated_median_WSE_{file_suffix}.tif',
    sigma_threshold=2.5,
    crs='EPSG:26917'
)

# %% 7.2 Write excel

weights_df = pd.DataFrame(weights, columns=wetland_ids)
# out_path = f"{out_dir}/kriging_weights_{file_suffix}.xlsx"

# with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#     weights_df.to_excel(writer, sheet_name="weights", index=False)
#     wells_df.to_excel(writer, sheet_name="wells", index=False)
#     coords_df.to_excel(writer, sheet_name="coords", index=False)

# %% 7.3 HDF5

with pd.HDFStore(f"{out_dir}/kriging_weights_{file_suffix}.h5", mode="w") as store:
    store.put("weights", weights_df, format="fixed")
    store.put("grid_coords", pd.DataFrame({"x": x_flat, "y": y_flat}), format="fixed")

metadata = {
    "description": "Ordinary Kriging weights, optimized fitted linear model",
    "weights_shape": list(weights.shape),
    "wetland_ids": wetland_ids.to_list(),
    "grid_shape": list(wtd_surface_med._x_grid.shape),
    "crs": "EPSG:26917",
    "notes": [
        "Rows correspond to flattened kriging grid cells in row-major order.",
        "Columns correspond to wells in the order listed in well_ids.",
    ]
}

import json
with open(f"{out_dir}/kriging_weights_metadata_{file_suffix}.json", "w") as f:
    json.dump(metadata, f, indent=2)
# %%
