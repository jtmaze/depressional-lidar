# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import rasterio
from rasterio.warp import reproject, Resampling

median_result_path = 'D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/interpolated_median_WSE_optimized_model.tif'
weights_path = 'D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/kriging_weights_optimized_fit.h5'
well_data_path = 'D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
meta_path = f"D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/kriging_weights_metadata_optimized.json"

dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_USGS.tif'
#out_path = 'D:/depressional_lidar/data/bradford/out_data/kriging_inundation_optimized.tif'

# %% 1.5 Read DEM and print shape

with rasterio.open(dem_path) as src:
    dem = src.read(1)
    print(f"DEM shape: {dem.shape}")

# %% 2.0 Look at the kriging output from median WSE map

# with rasterio.open(dem_path) as dem_src:
#     dem = dem_src.read(1)
#     dem_profile = dem_src.profile.copy()

#     # Reproject kriging WSE (band 1) to DEM grid
#     wse = np.empty_like(dem, dtype=np.float32)

#     with rasterio.open(median_result_path) as krig_src:
#         reproject(
#             source=krig_src.read(1),
#             destination=wse,
#             src_transform=krig_src.transform,
#             src_crs=krig_src.crs,
#             dst_transform=dem_src.transform,
#             dst_crs=dem_src.crs,
#             resampling=Resampling.bilinear,
#             src_nodata=krig_src.nodata,
#             dst_nodata=np.nan,
#         )


# %% 3.0 Read the kriging weights from hdf5 and apply it to well data

well_data = pd.read_csv(well_data_path)

with pd.HDFStore(weights_path, mode="r") as store:
    weights_df = store["weights"]
    grid_coords = store["grid_coords"]

with open(meta_path, "r") as f:
    metadata = json.load(f)
print(metadata)

dims = metadata["grid_shape"][0] # equal columns and rows
nrows = dims
ncols = dims

# %% 3.1 Filter well_data for wells in weights and compute median by wetland_id

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'z_dem', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)

well_ids_in_weights = weights_df.columns.tolist()
well_data_filtered = well_data[well_data['wetland_id'].isin(well_ids_in_weights)]
median_by_wetland = well_data_filtered.groupby('wetland_id').agg({'well_depth_m': 'median'})

# Join well_points to get dem_z
well_points_dem = well_points[['wetland_id', 'z_dem']].drop_duplicates('wetland_id')
median_by_wetland = median_by_wetland.join(well_points_dem.set_index('wetland_id'), on='wetland_id')

# Calculate WSE
median_by_wetland['wse'] = median_by_wetland['well_depth_m'] + median_by_wetland['z_dem']

# %% 4.0 Apply weights to well data

print(median_by_wetland)

median_wse = median_by_wetland.reindex(well_ids_in_weights)["wse"]

print(median_wse)

surface_med_1d = weights_df.to_numpy() @ median_wse.to_numpy()

# Write median_wse and weights_df to Excel for collaborator review
# with pd.ExcelWriter('D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/crude_kriging_arrays.xlsx', engine='openpyxl') as writer:
#     pd.DataFrame(median_wse.to_numpy()).to_excel(writer, sheet_name='median_wse', header=False, index=False)
#     pd.DataFrame(weights_df.to_numpy()).to_excel(writer, sheet_name='weights', header=False, index=False)

wse_surface_med = surface_med_1d.reshape(dims, dims)


# %% 5.0 Plot alongside the median kriging result to test

x_grid = grid_coords["x"].to_numpy()
y_grid = grid_coords["y"].to_numpy()

# Read pykrige median raster on its native grid
with rasterio.open(median_result_path) as src:
    pykrige_med = src.read(1).astype(np.float32)
    nodata = src.nodata
    t = src.transform
    nrows_pk, ncols_pk = pykrige_med.shape
    pk_x = np.array([t.c + (j + 0.5) * t.a for j in range(ncols_pk)])
    pk_y = np.array([t.f + (i + 0.5) * t.e for i in range(nrows_pk)])
    pk_x_grid, pk_y_grid = np.meshgrid(pk_x, pk_y)
    if nodata is not None:
        pykrige_med[pykrige_med == nodata] = np.nan

well_points_plot = well_points.merge(
    median_by_wetland[["wse"]],
    left_on="wetland_id",
    right_index=True,
    how="inner"
)

# Shared color range
vmin = np.nanmin([np.nanmin(wse_surface_med), np.nanmin(pykrige_med)])
vmax = np.nanmax([np.nanmax(wse_surface_med), np.nanmax(pykrige_med)])

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)

# --- Left: weights-derived median ---
im0 = axes[0].pcolormesh(x_grid, y_grid, wse_surface_med, shading="auto", vmin=vmin, vmax=vmax)
well_points_plot.plot(ax=axes[0], column="wse", edgecolor="black", markersize=40, legend=False)
fig.colorbar(im0, ax=axes[0], label="WSE (m)")
axes[0].set_title("Weights × median WSE (timeseries workflow)")
axes[0].set_xlabel("Easting")
axes[0].set_ylabel("Northing")
axes[0].set_aspect("equal")

# --- Right: pykrige direct output ---
im1 = axes[1].pcolormesh(pk_x_grid, pk_y_grid, pykrige_med, shading="auto", vmin=vmin, vmax=vmax)
well_points_plot.plot(ax=axes[1], column="wse", edgecolor="black", markersize=40, legend=False)
fig.colorbar(im1, ax=axes[1], label="WSE (m)")
axes[1].set_title("pykrige direct output")
axes[1].set_xlabel("Easting")
axes[1].set_aspect("equal")

plt.suptitle("Median WSE comparison — weights vs. pykrige output", y=1.01)
plt.tight_layout()
plt.show()


# %%
