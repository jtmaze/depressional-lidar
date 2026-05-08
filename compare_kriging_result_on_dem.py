# %% 1.0 Libraries and file paths
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds

median_result_path = 'D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/interpolated_median_WSE_optimized_model.tif'
weights_path = 'D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/kriging_weights_optimized_fit.h5'
well_data_path = 'D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
meta_path = f"D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/kriging_weights_metadata_optimized.json"

dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_USGS.tif'
#out_path = 'D:/depressional_lidar/data/bradford/out_data/kriging_inundation_optimized.tif'
target_dem_resolution_m = 10.0
max_dem_cells = 8_000_000

# %% 1.5 Read DEM and print shape

with rasterio.open(dem_path) as src:
    print(f"DEM shape (native): ({src.height}, {src.width})")
    print(f"DEM resolution (native): ({abs(src.transform.a):.3f} m, {abs(src.transform.e):.3f} m)")

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

grid_shape = (1_000, 1_000)

nrows, ncols = grid_shape

# Use string IDs everywhere so joins/reindex operations are robust.
weights_df.columns = weights_df.columns.astype(str)
well_data["wetland_id"] = well_data["wetland_id"].astype(str)

# %% 3.1 Filter well_data for wells in weights and compute median by wetland_id

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'z_dem', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)
well_points["wetland_id"] = well_points["wetland_id"].astype(str)

well_ids_in_weights = weights_df.columns.tolist()
well_points_krig = well_points[well_points["wetland_id"].isin(well_ids_in_weights)].copy()

if well_points_krig.empty:
    raise ValueError("No well points matched the kriging weights wetland IDs.")

well_data_filtered = well_data[well_data['wetland_id'].isin(well_ids_in_weights)]
median_by_wetland = well_data_filtered.groupby('wetland_id').agg({
    'well_depth_m': lambda x: np.percentile(x.dropna().astype(np.float64), 10)
})
# Join well_points to get dem_z
well_points_dem = well_points_krig[['wetland_id', 'z_dem']].drop_duplicates('wetland_id')
median_by_wetland = median_by_wetland.join(well_points_dem.set_index('wetland_id'), on='wetland_id')

# Calculate WSE
median_by_wetland['wse'] = median_by_wetland['well_depth_m'] + median_by_wetland['z_dem']

# %% 4.0 Apply weights to well data

print(median_by_wetland)

median_wse = median_by_wetland.reindex(well_ids_in_weights)["wse"]

missing_wells = median_wse[median_wse.isna()].index.tolist()
if missing_wells:
    raise ValueError(
        f"Missing median WSE values for {len(missing_wells)} wells used by kriging weights. "
        f"Example IDs: {missing_wells[:5]}"
    )

print(median_wse)

surface_med_1d = weights_df.to_numpy() @ median_wse.to_numpy()

# Write median_wse and weights_df to Excel for collaborator review
# with pd.ExcelWriter('D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/crude_kriging_arrays.xlsx', engine='openpyxl') as writer:
#     pd.DataFrame(median_wse.to_numpy()).to_excel(writer, sheet_name='median_wse', header=False, index=False)
#     pd.DataFrame(weights_df.to_numpy()).to_excel(writer, sheet_name='weights', header=False, index=False)

wse_surface_med = surface_med_1d.reshape(nrows, ncols)


# %% 5.0 Plot alongside the median kriging result to test

# Read pykrige median raster on its native grid
with rasterio.open(median_result_path) as src:
    pykrige_med = src.read(1).astype(np.float32)
    nodata = src.nodata
    krig_transform = src.transform
    krig_crs = src.crs
    nrows_pk, ncols_pk = pykrige_med.shape
    pk_x = np.array([krig_transform.c + (j + 0.5) * krig_transform.a for j in range(ncols_pk)])
    pk_y = np.array([krig_transform.f + (i + 0.5) * krig_transform.e for i in range(nrows_pk)])
    pk_x_grid, pk_y_grid = np.meshgrid(pk_x, pk_y)
    if nodata is not None:
        pykrige_med[pykrige_med == nodata] = np.nan

if wse_surface_med.shape != pykrige_med.shape:
    if wse_surface_med.size == pykrige_med.size:
        wse_surface_med = wse_surface_med.reshape(pykrige_med.shape)
    else:
        raise ValueError(
            f"Weights surface shape {wse_surface_med.shape} cannot be matched to "
            f"pykrige raster shape {pykrige_med.shape}."
        )

# The weights grid is stored south->north by construction, while GeoTIFF rows are north->south.
# Flip once so the weights surface is in the same geospatial orientation as the pykrige raster.
wse_surface_med_geo = np.flipud(wse_surface_med)

# Use pykrige's finite footprint as the shared uncertainty mask for both surfaces.
pykrige_uncertainty_mask_native = np.isfinite(pykrige_med)
wse_surface_med_masked = np.where(pykrige_uncertainty_mask_native, wse_surface_med_geo, np.nan)
pykrige_med_masked = np.where(pykrige_uncertainty_mask_native, pykrige_med, np.nan)

well_points_plot = well_points_krig.merge(
    median_by_wetland[["wse"]],
    left_on="wetland_id",
    right_index=True,
    how="inner"
)

# Shared color range
vmin = np.nanmin([np.nanmin(wse_surface_med_masked), np.nanmin(pykrige_med_masked)])
vmax = np.nanmax([np.nanmax(wse_surface_med_masked), np.nanmax(pykrige_med_masked)])

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)

# --- Left: weights-derived median ---
im0 = axes[0].pcolormesh(pk_x_grid, pk_y_grid, wse_surface_med_masked, shading="auto", vmin=vmin, vmax=vmax)
well_points_plot.plot(ax=axes[0], column="wse", edgecolor="black", markersize=40, legend=False)
fig.colorbar(im0, ax=axes[0], label="WSE (m)")
axes[0].set_title("Weights × median WSE (timeseries workflow)")
axes[0].set_xlabel("Easting")
axes[0].set_ylabel("Northing")
axes[0].set_aspect("equal")

# --- Right: pykrige direct output ---
im1 = axes[1].pcolormesh(pk_x_grid, pk_y_grid, pykrige_med_masked, shading="auto", vmin=vmin, vmax=vmax)
well_points_plot.plot(ax=axes[1], column="wse", edgecolor="black", markersize=40, legend=False)
fig.colorbar(im1, ax=axes[1], label="WSE (m)")
axes[1].set_title("pykrige direct output")
axes[1].set_xlabel("Easting")
axes[1].set_aspect("equal")

plt.suptitle("Median WSE comparison — weights vs. pykrige output", y=1.01)
plt.tight_layout()
plt.show()


# %% 6.0 Compare inundation on DEM (weights workflow vs pykrige output)

with rasterio.open(dem_path) as dem_src:
    src_h, src_w = dem_src.height, dem_src.width
    src_res_x = abs(dem_src.transform.a)
    src_res_y = abs(dem_src.transform.e)

    if target_dem_resolution_m > max(src_res_x, src_res_y):
        out_w = max(1, int(np.ceil(src_w * (src_res_x / target_dem_resolution_m))))
        out_h = max(1, int(np.ceil(src_h * (src_res_y / target_dem_resolution_m))))
    else:
        out_w = src_w
        out_h = src_h

    if (out_h * out_w) > max_dem_cells:
        safety_scale = np.sqrt((out_h * out_w) / max_dem_cells)
        out_h = max(1, int(np.ceil(out_h / safety_scale)))
        out_w = max(1, int(np.ceil(out_w / safety_scale)))

    dem = dem_src.read(
        1,
        out_shape=(out_h, out_w),
        resampling=Resampling.average,
    ).astype(np.float32)

    dem_transform = dem_src.transform * dem_src.transform.scale(src_w / out_w, src_h / out_h)
    dem_crs = dem_src.crs
    dem_nodata = dem_src.nodata

print(f"DEM shape (analysis): {dem.shape}")
print(f"DEM target resolution (m): {target_dem_resolution_m}")
print(f"DEM max cell budget: {max_dem_cells}")

if dem_nodata is not None:
    dem = np.where(dem == dem_nodata, np.nan, dem)

dem_valid_mask = np.isfinite(dem)

weights_wse_dem = np.full(dem.shape, np.nan, dtype=np.float32)
pykrige_wse_dem = np.full(dem.shape, np.nan, dtype=np.float32)

reproject(
    source=wse_surface_med_masked.astype(np.float32),
    destination=weights_wse_dem,
    src_transform=krig_transform,
    src_crs=krig_crs,
    dst_transform=dem_transform,
    dst_crs=dem_crs,
    resampling=Resampling.bilinear,
    dst_nodata=np.nan,
)

reproject(
    source=pykrige_med_masked.astype(np.float32),
    destination=pykrige_wse_dem,
    src_transform=krig_transform,
    src_crs=krig_crs,
    dst_transform=dem_transform,
    dst_crs=dem_crs,
    resampling=Resampling.bilinear,
    dst_nodata=np.nan,
)

# Apply the same pykrige-derived uncertainty footprint to both inundation maps.
shared_uncertainty_mask_dem = dem_valid_mask & np.isfinite(pykrige_wse_dem) & np.isfinite(weights_wse_dem)
weights_wse_dem = np.where(shared_uncertainty_mask_dem, weights_wse_dem, np.nan)
pykrige_wse_dem = np.where(shared_uncertainty_mask_dem, pykrige_wse_dem, np.nan)

inund_weights = np.where(shared_uncertainty_mask_dem, (weights_wse_dem > dem).astype(np.float32), np.nan)
inund_pykrige = np.where(shared_uncertainty_mask_dem, (pykrige_wse_dem > dem).astype(np.float32), np.nan)

weights_pct = float(np.nanmean(inund_weights) * 100)
pykrige_pct = float(np.nanmean(inund_pykrige) * 100)

print(f"Weights workflow inundation: {weights_pct:.2f}%")
print(f"Pykrige-output inundation: {pykrige_pct:.2f}%")

left, bottom, right, top = array_bounds(dem.shape[0], dem.shape[1], dem_transform)
dem_extent = (left, right, bottom, top)

well_points_dem = well_points_krig.to_crs(dem_crs)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

im0 = axes[0].imshow(
    inund_weights,
    cmap="Blues",
    vmin=0,
    vmax=1,
    origin="upper",
    extent=dem_extent,
)
axes[0].set_title(f"Weights workflow\nInundated: {weights_pct:.1f}%")
axes[0].scatter(well_points_dem.geometry.x, well_points_dem.geometry.y, color="red", s=8, alpha=0.7)

im1 = axes[1].imshow(
    inund_pykrige,
    cmap="Blues",
    vmin=0,
    vmax=1,
    origin="upper",
    extent=dem_extent,
)
axes[1].set_title(f"Pykrige direct output\nInundated: {pykrige_pct:.1f}%")
axes[1].scatter(well_points_dem.geometry.x, well_points_dem.geometry.y, color="red", s=8, alpha=0.7)

axes[0].set_xlabel("Easting")
axes[0].set_ylabel("Northing")
for ax in axes:
    ax.set_aspect("equal")

# Remove tick marks and axis labels from the last (right-hand) subplot
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_xlabel("")
axes[1].set_ylabel("")

plt.suptitle("Inundation comparison on DEM grid", y=1.02)
plt.show()

# %%
