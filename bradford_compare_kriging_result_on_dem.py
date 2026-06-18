# %% 1.0 Libraries and file paths
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds

median_result_path = 'D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/interpolated_median_WSE_optimized_model_wetlands_only.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
gauges_points_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/dummy_stream_gauges.shp'
dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_USGS.tif'
boundary_path = 'D:/depressional_lidar/data/bradford/bradford_krig_domain.shp'
target_dem_resolution_m = 1.5

# %% 2.0 Read median kriging WSE raster

with rasterio.open(median_result_path) as src:
    wse = src.read(1).astype(np.float32)
    if src.nodata is not None:
        wse[wse == src.nodata] = np.nan
    krig_transform = src.transform
    krig_crs = src.crs

# %% 3.0 Read DEM downsampled to target resolution

with rasterio.open(dem_path) as src:
    src_h, src_w = src.height, src.width
    src_res = abs(src.transform.a)
    scale = src_res / target_dem_resolution_m
    out_h = max(1, int(np.ceil(src_h * scale)))
    out_w = max(1, int(np.ceil(src_w * scale)))

    dem = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.average).astype(np.float32)
    dem_transform = src.transform * src.transform.scale(src_w / out_w, src_h / out_h)
    dem_crs = src.crs

    if src.nodata is not None:
        dem[dem == src.nodata] = np.nan

print(f"DEM shape at {target_dem_resolution_m} m: {dem.shape}")

# %% 3.1 Build the kriging boundary from the well points

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'geometry', 'z_dem', 'site']]
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)

basin_13_ids = ['13_263', '13_267', '13_271', '13_410', '13_274', 'Donor_wetland', 'Receiver_wetland']
well_points = well_points[~well_points['wetland_id'].isin(basin_13_ids)]

boundary = gpd.read_file(boundary_path)
boundary = boundary.iloc[0].geometry

boundary_mask = geometry_mask(
    [boundary],
    out_shape=dem.shape,
    transform=dem_transform,
    invert=True,
)

dem_clipped = np.where(boundary_mask, dem, np.nan)

# %% 4.0 Reproject WSE to DEM grid

wse_on_dem = np.full(dem.shape, np.nan, dtype=np.float32)
reproject(
    source=wse,
    destination=wse_on_dem,
    src_transform=krig_transform,
    src_crs=krig_crs,
    dst_transform=dem_transform,
    dst_crs=dem_crs,
    resampling=Resampling.bilinear,
    dst_nodata=np.nan,
)

# %% 5.0 Compute inundation

valid = np.isfinite(dem) & np.isfinite(wse_on_dem) & boundary_mask
inundation = np.where(valid, (wse_on_dem > dem).astype(np.float32), np.nan)
inund_pct = float(np.nanmean(inundation) * 100)
print(f"Inundated: {inund_pct:.2f}%")

# %% 6.0 Plot inundation overlaid on DEM

stream_gauges = gpd.read_file(gauges_points_path)

stream_gauges = stream_gauges.to_crs(well_points.crs)


left, bottom, right, top = array_bounds(dem.shape[0], dem.shape[1], dem_transform)
extent = (left, right, bottom, top)

fig, ax = plt.subplots(figsize=(10, 10))

water_hex = "#009FFC"  # strong blue

inundation_cmap = ListedColormap([
    (0, 0, 0, 0),                    # transparent for non-inundated
    to_rgba(water_hex, alpha=0.85)   # blue with transparency
])
dem_im = ax.imshow(dem_clipped, cmap='Greys_r', origin='upper', extent=extent, alpha=1.0, vmin=35, vmax=50)
ax.imshow(inundation, cmap=inundation_cmap, vmin=0, vmax=1, origin='upper', extent=extent, interpolation='nearest')
ax.plot(*boundary.exterior.xy, color='red', linewidth=2.5, zorder=4)
ax.scatter(well_points.geometry.x, well_points.geometry.y, color='red', s=80, zorder=5, edgecolors='black')
#ax.scatter(stream_gauges.geometry.x, stream_gauges.geometry.y, color='orange', s=40, zorder=6, edgecolors='black')
fig.colorbar(dem_im, ax=ax, fraction=0.046, pad=0.04, label='DEM elevation (m)')
ax.set_title(f"Median WSE inundation on DEM — {inund_pct:.1f}% inundated")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
minx, miny, maxx, maxy = boundary.bounds
pad = 0.05 * max(maxx - minx, maxy - miny)
ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()

# %%
