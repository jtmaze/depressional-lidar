# %% 1.0 Libraries and file paths

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import rasterio as rio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import Resampling, reproject
from matplotlib.colors import TwoSlopeNorm

data_dir = 'D:/depressional_lidar/data/bradford'

tgt_res = 3 # meter
tgt_var = 'wse_m_sd2' # 'wse_m_sd2' of 'wse_m_mean'

krig_res_path = f"{data_dir}/out_data/well_wse_interpolations/kriging_weights_optimized_model_dummy_gauges.h5"
well_ts_path = f"{data_dir}/in_data/stage_data/bradford_well_data_long_gapfilled.csv"
well_points_path = "D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp"
dem_path = f"{data_dir}/in_data/bradford_DEM_cleaned_USGS.tif"
gauges_points_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/dummy_stream_gauges.shp'
synthetic_stream_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/synthetic_stream_timeseries.csv'
boundary_path = 'D:/depressional_lidar/data/bradford/bradford_krig_domain.shp'

out_dir = f'{data_dir}/out_data/bradford_wtd_maps/'
# %% 2.0 Read kriging weights and grid info

weights = pd.read_hdf(krig_res_path, "weights")
grid_coords = pd.read_hdf(krig_res_path, "grid_coords")

weights.columns = weights.columns.astype(str)

x_grid = np.sort(grid_coords["x"].unique())
y_grid = np.sort(grid_coords["y"].unique())

nx = len(x_grid)
ny = len(y_grid)

krig_transform = from_bounds(
    x_grid.min(), y_grid.min(),
    x_grid.max(), y_grid.max(),
    nx, ny
)

# %% 3.0 Read well timeseries and synthetic stream timeseries

well_ts = pd.read_csv(well_ts_path)
well_ts = well_ts.rename(
    columns={
        "well_id": "wetland_id",
        "water_depth_m": "well_depth_m",
    }
)

well_ts = well_ts[["date", "wetland_id", "well_depth_m"]]
well_ts["date"] = pd.to_datetime(well_ts["date"]).dt.normalize()
well_ts["wetland_id"] = well_ts["wetland_id"].astype(str)

stream_gauges = gpd.read_file(gauges_points_path)

stream_gauges['wetland_id'] = 'stream_' + stream_gauges['id'].astype('str')
gauge_ids = stream_gauges['wetland_id'].unique()
synthetic_gauges = pd.read_csv(synthetic_stream_path)
synthetic_gauges['date'] = pd.to_datetime(synthetic_gauges['date'])

gauge_ts = pd.concat([
    synthetic_gauges.assign(wetland_id=gid) 
    for gid in gauge_ids
], ignore_index=True)

gauge_ts.rename(columns={'depth': 'well_depth_m'}, inplace=True)
gauge_ts['flag'] = 0 

well_ts = pd.concat([well_ts, gauge_ts])

# %% 4.0 Generate summary tables from the well/stream timeseries

sd2_per_id = (
    well_ts.groupby('wetland_id')['well_depth_m']
    .agg(['mean', 'std'])
    .assign(mean_plus_2sd=lambda df: df['mean'] + 2 * df['std'])
    .reset_index()
    .rename(columns={'mean': 'mean_well_depth_m', 'std': 'sd_well_depth_m', 'mean_plus_2sd': 'mean_plus_2sd_well_depth_m'})
    .sort_values('mean_plus_2sd_well_depth_m', ascending=False)
)

print(sd2_per_id)

# %% 5.0 Read the DEM

with rio.open(dem_path) as src:
    src_h, src_w = src.height, src.width
    src_res = abs(src.transform.a)

    scale = src_res / tgt_res
    out_h = max(1, int(np.ceil(src_h * scale)))
    out_w = max(1, int(np.ceil(src_w * scale)))

    dem = src.read(
        1,
        out_shape=(out_h, out_w),
        resampling=Resampling.average
    ).astype(np.float32)

    dem_transform = src.transform * src.transform.scale(src_w / out_w, src_h / out_h)
    dem_crs = src.crs

    if src.nodata is not None:
        dem[dem == src.nodata] = np.nan

boundary = gpd.read_file(boundary_path)

boundary_mask = geometry_mask(
    boundary.geometry,
    out_shape=dem.shape,
    transform=dem_transform,
    invert=True,
)

# %% 6.0 Prepare the points with z_dem values

well_points = (
    gpd.read_file(well_points_path)[["wetland_id", "type", "geometry", "z_dem", "site"]]
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)

basin_13_ids = [
    "13_263", "13_267", "13_271", "13_410", "13_274",
    "Donor_wetland", "Receiver_wetland"
]

well_points = well_points[~well_points["wetland_id"].isin(basin_13_ids)]
well_points["wetland_id"] = well_points["wetland_id"].astype(str)

stream_gauges.to_crs(well_points.crs, inplace=True)
all_points = gpd.GeoDataFrame(pd.concat([stream_gauges, well_points], axis=0), crs=well_points.crs)

wse = sd2_per_id.merge(
    all_points[["wetland_id", "z_dem"]],
    on="wetland_id",
    how="inner"
)

wse["wse_m_mean"] = wse["mean_well_depth_m"] + wse["z_dem"]
wse['wse_m_sd2'] = wse['mean_plus_2sd_well_depth_m'] + wse['z_dem']

well_wse = (
    wse.set_index("wetland_id")[tgt_var]
    .reindex(weights.columns)
)

print(well_wse)

# %% 7.0 Apply kriging weights and plot water table depth map

left, bottom, right, top = array_bounds(dem.shape[0], dem.shape[1], dem_transform)
extent = (left, right, bottom, top)

wse_flat = well_wse.to_numpy() @ weights.to_numpy().T

# Reshape to pykrige grid shape.
# PyKrige grid output is y by x.
wse_grid = wse_flat.reshape(ny, nx).astype(np.float32)

# Flip so row 0 is north/top for rasterio.
wse_grid = np.flipud(wse_grid)

# Reproject WSE grid to DEM grid.
wse_on_dem = np.full(dem.shape, np.nan, dtype=np.float32)

reproject(
    source=wse_grid,
    destination=wse_on_dem,
    src_transform=krig_transform,
    src_crs=dem_crs,
    dst_transform=dem_transform,
    dst_crs=dem_crs,
    resampling=Resampling.bilinear,
    dst_nodata=np.nan,
)

valid = boundary_mask & np.isfinite(dem) & np.isfinite(wse_on_dem)
wtd_m = np.where(valid, dem - wse_on_dem, np.nan).astype(np.float32)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(
    wtd_m,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.5, vmax=2.5),
    origin="upper",
    extent=extent,
    interpolation="nearest",
)

ax.plot(*boundary.geometry.iloc[0].exterior.xy, color="black", linewidth=2.0, zorder=4)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Water table depth (m; dem - wse)")
if tgt_var == "wse_m_mean":
    title = "Mean Values"
elif tgt_var == "wse_m_sd2":
    title = "Mean Vals +2SD"

ax.set_title(f"WTD ({title})")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %% 8.0 Write the output as a geotiff

out_path = out_dir + tgt_var + '.tiff'

with rio.open(
    out_path,
    "w",
    driver="GTiff",
    height=wtd_m.shape[0],
    width=wtd_m.shape[1],
    count=1,
    dtype="float32",
    crs=dem_crs,
    transform=dem_transform,
    nodata=np.nan,
) as dst:
    dst.write(wtd_m, 1)

print(f"Written: {out_path}")

# %%
