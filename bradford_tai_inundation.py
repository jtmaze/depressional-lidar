# %% 1.0 File paths and libraries 
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds, array_bounds


from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba

data_dir = "D:/depressional_lidar/data/bradford"
kriging_method = 'stream_conditioning'

krig_res_path = f"{data_dir}/out_data/well_wse_interpolations/kriging_weights_optimized_model_{kriging_method}.h5"
well_ts_path = f"{data_dir}/in_data/stage_data/bradford_well_data_gapfilled.csv"
well_points_path = "D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp"
dem_path = f"{data_dir}/in_data/bradford_DEM_cleaned_USGS.tif"
gauges_points_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/dummy_stream_gauges.shp'
synthetic_stream_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/synthetic_stream_timeseries.csv'
boundary_path = 'D:/depressional_lidar/data/bradford/bradford_krig_domain.shp'

target_dem_resolution_m = 10

dates_to_plot = [
    "2023-02-01",
    "2024-04-15",
    "2024-06-20",
]

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

# %% 3.0 Read well timeseries and well points

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

# %% 4.0 Append synthetic stream gauge timeseries to wetland timeseries if used in kriging

if kriging_method == 'stream_conditioning':

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

else:
    well_ts = well_ts


# %% 5.0 Read the DEM into target resolution

with rasterio.open(dem_path) as src:
    src_h, src_w = src.height, src.width
    src_res = abs(src.transform.a)

    scale = src_res / target_dem_resolution_m
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

print(f"DEM shape at {target_dem_resolution_m} m: {dem.shape}")

# %% 3.2 Read the well points

dates_to_plot = pd.to_datetime(dates_to_plot).normalize()

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

boundary = gpd.read_file(boundary_path)

boundary_mask = geometry_mask(
    boundary.geometry,
    out_shape=dem.shape,
    transform=dem_transform,
    invert=True,
)

stream_gauges.to_crs(well_points.crs, inplace=True)
all_points = gpd.GeoDataFrame(pd.concat([stream_gauges, well_points], axis=0), crs=well_points.crs)

# %% Plot the average timeseries for all wells

avg_ts = (
    well_ts
    .groupby("date", as_index=False)["well_depth_m"]
    .mean()
    .sort_values("date")
)

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(
    avg_ts["date"],
    avg_ts["well_depth_m"],
    color="blue",
    linewidth=2.5,
    alpha=1,
    label="Daily mean"
)

for target_date in dates_to_plot:
    ax.axvline(target_date, color="red", linestyle="--", linewidth=1.5, alpha=0.85)

ax.set_title("Average well water depth across Bradford wells")
ax.set_xlabel("Date")
ax.set_ylabel("Water depth (m)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

plt.tight_layout()
plt.show()


# %% 6.0 Prepare WSE table

wse = well_ts.merge(
    all_points[["wetland_id", "z_dem"]],
    on="wetland_id",
    how="inner"
)

wse["wse_m"] = wse["well_depth_m"] + wse["z_dem"]

wse_wide = (
    wse
    .pivot_table(index="date", columns="wetland_id", values="wse_m", aggfunc="mean")
    .reindex(columns=weights.columns)
)

# %% 7.0 Apply weights and plot inundation for each date

left, bottom, right, top = array_bounds(dem.shape[0], dem.shape[1], dem_transform)
extent = (left, right, bottom, top)

water_hex = "#009FFC"

inundation_cmap = ListedColormap([
    (0, 0, 0, 0),
    to_rgba(water_hex, alpha=0.85),
])

for target_date in dates_to_plot:

    if target_date not in wse_wide.index:
        print(f"Skipping {target_date.date()}: date not found in timeseries.")
        continue

    well_wse = wse_wide.loc[target_date]

    if well_wse.isna().any():
        missing = well_wse.index[well_wse.isna()].tolist()
        print(f"Skipping {target_date.date()}: missing WSE for {missing}")
        continue

    # Apply kriging weights.
    # Result is flattened over all kriging grid cells.
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

    # Compute inundation.
    valid = np.isfinite(dem) & np.isfinite(wse_on_dem) & boundary_mask
    inundation = np.where(valid, (wse_on_dem > dem).astype(np.float32), np.nan)

    inund_pct = float(np.nanmean(inundation) * 100)
    print(f"{target_date.date()} inundated: {inund_pct:.2f}%")

    # Plot.
    fig, ax = plt.subplots(figsize=(10, 10))

    dem_im = ax.imshow(
        dem,
        cmap="Greys_r",
        origin="upper",
        extent=extent,
        alpha=1.0,
        vmin=35,
        vmax=50,
    )

    ax.imshow(
        inundation,
        cmap=inundation_cmap,
        vmin=0,
        vmax=1,
        origin="upper",
        extent=extent,
        interpolation="nearest",
    )

    ax.plot(*boundary.geometry.iloc[0].exterior.xy, color="red", linewidth=2.5, zorder=4)

    well_points.to_crs(dem_crs).plot(
    ax=ax,
    color="red",
    markersize=40,
    edgecolor="white",
    zorder=5,
)

    stream_gauges.to_crs(dem_crs).plot(
        ax=ax,
        color="orange",
        markersize=40,
        edgecolor="black",
        zorder=5,
    )

    fig.colorbar(
        dem_im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="DEM elevation (m)",
    )

    ax.set_title(
        f"WSE inundation on DEM — {target_date.date()} — {inund_pct:.1f}% inundated"
    )
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 0.05 * max(maxx - minx, maxy - miny)

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


# %% 8.0 Calculate percent time inundated for each cell WY 2024

tgt_wy = '2024'

wy_start = pd.Timestamp(f"{int(tgt_wy) - 1}-10-01")
wy_end   = pd.Timestamp(f"{tgt_wy}-09-30")

wy_dates = wse_wide.index[(wse_wide.index >= wy_start) & (wse_wide.index <= wy_end)]
wy_wse   = wse_wide.loc[wy_dates].dropna()

print(f"WY{tgt_wy}: {len(wy_wse)} complete days")

inundation_stack = np.zeros(dem.shape, dtype=np.float32)
count = 0

for date in wy_wse.index:
    well_wse = wy_wse.loc[date]

    wse_flat = well_wse.to_numpy() @ weights.to_numpy().T
    wse_grid = wse_flat.reshape(ny, nx).astype(np.float32)
    wse_grid = np.flipud(wse_grid)

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

    valid = np.isfinite(dem) & np.isfinite(wse_on_dem) & boundary_mask
    inundated = np.where(valid, (wse_on_dem > dem).astype(np.float32), np.nan)

    inundation_stack = np.where(
        np.isfinite(inundated),
        inundation_stack + inundated,
        inundation_stack
    )
    count += 1

inund_freq = np.where(boundary_mask & np.isfinite(dem), inundation_stack / count * 100, np.nan)

# %% 8.1 Plot inundation frequency map

fig, ax = plt.subplots(figsize=(10, 10))

freq_cmap = LinearSegmentedColormap.from_list(
    "brown_white_blue",
    ["#8c510a", "#f7f7f7", "#2166ac"],
)

freq_im = ax.imshow(
    inund_freq,
    cmap=freq_cmap,
    vmin=0,
    vmax=100,
    origin="upper",
    extent=extent,
    interpolation="nearest",
)

ax.plot(*boundary.geometry.iloc[0].exterior.xy, color="black", linewidth=2.5, zorder=4)

well_points.to_crs(dem_crs).plot(
    ax=ax,
    color="red",
    markersize=40,
    edgecolor="white",
    zorder=5,
)

# stream_gauges.to_crs(dem_crs).plot(
#     ax=ax,
#     color="orange",
#     markersize=40,
#     edgecolor="black",
#     zorder=5,
# )

fig.colorbar(
    freq_im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
    label="% of days inundated",
)

ax.set_title(f"Inundation frequency — WY{tgt_wy}")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")

minx, miny, maxx, maxy = boundary.total_bounds
pad = 0.05 * max(maxx - minx, maxy - miny)

ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)
ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %% 9.0 Map the Terrestrial-Aquatic Interface (TAI) — WY 2024
# TAI = cells that are intermittently inundated (neither always dry nor always wet).
# Mask out fully dry (0%) and fully wet (100%) cells; map the gradient between.

tai = np.where(
    (inund_freq > 0) & (inund_freq < 100),
    inund_freq,
    np.nan,
)

fig, ax = plt.subplots(figsize=(10, 10))

# Always-dry cells (inund_freq == 0) → white
always_dry = np.where(
    boundary_mask & np.isfinite(dem) & (inund_freq == 0),
    1.0,
    np.nan,
)
ax.imshow(
    always_dry,
    cmap=ListedColormap(["white"]),
    vmin=0,
    vmax=1,
    origin="upper",
    extent=extent,
    interpolation="nearest",
)

# Always-wet cells (inund_freq == 100) → dark grey
always_wet = np.where(
    boundary_mask & np.isfinite(dem) & (inund_freq == 100),
    1.0,
    np.nan,
)
ax.imshow(
    always_wet,
    cmap=ListedColormap(["#797979"]),
    vmin=0,
    vmax=1,
    origin="upper",
    extent=extent,
    interpolation="nearest",
)

tai_im = ax.imshow(
    tai,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=100,
    origin="upper",
    extent=extent,
    interpolation="nearest",
)

ax.plot(*boundary.geometry.iloc[0].exterior.xy, color="black", linewidth=2.5, zorder=4)

well_points.to_crs(dem_crs).plot(
    ax=ax,
    color="red",
    markersize=40,
    edgecolor="white",
    zorder=5,
)

# stream_gauges.to_crs(dem_crs).plot(
#     ax=ax,
#     color="orange",
#     markersize=50,
#     edgecolor="black",
#     zorder=5,
# )

fig.colorbar(
    tai_im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
    label="% of days inundated (TAI cells only)",
)

ax.set_title(f"Terrestrial-Aquatic Interface — WY{tgt_wy}")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")

minx, miny, maxx, maxy = boundary.total_bounds
pad = 0.05 * max(maxx - minx, maxy - miny)

ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)
ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %% 10.0 Map wet-to-dry transition count — WY 2024
# A transition is counted when a cell is inundated on day t-1 and dry on day t.

transitions = np.zeros(dem.shape, dtype=np.float32)
prev_inundated = None

for date in wy_wse.index:
    well_wse = wy_wse.loc[date]

    wse_flat = well_wse.to_numpy() @ weights.to_numpy().T
    wse_grid = wse_flat.reshape(ny, nx).astype(np.float32)
    wse_grid = np.flipud(wse_grid)

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

    valid = np.isfinite(dem) & np.isfinite(wse_on_dem) & boundary_mask
    inundated = np.where(valid, (wse_on_dem > dem).astype(np.float32), np.nan)

    if prev_inundated is not None:
        # Wet-to-dry: previous day inundated (1), today dry (0), both valid
        wet_to_dry = (
            np.isfinite(prev_inundated) & np.isfinite(inundated) &
            (prev_inundated == 1) & (inundated == 0)
        )
        transitions = np.where(wet_to_dry, transitions + 1, transitions)

    prev_inundated = inundated

transitions = np.where(boundary_mask & np.isfinite(dem), transitions, np.nan)

# %% 10.1 Plot wet-to-dry transition map

fig, ax = plt.subplots(figsize=(10, 10))

# Consistently dry (inund_freq == 0) → white
ax.imshow(
    np.where(boundary_mask & np.isfinite(dem) & (inund_freq == 0), 1.0, np.nan),
    cmap=ListedColormap(["white"]),
    vmin=0,
    vmax=1,
    origin="upper",
    extent=extent,
    interpolation="nearest",
)

# Consistently inundated (inund_freq == 100) → dark grey
ax.imshow(
    np.where(boundary_mask & np.isfinite(dem) & (inund_freq == 100), 1.0, np.nan),
    cmap=ListedColormap(["#797979"]),
    vmin=0,
    vmax=1,
    origin="upper",
    extent=extent,
    interpolation="nearest",
)

# Mask always-dry and always-wet cells so the base layers show through.
trans_masked = np.where(
    (inund_freq == 0) | (inund_freq == 100),
    np.nan,
    transitions,
)

trans_im = ax.imshow(
    trans_masked,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=np.nanpercentile(transitions[np.isfinite(transitions)], 98),
    origin="upper",
    extent=extent,
    interpolation="nearest",
)

ax.plot(*boundary.geometry.iloc[0].exterior.xy, color="black", linewidth=2.5, zorder=4)

well_points.to_crs(dem_crs).plot(
    ax=ax,
    color="black",
    markersize=120,
    edgecolor="white",
    zorder=5,
)

stream_gauges.to_crs(dem_crs).plot(
    ax=ax,
    color="cyan",
    marker="^",
    markersize=120,
    edgecolor="black",
    zorder=5,
)

fig.colorbar(
    trans_im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
    label="Wet-to-dry transitions (count)",
)

ax.set_title(f"Wet-to-dry transitions — WY{tgt_wy}")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")

minx, miny, maxx, maxy = boundary.total_bounds
pad = 0.05 * max(maxx - minx, maxy - miny)

ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)
ax.set_aspect("equal")

plt.tight_layout()
plt.show()
# %%
