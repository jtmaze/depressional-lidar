# %% 1.0 Libraries and file paths

import json

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds
from rasterio.warp import Resampling, reproject


data_dir = 'D:/depressional_lidar/data/bradford/'

weights_path = f'{data_dir}/out_data/well_wse_interpolations/kriging_weights_optimized_fit.h5'
well_data_path = f'{data_dir}/in_data/stage_data/bradford_well_data_long_gapfilled.csv'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
meta_path = f'{data_dir}/out_data/well_wse_interpolations/kriging_weights_metadata_optimized.json'
dem_path = f'{data_dir}/in_data/bradford_DEM_cleaned_USGS.tif'
kriging_template_path = f'{data_dir}/out_data/well_wse_interpolations/interpolated_median_WSE_optimized_model.tif'


# %% 2.0 Read weights, metadata, and well data

with pd.HDFStore(weights_path, mode='r') as store:
    weights_df = store['weights']
    weights = weights_df.to_numpy(dtype=np.float32)

with open(meta_path, 'r') as f:
    metadata = json.load(f)

grid_shape = (1_000, 1_000)

well_ids_in_weights = weights_df.columns.astype(str).tolist()

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'z_dem', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
    .copy()
)
well_points['wetland_id'] = well_points['wetland_id'].astype(str)
well_points = well_points[well_points['wetland_id'].isin(well_ids_in_weights)]

well_data = pd.read_csv(well_data_path)[['date', 'well_id', 'water_depth_m']].copy()
well_data['date'] = pd.to_datetime(well_data['date'])
well_data['well_id'] = well_data['well_id'].astype(str)
well_data = well_data[well_data['well_id'].isin(well_ids_in_weights)]

boundary = well_points.geometry.union_all().convex_hull.buffer(500)
boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=well_points.crs)


# %% 3.0 Functions for applying kriging weights to well values

def _aggregate_well_data(
    well_points: gpd.GeoDataFrame,
    well_data: pd.DataFrame,
    timestep: tuple,
    well_ids_in_weights: list,
) -> np.ndarray:
    begin = pd.to_datetime(timestep[0])
    end = pd.to_datetime(timestep[1])

    well_data_filtered = well_data[(well_data['date'] >= begin) & (well_data['date'] < end)].copy()

    well_wse_vals = well_data_filtered.groupby('well_id', as_index=True)['water_depth_m'].mean().to_frame()
    well_z = well_points[['wetland_id', 'z_dem']].drop_duplicates('wetland_id').set_index('wetland_id')
    well_wse_vals = well_wse_vals.join(well_z, how='left')
    well_wse_vals['wse'] = well_wse_vals['water_depth_m'] + well_wse_vals['z_dem']

    wse_ordered = well_wse_vals.reindex(well_ids_in_weights)['wse']
    if wse_ordered.isna().any():
        n_missing = int(wse_ordered.isna().sum())
        raise ValueError(f'Missing WSE values for {n_missing} wells in timestep {timestep}.')

    return wse_ordered.to_numpy(dtype=np.float32)


def _build_coarse_wse_grid(weights: np.ndarray, well_vals: np.ndarray, grid_shape: tuple) -> np.ndarray:
    surface_flat = weights @ well_vals
    return surface_flat.reshape(grid_shape)

def wse_grid_at_timestep(
    weights: np.ndarray,
    well_points: gpd.GeoDataFrame,
    well_data: pd.DataFrame,
    timestep: tuple,
    well_ids_in_weights: list,
    grid_shape: tuple,
) -> np.ndarray:
    well_vals = _aggregate_well_data(well_points, well_data, timestep, well_ids_in_weights)
    return _build_coarse_wse_grid(weights, well_vals, grid_shape)


# %% 4.0 Read DEM and kriging template georeferencing

target_dem_resolution_m = 10

with rasterio.open(dem_path) as dem_src:
    dem_crs = dem_src.crs
    boundary_dem = boundary_gdf.to_crs(dem_crs)

    minx, miny, maxx, maxy = boundary_dem.total_bounds
    dem_window = from_bounds(minx, miny, maxx, maxy, transform=dem_src.transform)
    dem_window = dem_window.round_offsets().round_lengths()
    window_transform = dem_src.window_transform(dem_window)

    src_res_x = abs(window_transform.a)
    src_res_y = abs(window_transform.e)
    out_width = max(1, int(np.ceil(dem_window.width * (src_res_x / target_dem_resolution_m))))
    out_height = max(1, int(np.ceil(dem_window.height * (src_res_y / target_dem_resolution_m))))

    dem = dem_src.read(
        1,
        window=dem_window,
        out_shape=(out_height, out_width),
        resampling=Resampling.average,
    ).astype(np.float32)

    dem_transform = window_transform * window_transform.scale(
        dem_window.width / out_width,
        dem_window.height / out_height,
    )
    dem_nodata = dem_src.nodata

if dem_nodata is not None:
    dem = np.where(dem == dem_nodata, np.nan, dem)

dem_valid_mask = geometry_mask(
    boundary_dem.geometry,
    transform=dem_transform,
    invert=True,
    out_shape=dem.shape,
)
dem = np.where(dem_valid_mask, dem, np.nan)

left, bottom, right, top = array_bounds(dem.shape[0], dem.shape[1], dem_transform)
dem_extent = (left, right, bottom, top)

with rasterio.open(kriging_template_path) as krig_src:
    krig_transform = krig_src.transform
    krig_crs = krig_src.crs

well_points_dem = well_points.to_crs(dem_crs)


# %% 5.0 Plot mean well timeseries for each timestep

time_steps = [
    ('2022-07-15', '2022-07-16'),
    ('2025-07-01', '2025-07-2'),
    ('2024-01-20', '2024-01-21'),
]

daily_mean = (
    well_data
    .groupby('date', as_index=True)['water_depth_m']
    .mean()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
ax.plot(daily_mean.index, daily_mean.values, color='0.25', linewidth=1.25, label='Daily mean water depth')

for i, (start, end) in enumerate(time_steps):
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    ax.axvspan(start_ts, end_ts, color='red', alpha=0.2)
    ax.axvline(start_ts, color='red', linewidth=1.25, linestyle='--')
    ax.axvline(end_ts, color='red', linewidth=1.25, linestyle='--')

ax.set_title('Well timeseries with selected inundation timesteps')
ax.set_ylabel('Water depth (m)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper right')
plt.show()


# %% 6.0 Generate inundation maps for different timesteps

tai_threshold_m = 0.2
tai_cmap = LinearSegmentedColormap.from_list('white_orange', ['white', 'orange'])

for timestep in time_steps:
    wse = wse_grid_at_timestep(
        weights=weights,
        well_points=well_points,
        well_data=well_data,
        timestep=timestep,
        well_ids_in_weights=well_ids_in_weights,
        grid_shape=grid_shape,
    )

    wse_resamp = np.empty_like(dem, dtype=np.float32)
    reproject(
        source=wse,
        destination=wse_resamp,
        src_transform=krig_transform,
        src_crs=krig_crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )

    wse_resamp = np.where(dem_valid_mask, wse_resamp, np.nan)
    valid_mask = dem_valid_mask & np.isfinite(dem) & np.isfinite(wse_resamp)
    inundated = np.where(valid_mask, (wse_resamp > dem).astype(np.float32), np.nan)
    tai_mask = np.where(valid_mask, (np.abs(wse_resamp - dem) <= tai_threshold_m).astype(np.float32), np.nan)

    inundated_pct = float(np.nanmean(inundated) * 100)
    tai_pct = float(np.nanmean(tai_mask) * 100)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    ax.imshow(
        inundated,
        cmap='Blues',
        vmin=0,
        vmax=1,
        origin='upper',
        extent=dem_extent,
    )

    tai_overlay = np.ma.masked_where(~np.isfinite(tai_mask) | (tai_mask < 0.5), tai_mask)
    ax.imshow(
        tai_overlay,
        cmap=tai_cmap,
        vmin=0,
        vmax=1,
        alpha=0.85,
        origin='upper',
        extent=dem_extent,
    )

    boundary_dem.boundary.plot(ax=ax, color='black', linewidth=1)
    ax.scatter(
        well_points_dem.geometry.x,
        well_points_dem.geometry.y,
        color='red',
        s=16,
        zorder=5,
    )
    ax.set_title(
        f'{timestep[0]} to {timestep[1]}\n'
        f'Inundated area: {inundated_pct:.1f}% | TAI area: {tai_pct:.1f}%'
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    plt.close(fig)
    del wse, wse_resamp, inundated, tai_mask

# %% 7.0 Build inundation and TAI frequency heatmaps (50m DEM, 20-day intervals)

interval_days = 5

start_date = well_data['date'].min().normalize()
end_date_exclusive = well_data['date'].max().normalize() + pd.Timedelta(days=1)

interval_edges = pd.date_range(start=start_date, end=end_date_exclusive, freq=f'{interval_days}D')
if len(interval_edges) == 0 or interval_edges[-1] < end_date_exclusive:
    interval_edges = interval_edges.append(pd.DatetimeIndex([end_date_exclusive]))

intervals = list(zip(interval_edges[:-1], interval_edges[1:]))

inundation_sum = np.zeros(dem.shape, dtype=np.float32)
tai_sum = np.zeros(dem.shape, dtype=np.float32)
valid_count = np.zeros(dem.shape, dtype=np.uint16)
wse_resamp = np.empty_like(dem, dtype=np.float32)

n_used = 0
for start_ts, end_ts in intervals:
    timestep = (start_ts, end_ts)

    try:
        wse = wse_grid_at_timestep(
            weights=weights,
            well_points=well_points,
            well_data=well_data,
            timestep=timestep,
            well_ids_in_weights=well_ids_in_weights,
            grid_shape=grid_shape,
        )
    except ValueError:
        continue

    reproject(
        source=wse,
        destination=wse_resamp,
        src_transform=krig_transform,
        src_crs=krig_crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )

    wse_resamp = np.where(dem_valid_mask, wse_resamp, np.nan)
    valid_mask = dem_valid_mask & np.isfinite(dem) & np.isfinite(wse_resamp)
    inundated = valid_mask & (wse_resamp > dem)
    tai_mask = valid_mask & (np.abs(wse_resamp - dem) <= tai_threshold_m)

    inundation_sum[inundated] += 1.0
    tai_sum[tai_mask] += 1.0
    valid_count[valid_mask] += 1
    n_used += 1

inundation_freq_pct = np.where(valid_count > 0, (inundation_sum / valid_count) * 100.0, np.nan)
tai_freq_pct = np.where(valid_count > 0, (tai_sum / valid_count) * 100.0, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

im0 = axes[0].imshow(
    inundation_freq_pct,
    cmap='Blues',
    vmin=0,
    vmax=100,
    origin='upper',
    extent=dem_extent,
)
boundary_dem.boundary.plot(ax=axes[0], color='black', linewidth=1)
axes[0].scatter(
    well_points_dem.geometry.x,
    well_points_dem.geometry.y,
    color='red',
    s=12,
    zorder=5,
)
axes[0].set_title(f'Inundation Frequency (%)')
axes[0].set_xticks([])
axes[0].set_yticks([])
cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
cbar0.set_label('% of time inundated')

im1 = axes[1].imshow(
    tai_freq_pct,
    cmap='Oranges',
    vmin=0,
    vmax=100,
    origin='upper',
    extent=dem_extent,
)
boundary_dem.boundary.plot(ax=axes[1], color='black', linewidth=1)
axes[1].scatter(
    well_points_dem.geometry.x,
    well_points_dem.geometry.y,
    color='red',
    s=12,
    zorder=5,
)
axes[1].set_title(f'TAI Frequency (%) |WSE - DEM| <= {tai_threshold_m:.1f}')
axes[1].set_xticks([])
axes[1].set_yticks([])
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
cbar1.set_label('% time in TAI')

plt.show()


# %%
