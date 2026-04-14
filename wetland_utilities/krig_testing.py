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

well_ts_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'

# %% 2.0 Read data

well_ts = pd.read_csv(well_ts_path)

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'geometry', 'z_dem', 'site']]
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)

basin_13_ids = ['13_263', '13_267', '13_271', '13_410', '13_274', 'Donor_wetland', 'Receiver_wetland']
well_points = well_points[~well_points['wetland_id'].isin(basin_13_ids)]
print(len(well_points))

boundary = well_points.geometry.union_all().convex_hull.buffer(500)

# Quickly visualize the wells
fig, ax = plt.subplots()
well_points.plot(ax=ax, color='red', markersize=20)
gpd.GeoSeries([boundary]).plot(ax=ax, facecolor='none', edgecolor='blue')
plt.show()

# %% 3.0 Run Kriging on Depth Scenarios

well_array_med = WellArray(
    well_pts=well_points,
    well_ts=well_ts,
    begin='2020-03-01',
    end='2027-03-28',
    percentile=50
)

wtd_surface_med = WTDSurface(
    well_array=well_array_med,
    krig_params={
        'variogram_model': 'exponential',
        'variogram_parameters': None,
        'n_lags': 10,
    },
    coarse_grid_dims=(1000, 1000),
    boundary=boundary,
    plot_variogram=True
)

# wtd_surface_med_gauss.plot_masked_result(sigma_threshold=1.0)
wtd_surface_med.plot_masked_result(sigma_threshold=1.25)

lags = wtd_surface_med.okr_result['lags']
weights = wtd_surface_med.okr_result['weights']

print(wtd_surface_med.okr_result['variogram_model_parameters'])

# %% 5.0 Summarize the pair counts for each lag dist

coords = np.column_stack([
    well_points.geometry.x.to_numpy(),
    well_points.geometry.y.to_numpy()
])

# pairwise Euclidean distance matrix
dx = coords[:, 0][:, None] - coords[:, 0][None, :]
dy = coords[:, 1][:, None] - coords[:, 1][None, :]
dist = np.hypot(dx, dy)

n_lags = len(lags)
dmax = dist[np.triu_indices_from(dist, k=1)].max()
dmin = dist[np.triu_indices_from(dist, k=1)].min()
dd = (dmax - dmin) / n_lags
bins = [dmin + n * dd for n in range(n_lags)]
bins.append(dmax + 0.001)

# Count unique pairs per bin (upper triangle only)
iu = np.triu_indices_from(dist, k=1)
pair_dists = dist[iu]

pair_counts = []
for n in range(n_lags):
    count = int(((pair_dists >= bins[n]) & (pair_dists < bins[n + 1])).sum())
    pair_counts.append(count)

lag_pair_df = pd.DataFrame({
    'low_dist_meters': np.round(bins[:-1], -1).astype(int),
    'high_dist_meters': np.round(bins[1:], -1).astype(int),
    'n_pairs': pair_counts
})

#print(lag_pair_df.to_string(index=False))

bin_centers = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]

plt.figure(figsize=(8, 4))
plt.bar(bin_centers, pair_counts, align='center', width=500, edgecolor='k')
plt.axvline(10000, color='red', linestyle='--', label='10,000 m')
plt.xlabel('Lag Bin Center (meters)')
plt.ylabel('Number of Pairs')
plt.title('Pair Counts per Lag Bin')
plt.legend()
plt.tight_layout()
plt.show()

# %% 6.0 Write the results

x_flat = wtd_surface_med._x_grid.ravel().astype(np.float32)
y_flat = wtd_surface_med._y_grid.ravel().astype(np.float32)
wetland_ids = wtd_surface_med.well_array.wtd_points['wetland_id'].to_list()

out_dir = r"D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations"

# # 5.1. CSV 
weights_df = pd.DataFrame(weights, columns=wetland_ids)
# weights_df.to_csv(f"{out_dir}/kriging_weights.csv", index=False, float_format="%.6f")

# # 5.2. HDF5 
with pd.HDFStore(f"{out_dir}/kriging_weights_optimized_fit.h5", mode="w") as store:
    store.put("weights", weights_df, format="fixed")
    store.put("grid_coords", pd.DataFrame({"x": x_flat, "y": y_flat}), format="fixed")

metadata = {
    "description": "Ordinary Kriging weights, optimized fitted exponential model",
    "weights_shape": list(weights.shape),
    "wetland_ids": wetland_ids,
    "grid_shape": list(wtd_surface_med._x_grid.shape),
    "crs": "EPSG:26917",
    "notes": [
        "Rows correspond to flattened kriging grid cells in row-major order.",
        "Columns correspond to wells in the order listed in well_ids.",
    ]
}

import json
with open(f"{out_dir}/kriging_weights_metadata_optimized.json", "w") as f:
    json.dump(metadata, f, indent=2)

wtd_surface_med.write_masked_tif(
    out_path=f'{out_dir}/interpolated_median_WSE_optimized_model.tif',
    sigma_threshold=1.25,
    crs='EPSG:26917'
)


# %%
