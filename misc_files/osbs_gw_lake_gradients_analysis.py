# %% 1.0 Libraries and file paths

import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.features import shapes
from shapely.geometry import shape as shp_shape
import plotly.graph_objects as go

data_dir = 'D:/depressional_lidar/data/osbs/'

well_ts_path = f'{data_dir}/in_data/neon_wse_data/neon_wse_data.csv'
well_meta_path = f'{data_dir}/in_data/neon_wse_data/neon_logger_meta.csv'
dem_path = f'{data_dir}/in_data/osbs_DEM_cleaned_neon_apr2023.tif'

well_ts = pd.read_csv(well_ts_path)
well_ts = well_ts[well_ts['flag'] == 0]
well_ts['timestamp'] = pd.to_datetime(well_ts['timestamp'], utc=True)
well_ts['date'] = well_ts['timestamp'].dt.normalize()

well_ts_daily = (
    well_ts.groupby(['lake_id', 'wetland_id', 'date'], as_index=False)['wse_m']
    .mean()
)

print(well_ts_daily.head(10))

well_meta = pd.read_csv(well_meta_path)
well_meta_simple = well_meta.drop(columns=['z_offset']).copy()
well_meta_simple = well_meta_simple.drop_duplicates().reset_index()
print(well_meta_simple.head(5))

# %% 2.0 Convert well points to GeoDataFrame

well_points = well_meta_simple.dropna(subset=['ref_lon', 'ref_lat']).copy()
well_gdf = gpd.GeoDataFrame(
	well_points,
	geometry=gpd.points_from_xy(well_points['ref_lon'], well_points['ref_lat']),
	crs='EPSG:4326',
)

well_gdf_utm = well_gdf.to_crs(epsg=32617)

# %% 3.0 Generate a simple lake wse column

lake_ts = well_ts_daily[well_ts_daily['wetland_id'].str.lower().str.startswith('lake')].copy()

lake_daily_wse = (
	lake_ts.groupby(['lake_id', 'date'], as_index=False)['wse_m']
	.mean()
	.rename(columns={'wse_m': 'lake_wse_m'})
)

print(lake_daily_wse.head())

lake_longrun_wse = (
	lake_ts.groupby('lake_id', as_index=False)['wse_m']
	.mean()
	.rename(columns={'wse_m': 'lake_wse_longrun_m'})
)

# %% 4.0 Group by lake_id and plot points on DEM

gw_well_lateral_dists = []

with rio.open(dem_path) as dem_src:

    dem_bounds = dem_src.bounds

    for lake_id, lake_wells in well_gdf_utm.groupby('lake_id'):

        xmin, ymin, xmax, ymax = lake_wells.total_bounds
        pad = 450

        left = max(xmin - pad, dem_bounds.left)
        right = min(xmax + pad, dem_bounds.right)
        bottom = max(ymin - pad, dem_bounds.bottom)
        top = min(ymax + pad, dem_bounds.top)

        window = rio.windows.from_bounds(left, bottom, right, top, dem_src.transform)
        dem_subset = dem_src.read(1, window=window, masked=True)
        dem_transform = dem_src.window_transform(window)

        fig, ax = plt.subplots(figsize=(8, 7))
        img = show(dem_subset, transform=dem_transform, ax=ax, cmap='terrain')

        # Add colorbar for elevation
        cbar = fig.colorbar(img.get_images()[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Elevation (m)')

        lake_surface = lake_longrun_wse.loc[
            lake_longrun_wse['lake_id'] == lake_id,
            'lake_wse_longrun_m',
        ].iloc[0]

        dem_vals = dem_subset.filled(np.nan)
        below_surface = (dem_vals < lake_surface) & ~np.isnan(dem_vals)

        inundation_polys = [
            shp_shape(geom)
            for geom, value in shapes(
                below_surface.astype('uint8'),
                mask=below_surface,
                transform=dem_transform,
            )
            if value == 1
        ]

        if inundation_polys:
            inundation_polys = [max(inundation_polys, key=lambda p: p.area)]

        lake_longrun_baseline = lake_longrun_wse.loc[
            lake_longrun_wse['lake_id'] == lake_id,
            'lake_wse_longrun_m',
        ].iloc[0]

        gpd.GeoDataFrame(geometry=inundation_polys, crs=well_gdf_utm.crs).plot(
            ax=ax,
            color='red',
            alpha=0.35,
            edgecolor='none',
            label=f'Below long-run lake_wse ({lake_longrun_baseline:.2f} m)',
        )

        # Compute lateral distance from GW wells to shoreline
        if inundation_polys:
            shoreline = inundation_polys[0].boundary
            for row in lake_wells.itertuples():
                if str(row.wetland_id).lower().startswith('gwlake'):
                    dist = row.geometry.distance(shoreline)
                    gw_well_lateral_dists.append({
                        'lake_id': lake_id,
                        'wetland_id': row.wetland_id,
                        'lateral_dist_m': dist
                    })

        wetland_ids = lake_wells['wetland_id'].astype(str).str.lower()
        is_gw = wetland_ids.str.startswith('gwlake')
        is_lake = wetland_ids.str.startswith('lake')

        lake_wells.loc[is_lake].plot(
            ax=ax,
            color='deepskyblue',
            edgecolor='black',
            markersize=100,
            marker='o',
            label='Lake gauge',
        )

        lake_wells.loc[is_gw].plot(
            ax=ax,
            color='pink',
            edgecolor='black',
            markersize=100,
            marker='^',
            label='GW Well',
        )

        # -------- Label wells --------
        for row in lake_wells.itertuples():
            label = str(row.wetland_id).split('_')[-1]

            ax.text(
                row.geometry.x + 5,
                row.geometry.y + 5,
                label,
                fontsize=14,
                ha='left',
                va='bottom'
            )
        # -----------------------------

        # Add a 100 m scale bar
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        sb_len = 100
        sb_x = x0 + (x1 - x0) * 0.05
        sb_y = y0 + (y1 - y0) * 0.05

        ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color='black', linewidth=3, zorder=10)

        ax.text(
            sb_x + sb_len/2,
            sb_y + (y1 - y0)*0.01,
            f'{sb_len} m',
            color='black',
            ha='center',
            va='bottom',
            fontsize=10,
            weight='bold',
            zorder=10
        )

        ax.set_title(f'{lake_id} wells on DEM')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        plt.show()

# %% 5.0 Combine distance results

dist_df = pd.DataFrame(gw_well_lateral_dists)
print(dist_df)

# %% 6.0 Calculate each gw well's elevation difference from thier respective lakes long-run WSE

# Bar plot of differences for each GW well.

gw_well_meta = (
	well_gdf_utm[well_gdf_utm['wetland_id'].str.lower().str.startswith('gwlake')]
	[['lake_id', 'wetland_id', 'ref_z', 'geometry']]
	.copy()
)

gw_vs_lake = gw_well_meta.merge(lake_longrun_wse, on='lake_id', how='left')
gw_vs_lake['elev_diff_m'] = gw_vs_lake['ref_z'] - gw_vs_lake['lake_wse_longrun_m']
gw_vs_lake = gw_vs_lake.sort_values(['lake_id', 'elev_diff_m']).reset_index(drop=True)

print(gw_vs_lake[['lake_id', 'wetland_id', 'ref_z', 'lake_wse_longrun_m', 'elev_diff_m']])

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = gw_vs_lake['elev_diff_m'].map(lambda x: 'steelblue' if x >= 0 else 'firebrick')

ax.bar(gw_vs_lake['wetland_id'], gw_vs_lake['elev_diff_m'], color=bar_colors)
ax.axhline(0, color='black', linewidth=1)
ax.set_title('GW Well Elevation Difference from Long-Run Lake WSE')
ax.set_xlabel('GW well')
ax.set_ylabel('ref_z - long-run lake WSE (m)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# %% 7.0 Box plot for head gradients at BARC

# Convert lists of lateral distances to a DataFrame
gw_dist_df = pd.DataFrame(gw_well_lateral_dists)

# Create a clean dataframe for WSE comparison that matches lake & gw daily wse for BARC
barc_lake_ts = well_ts_daily[well_ts_daily['wetland_id'].str.lower().str.startswith('lake')]
barc_lake_ts = barc_lake_ts[barc_lake_ts['lake_id'] == 'SUGG']

barc_gw_ts = well_ts_daily[well_ts_daily['wetland_id'].str.lower().str.startswith('gwlake')]
barc_gw_ts = barc_gw_ts[barc_gw_ts['lake_id'] == 'SUGG'].copy()

# Rename columns to merge daily lake and gw levels cleanly
barc_lake_ts = barc_lake_ts.rename(columns={'wse_m': 'lake_wse_m'})[['date', 'lake_wse_m']]
barc_gw_ts = barc_gw_ts[['wetland_id', 'date', 'wse_m']]

# Merge lake WSE, GW WSE, and distance
barc_merged = barc_gw_ts.merge(barc_lake_ts, on='date', how='inner')
barc_merged = barc_merged.merge(gw_dist_df, on='wetland_id', how='left')

# Calculate gradient: dh / dl = (lake_wse - well_wse) / dist
barc_merged['dh_dl'] = (barc_merged['lake_wse_m'] - barc_merged['wse_m']) / barc_merged['lateral_dist_m']

# Create box plot
fig, ax = plt.subplots(figsize=(8, 8))

barc_merged.boxplot(column='dh_dl', by='wetland_id', ax=ax, grid=False, showfliers=False)
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.set_title('Head Gradients (Lake to well) for SUGG Wells')
ax.set_ylabel('dh/dl (m/m)\n(Positive means lake > well)')
ax.set_xlabel('GW well')
plt.suptitle('') # Remove annoying pandas auto title
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() 

# %% 8.0 Make a scatter plot for each GW well correlating dh/dL (y-axis) with lake water levels (x-axis)

unique_wells = barc_merged['wetland_id'].unique()
unique_wells = unique_wells[unique_wells != 'GWlake_BARC_304']
print(unique_wells)

n_wells = len(unique_wells)

# Calculate grid size for subplots
ymin = -0.04 #barc_merged['dh_dl'].min()
ymax = 0.07 #barc_merged['dh_dl'].max()
n_cols = 4
n_rows = (n_wells + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
axes = axes.flatten()

barc_merged['year'] = barc_merged['date'].dt.year

for i, well_id in enumerate(unique_wells):
    ax = axes[i]
    subset = barc_merged[barc_merged['wetland_id'] == well_id]
    print(subset['date'].min())
    print(subset['date'].max())
    
    # Scatter plot: x = lake water level, y = dh/dl
    sc = ax.scatter(subset['lake_wse_m'], subset['dh_dl'], 
                    c=subset['year'], cmap='viridis', alpha=0.7)
    # Add a horizontal line at 0 for visual reference
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    
    ax.set_title(f'{well_id.split('_')[-1]}')
    ax.set_ylabel('dh/dl (m/m)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(ymin, ymax)

# Turn off any unused leftover subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, location='right')
cbar.set_label('Year')
cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.xlabel('Lake Barco WSE (m_', fontsize=20)
plt.show()
# %%
