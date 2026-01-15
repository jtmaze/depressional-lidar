# %% 1.0 Libraries and file paths

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import geopandas as gpd

rtk_test_path = 'D:/depressional_lidar/data/misc/RTK_points_bradford_20260108.shp'
dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg1.tif'

with rio.open(dem_path) as src:
    dem_crs = src.crs

rtk_test = gpd.read_file(rtk_test_path)
rtk_test = rtk_test.to_crs(dem_crs)
rtk_test.drop(columns=['time', 'start_time', 'speed', 'bearing'], inplace=True)

# %% 2.0 Function to estimate the point's elevation from the DEM

def estimate_pts_dem_elevation(
    pts_gdf: gpd.GeoDataFrame,
    dem_path: str,
    method: str = 'single',
    window_size: int = None,
    elevation_colname: str = 'elv_dem'
) -> gpd.GeoDataFrame:
    
    """
    Estimate elevation of points from a DEM
    """
    with rio.open(dem_path) as dem_src:
        if method == 'single':
            # Sample each point individually
            pts_gdf[elevation_colname] = pts_gdf.geometry.apply(
                lambda geom: list(rio.sample.sample_gen(dem_src, [(geom.x, geom.y)]))[0][0]
            )

        elif method == 'window_mean':
            # Sample NxN window around point and take mean
            def _sample_window(geom):
                try:
                    # Convert point to pixel coordinates
                    x, y = geom.x, geom.y
                    row, col = dem_src.index(x, y)
                    # Define window around the point
                    half_window = window_size // 2
                    window = rio.windows.Window(
                        col - half_window, 
                        row - half_window, 
                        window_size, 
                        window_size
                    )
                    # Read the window
                    data = dem_src.read(1, window=window)
                    # Handle nodata
                    nodata = dem_src.nodata
                    if nodata is not None:
                        data = np.where(data == nodata, np.nan, data)
                    
                    return np.nanmean(data)
                    
                except:
                    print(f"Error sampling point {geom}")
                    return np.nan
            
            pts_gdf[elevation_colname] = pts_gdf.geometry.apply(_sample_window)
        
    # Return the points GeoDataFrame with elevation estimates
    return pts_gdf


# %% 3.0 Run the function to estimate DEM elevation

rtk_test = estimate_pts_dem_elevation(
    pts_gdf=rtk_test,
    dem_path=dem_path,
    method='window_mean',
    window_size=3,
    elevation_colname='elv_dem'
)

# %% 4.0 Identify the road control points for an initial check

rtk_test['road_control_pt'] = rtk_test['remarks'].str.contains(
    'road', case=False, na=False
).astype(int)
rtk_test['elv_diff'] = rtk_test['ortho_ht'] - rtk_test['elv_dem']
rtk_test['wetland'] = rtk_test['remarks'].str.split('_').str[0]

# Test idea of applying an offset to RTK data
rtk_test['orth_ht_offset'] = rtk_test['ortho_ht'] + 1
rtk_test['elv_diff_offset'] = rtk_test['ortho_ht_offset'] - rtk_test['elv_dem']


# %% 5.0 Histogram for all RTK points

diff = rtk_test['elv_diff'].dropna()
mean_diff = np.nanmean(diff)

plt.figure(figsize=(8, 6))
plt.hist(diff, bins=20, edgecolor='black')
plt.axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_diff:.2f}')
plt.title(f'All points (n={len(diff)}) Elevation Comparison')
plt.xlabel('RTK - DEM (meters)')
plt.ylabel('Count')
plt.grid(True)
plt.legend()
plt.show()

# %% 6.0 Histogram for the difference for the road control points

control_pts = rtk_test[rtk_test['road_control_pt'] == 1].copy()
diff = control_pts['elv_diff'].dropna()
mean_diff = np.nanmean(diff)

plt.figure(figsize=(8, 6))
plt.hist(diff, bins=20, edgecolor='black')
plt.axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_diff:.2f}')
plt.title(f'Road Control (n={len(diff)}) Elevation Comparison')
plt.xlabel('RTK - DEM (meters)')
plt.ylabel('Count')
plt.grid(True)
plt.legend()
plt.show()

# %% 7.0 Biplot to see if elv_diff relates to vertical accuracy

plt.figure(figsize=(8, 6))

plot_df = rtk_test[
    (rtk_test['fix_qualit'] == 9901) & (rtk_test['road_control_pt'] == 1)
]

wetlands = plot_df['wetland'].fillna('__NA__')
unique_wetlands = wetlands.unique()
cmap = plt.cm.get_cmap('tab20', len(unique_wetlands))
for i, z in enumerate(unique_wetlands):
    sel = plot_df[wetlands == z]
    plt.scatter(sel['accuracy_v'], abs(sel['elv_diff']), label=z, color=cmap(i), alpha=0.7)
plt.title('Road Points Only')
plt.xlabel('RTK Vertical Accuracy (meters)')
plt.ylabel('RTK - DEM (abs value, meters)')
plt.grid(True)
plt.legend(title='wetland', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% 8.0 See how offset could improve datum mismatch

# Histogram for all RTK points with offset
diff_offset = rtk_test['elv_diff_offset'].dropna()
mean_diff_offset = np.nanmean(diff_offset)

plt.figure(figsize=(8, 6))
plt.hist(diff_offset, bins=20, edgecolor='black')
plt.axvline(mean_diff_offset, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_diff_offset:.2f}')
plt.title(f'All points (n={len(diff_offset)}) Elevation Comparison (Offset)')
plt.xlabel('RTK Offset - DEM (meters)')
plt.ylabel('Count')
plt.grid(True)
plt.legend()
plt.show()

# Histogram for road control points with offset
control_pts = rtk_test[rtk_test['road_control_pt'] == 1].copy()
diff_offset = control_pts['elv_diff_offset'].dropna()
mean_diff_offset = np.nanmean(diff_offset)

plt.figure(figsize=(8, 6))
plt.hist(diff_offset, bins=20, edgecolor='black')
plt.axvline(mean_diff_offset, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_diff_offset:.2f}')
plt.title(f'Road Control (n={len(diff_offset)}) Elevation Comparison (Offset)')
plt.xlabel('RTK Offset - DEM (meters)')
plt.ylabel('Count')
plt.grid(True)
plt.legend()
plt.show()

# Biplot with offset
plt.figure(figsize=(8, 6))

plot_df = rtk_test[
    (rtk_test['fix_qualit'] == 9901) & (rtk_test['road_control_pt'] == 1)
]

wetlands = plot_df['wetland'].fillna('__NA__')
unique_wetlands = wetlands.unique()
cmap = plt.cm.get_cmap('tab20', len(unique_wetlands))
for i, z in enumerate(unique_wetlands):
    sel = plot_df[wetlands == z]
    plt.scatter(sel['accuracy_v'], abs(sel['elv_diff_offset']), label=z, color=cmap(i), alpha=0.7)
plt.title('Road Points Only (Offset)')
plt.xlabel('RTK Vertical Accuracy (meters)')
plt.ylabel('RTK Offset - DEM (abs value, meters)')
plt.grid(True)
plt.legend(title='wetland', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
