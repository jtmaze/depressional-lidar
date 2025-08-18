# %% 1.0

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio

os.chdir('D:/depressional_lidar/data/')
bradford_points = pd.read_excel('doe_point_data_master.xlsx', sheet_name='Bradford')
osbs_points = pd.read_excel('doe_point_data_master.xlsx', sheet_name='OSBS')

# %%

def convert_pts_to_gpd(
    pts_df: pd.DataFrame,
    crs: str, 
    latitude_colname: str,
    longitude_colname: str
) -> gpd.GeoDataFrame:
    """
    Convert a pandas DataFrame with lat/lon columns to a GeoDataFrame
    
    Args:
        pts_df: DataFrame containing point coordinates
        crs: Coordinate reference system string (e.g. 'EPSG:4326')
        latitude_colname: Name of latitude column
        longitude_colname: Name of longitude column
        
    Returns:
        GeoDataFrame with Point geometry
    """

    # Drop rows where lat/lon can't be converted to float
    pts_df[latitude_colname] = pd.to_numeric(pts_df[latitude_colname], errors='coerce')
    pts_df[longitude_colname] = pd.to_numeric(pts_df[longitude_colname], errors='coerce')

    return gpd.GeoDataFrame(
        pts_df,
        geometry=gpd.points_from_xy(
            pts_df[longitude_colname],
            pts_df[latitude_colname]
        ),
        crs=crs
    )

def estimate_pts_dem_elevation(
    pts_gdf: gpd.GeoDataFrame,
    dem_path: str,
    method: str = 'single',
    window_size: int = None,
    elevation_colname: str = 'elevation_dem'
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
            def sample_window(geom):
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
            
            pts_gdf[elevation_colname] = pts_gdf.geometry.apply(sample_window)
        

    # Return the points GeoDataFrame with elevation estimates
    return pts_gdf

# %% Convert Bradford and OSBS RTK points to GeoDataFrames

bradford_gdf = convert_pts_to_gpd(bradford_points, crs='EPSG:4326', latitude_colname='rtk_latitude', longitude_colname='rtk_longitude')
bradford_gdf['site'] = 'bradford'
osbs_gdf = convert_pts_to_gpd(osbs_points, crs='EPSG:4326', latitude_colname='rtk_latitude', longitude_colname='rtk_longitude')
osbs_gdf['site'] = 'osbs'

# %% Estimate DEM elevation for Bradford and OSBS points
# Convert to DEM crs for elevation extraction and filter empty geometries
bradford_gdf = bradford_gdf.to_crs('EPSG:26917')
bradford_gdf = bradford_gdf[~bradford_gdf.geometry.is_empty & bradford_gdf.geometry.notna()]
osbs_gdf = osbs_gdf.to_crs('EPSG:26917')
osbs_gdf = osbs_gdf[~osbs_gdf.geometry.is_empty & osbs_gdf.geometry.notna()]

bradford_gdf = estimate_pts_dem_elevation(
    bradford_gdf,
    dem_path='./bradford/in_data/bradford_DEM_cleaned_veg.tif',
    method='single',
    window_size=None,
    elevation_colname='elevation_dem_single'
)

bradford_gdf = estimate_pts_dem_elevation(
    bradford_gdf,
    dem_path='./bradford/in_data/bradford_DEM_cleaned_veg.tif',
    method='window_mean',
    window_size=3,
    elevation_colname='elevation_dem_windowed'
)

osbs_gdf = estimate_pts_dem_elevation(
    osbs_gdf,
    dem_path='./osbs/in_data/osbs_DEM_cleaned_veg.tif',
    method='single',
    window_size=None,
    elevation_colname='elevation_dem_single'
)

osbs_gdf = estimate_pts_dem_elevation(
    osbs_gdf,
    dem_path='./osbs/in_data/osbs_DEM_cleaned_veg.tif',
    method='window_mean',
    window_size=3,
    elevation_colname='elevation_dem_windowed'
)

# %%

combined_gdf = pd.concat([bradford_gdf, osbs_gdf], ignore_index=True)
combined_gdf = gpd.GeoDataFrame(combined_gdf, crs='EPSG:26917')
combined_gdf['rtk_dem_diff'] = combined_gdf['elevation_dem_single'] - combined_gdf['rtk_elevation']

# %%
"""
Plots to explore RTK and DEM efficacy
"""

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot histograms for each site
plt.hist([
    pd.to_numeric(combined_gdf[combined_gdf['site'] == 'bradford']['rtk_vert_accuracy'], errors='coerce') / 1_000,
    pd.to_numeric(combined_gdf[combined_gdf['site'] == 'osbs']['rtk_vert_accuracy'], errors='coerce') / 1_000   
], bins=50, label=['Bradford', 'OSBS'], edgecolor='black', alpha=0.7)

plt.title('Vertical Accuracy reported by RTK GPS by Site')
plt.xlabel('RTK reported accuracy (meters)')
plt.ylabel('Observations')
plt.legend()
plt.show()

# %%
reasonable_observations = combined_gdf[
    (pd.to_numeric(combined_gdf['rtk_vert_accuracy'], errors='coerce') / 1_000) < 0.5
]

plt.figure(figsize=(10, 6))

# Plot histograms for each site
plt.hist([
    pd.to_numeric(reasonable_observations[reasonable_observations['site'] == 'bradford']['rtk_vert_accuracy'], errors='coerce') / 1_000,
    pd.to_numeric(reasonable_observations[reasonable_observations['site'] == 'osbs']['rtk_vert_accuracy'], errors='coerce') / 1_000   
], bins=30, label=['Bradford', 'OSBS'], edgecolor='black', alpha=0.7)

plt.title('Vertical Accuracy reported by RTK GPS by Site (filtered < 0.5m)')
plt.xlabel('RTK reported accuracy (meters)')
plt.ylabel('Observations')
plt.legend()
plt.show()

# %% Plot the DEM RTK elevation difference color-coded by site
plot_df = combined_gdf[abs(combined_gdf['rtk_dem_diff']) < 100]
plt.figure(figsize=(10, 6))

# Plot histograms for each site
plt.hist([
    plot_df[plot_df['site'] == 'bradford']['rtk_dem_diff'],
    plot_df[plot_df['site'] == 'osbs']['rtk_dem_diff']
], bins=30, label=['Bradford', 'OSBS'], edgecolor='black', alpha=0.7)

plt.axvline(0, color='red', linestyle='--', linewidth=1, label='No Difference')
plt.title('Difference between DEM and RTK Elevations by Site')
plt.xlabel('DEM - RTK Elevation Difference (m)')
plt.ylabel('Observations')
plt.legend()
plt.show()

# %% ..

combined_gdf['abs_dem_diff'] = abs(combined_gdf['rtk_dem_diff'])
combined_gdf = combined_gdf[combined_gdf['abs_dem_diff'] < 2]
combined_gdf['rtk_dem_diff'] = combined_gdf['rtk_dem_diff'].astype(float)
combined_gdf.to_file('rtk_pts_with_dem_elevations.shp')



# %% 

def plot_well_elevations(
    gdf: gpd.GeoDataFrame,
    tgt_well: str
):
    well_data = gdf[gdf['nearest_well_id'] == tgt_well]
    well_data['observation_index'] = well_data['nearest_well_id'].astype(str) + '_' + well_data['type'] + '_' + well_data['location']
    well_data = well_data.sort_values(by='rtk_elevation')
    if well_data.empty:
        print(f"No data found for well: {tgt_well}")
        return

    plt.figure(figsize=(10, 6))
    x_positions = range(len(well_data))
    plt.scatter(x_positions, well_data['elevation_dem_single'], label='Approx DEM Elevation (Single Cell)', color='blue')
    plt.scatter(x_positions, well_data['elevation_dem_windowed'], label='Approx DEM Elevation (Windowed 3x3)', color='green')
    plt.scatter(x_positions, well_data['rtk_elevation'], label='RTK Elevation', color='red')
    plt.xticks(x_positions, well_data['observation_index'], rotation=45, ha='right')
    plt.title(f'Elevation Comparison for Well: {tgt_well}')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
plot_df = combined_gdf[combined_gdf['type'] == 'soil_moisture_sensor']
plot_well_elevations(plot_df, tgt_well='')
