# %% 1.0 Libraries and file paths

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio

os.chdir('D:/depressional_lidar/data/')
bradford_points = pd.read_excel('doe_point_data_initial.xlsx', sheet_name='Bradford')
osbs_points = pd.read_excel('doe_point_data_initial.xlsx', sheet_name='OSBS')

# %% 2.0 Define functions to covert pts to gpd and find elevations

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

# %% 3.0 Convert Bradford and OSBS RTK points to GeoDataFrames

bradford_gdf = convert_pts_to_gpd(bradford_points, crs='EPSG:4326', latitude_colname='lat_coord', longitude_colname='long_coord')
osbs_gdf = convert_pts_to_gpd(osbs_points, crs='EPSG:4326', latitude_colname='lat_coord', longitude_colname='long_coord')

# %% 4.0 Estimate DEM elevation for Bradford and OSBS points
# Convert to DEM's crs for elevation extraction and filter empty geometries
bradford_gdf = bradford_gdf.to_crs('EPSG:26917')
bradford_gdf = bradford_gdf[~bradford_gdf.geometry.is_empty & bradford_gdf.geometry.notna()]
osbs_gdf = osbs_gdf.to_crs('EPSG:26917')
osbs_gdf = osbs_gdf[~osbs_gdf.geometry.is_empty & osbs_gdf.geometry.notna()]

bradford_gdf = estimate_pts_dem_elevation(
    bradford_gdf,
    dem_path='./bradford/in_data/bradford_DEM_cleaned_veg.tif',
    method='window_mean',
    window_size=3,
    elevation_colname='z_dem'
)

osbs_gdf = estimate_pts_dem_elevation(
    osbs_gdf,
    dem_path='./osbs/in_data/osbs_DEM_cleaned_veg.tif',
    method='window_mean',
    window_size=3,
    elevation_colname='z_dem'
)

# %% 5.0 Combine and write to file

combined_gdf = pd.concat([bradford_gdf, osbs_gdf], ignore_index=True)
combined_gdf = gpd.GeoDataFrame(combined_gdf, crs='EPSG:26917')
combined_gdf['rtk_dem_diff'] = combined_gdf['z_dem'] - pd.to_numeric(combined_gdf['rtk_z'], errors='coerce')

combined_gdf['long_proj'] = combined_gdf.geometry.x
combined_gdf['lat_proj'] = combined_gdf.geometry.y

# %% 6.0  Write the files

combined_gdf.to_file('rtk_pts_with_dem_elevations.shp')
combined_df = combined_gdf.drop(columns=['geometry'])
combined_df.to_excel('doe_point_data_master.xlsx', index=False)

# %%
