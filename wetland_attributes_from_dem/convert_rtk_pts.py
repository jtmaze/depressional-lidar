# %% 1.0

import os
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

estimate_


# %%

bradford_gdf = convert_pts_to_gpd(bradford_points, crs='EPSG:4326', latitude_colname='rtk_latitude', longitude_colname='rtk_longitude')
bradford_gdf['site'] = 'bradford'
osbs_gdf = convert_pts_to_gpd(osbs_points, crs='EPSG:4326', latitude_colname='rtk_latitude', longitude_colname='rtk_longitude')
osbs_gdf['site'] = 'osbs'
# %%

combined_gdf = pd.concat([bradford_gdf, osbs_gdf], ignore_index=True)
combined_gdf = gpd.GeoDataFrame(combined_gdf, crs='EPSG:4326')

# %%

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

# %%
