
# %% 1.0 Libraries and file paths

import json
import pandas as pd
import geopandas as gpd

rtk_test_path = 'D:/depressional_lidar/data/misc/RTK_points_bradford_20260108.csv'
shp_file_out_path = 'D:/depressional_lidar/data/misc/RTK_points_bradford_20260108.shp'

rtk_test = pd.read_csv(rtk_test_path)
rtk_test.drop(
    columns=['uuid', 'fid', 'seq', 'snap_id', 'additional_data'],
    inplace=True
)

# %% 2.0 Calculate the orthometric height

# Parse the JSON strings and extract Geoid Difference
rtk_test['geoid_difference'] = rtk_test['pos_data'].apply(
    lambda x: json.loads(x)['Geoid Difference'] if pd.notna(x) else None
)

rtk_test.drop(
    columns=['pos_data'],
    inplace=True
)

# They did not enter the 2.2m pole height into the unit 
rtk_test['pole_ht'] = 2.2
rtk_test['elv'] = rtk_test['elv'] - rtk_test['pole_ht'] 
# Covert orthometric ht to elevation using the geoid difference
rtk_test['ortho_ht'] = rtk_test['elv'] - rtk_test['geoid_difference']

# %% 3.0 Convert to a shapefile

# GPS data is typically WGS84 (EPSG:4326)
gdf_rtk = gpd.GeoDataFrame(
    rtk_test, 
    geometry=gpd.points_from_xy(rtk_test.lon, rtk_test.lat),
    crs="EPSG:4326"
)

# Write to shapefile
gdf_rtk.to_file(shp_file_out_path, index=False)



# %%
