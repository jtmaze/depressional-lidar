# %% 1.0 Libraries and File Paths

import pandas as pd
import geopandas as gpd
#import rasterio as rio

well_path = 'D:/depressional_lidar/data/delmarva/in_data/waterlevel_data/compiled_dmv_waterlevel.csv'
meta_path = 'D:/depressional_lidar/data/delmarva/in_data/waterlevel_data/dmv_well_metadata_submit.csv'

# %% 2.0 Reformat the wetland well data

well_data = pd.read_csv(well_path)
well_data = well_data.rename(
    columns={
        'waterLevel': 'well_depth_m',
        'well_id': 'wetland_id', 
        'Notes': 'notes', 
        'Flag': 'flag'
    }
)

well_data['day'] = pd.to_datetime(well_data['Timestamp']).dt.date
daily_well_data = well_data.groupby(['day', 'wetland_id']).agg({
    'well_depth_m': 'mean',
    'flag': lambda x: x.mode()[0] if not x.mode().empty else None,
    'notes': lambda x: x.mode()[0] if not x.mode().empty else None
}).reset_index()

# %% 2.1 Write the aggregated daily 

out_path = 'D:/depressional_lidar/data/delmarva/in_data/waterlevel_data/daily_well_depth_Fall2025.csv'
daily_well_data.to_csv(out_path, index=False)

# %% 3.0 Reformat the well's spatial metadata

well_meta_data = pd.read_csv(meta_path)

well_meta_data = well_meta_data.rename(
    columns={
        'well_id': 'wetland_id', 
        'well_type': 'type'
    }
)

well_meta_data = well_meta_data[['wetland_id', 'type', 'latitude', 'longitude', 'catchment']]

gdf_well_meta = gpd.GeoDataFrame(
    well_meta_data, 
    geometry=gpd.points_from_xy(well_meta_data.longitude, well_meta_data.latitude),
    crs="EPSG:4326"
)

well_meta_data_proj = gdf_well_meta.to_crs(epsg='26918')
well_meta_data_proj['rtk_z'] = "Missing"
print(well_meta_data_proj.columns)

# %% 4.0 Write the well metadata 

out_shapes = 'D:/depressional_lidar/data/delmarva/delmarva_well_points.shp'
well_meta_data_proj.to_file(out_shapes, index=False)

# %% 5.0 Rewrite no data

import rasterio as rio

source_dem = f'D:/depressional_lidar/data/delmarva/in_data/2007_1m_DEM.tif'

with rio.open(source_dem) as dem:
    profile = dem.profile
    data = dem.read()

profile.update(nodata=0)

out_dem = 'D:/depressional_lidar/data/delmarva/in_data/2007_1m_DEM_modified.tif'
with rio.open(out_dem, 'w', **profile) as dst:
    dst.write(data)

# %%
