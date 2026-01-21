# %% 1.0 Imports, directories and file paths
import sys
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wetland_utilities.basin_attributes import WetlandBasin

source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'


footprints = gpd.read_file(footprints_path)
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry', 'site']]
    .rename(columns={'rtk_elevat': 'rtk_elevation'})
    .query("type in ['core_well', 'wetland_well'] and site == 'bradford'")
)

well_ids = well_point['wetland_id'].unique().tolist()
dem_buffer = 250

# %% 2.0 Visualize the wetland's DEM

for i in well_ids:

    log_basin = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=dem_buffer
    )

    log_basin.visualize_shape(show_well=True, show_deepest=False)

# %%
