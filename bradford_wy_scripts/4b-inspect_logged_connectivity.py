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
wetland_pairs_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_all_wells.csv'
footprints_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'

wetland_pairs = pd.read_csv(wetland_pairs_path)
logged_ids = wetland_pairs['logged_id'].unique()
print(logged_ids)

footprints = gpd.read_file(footprints_path)
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
    .rename(columns={'rtk_elevat': 'rtk_elevation'})
    .query("type in ['core_well', 'wetland_well']")
)

buffer = 100

# %%

for i in logged_ids:

    log_basin = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=buffer
    )

    log_basin.visualize_shape(show_well=True, show_deepest=False)

# %%
