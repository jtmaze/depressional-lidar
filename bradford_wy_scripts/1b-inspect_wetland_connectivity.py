# %% 1.0 Imports, directories and file paths
import sys
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wetland_utilities.basin_attributes import WetlandBasin
dem_buffer = 0

source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_USGS.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/out_data/bradford_well_basins.shp'
wetland_connectivity_path = 'D:/depressional_lidar/data/bradford/bradford_wetland_connect_logging_key.xlsx'

footprints = gpd.read_file(footprints_path)
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry', 'site']]
    .rename(columns={'rtk_z': 'rtk_z'})
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)

wetland_ids = well_point['wetland_id'].unique().tolist()


connectivity = pd.read_excel(wetland_connectivity_path)


# %% 2.0 Visualize the wetland's DEM

results = []

for i in wetland_ids:

    fp = footprints[footprints['wetland_id'] == i]
    basin = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path, 
        footprint=fp,
        transect_buffer=dem_buffer
    )
    connectivity_class = connectivity[connectivity['wetland_id'] == i].iloc[0]['connectivity']
    
    print(f'Well ID: {i}, Connectivity: {connectivity_class}')

    basin.visualize_shape(show_shape=True, show_well=True, show_deepest=True, show_spill=True)
    basin.plot_basin_hypsometry(plot_points=True, plot_spill=True, plot_deepest=True)

    min_elev = basin.deepest_point.elevation
    spill_elev = basin.spill_point.elevation
    well_elev = basin.well_point.elevation_dem


    r = {
        'wetland_id': i,
        'min_elev': min_elev,
        'spill_elev': spill_elev,
        'well_elev': well_elev
    }
    results.append(r)

# %% 2.0 Plot the wetland spill depths as a histogram

results_df = pd.DataFrame(results)

results_df['spill_depth'] = results_df['spill_elev'] - results_df['min_elev']



# %%
