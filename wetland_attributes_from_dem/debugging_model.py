"""
Tests the WetlandBasin and WetlandDynamics classes.
"""

# 1.0 Libraries and file paths
import geopandas as gpd
import pandas as pd
from basin_attributes import WetlandBasin
from basin_dynamics import BasinDynamics, WellStageTimeseries
from wetland_model import WetlandModel

site = 'bradford'

if __name__ == '__main__':
    wetland_id = '7_243'
    source_dem = f'D:/depressional_lidar/data/{site}/in_data/{site}_DEM_cleaned_veg.tif'
    basins_path = f'D:/depressional_lidar/data/{site}/in_data/{site}_basins_assigned_wetland_ids_KG.shp'
    well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
    # osbs {site}_core_wells_tracked_datum.csv'
    # brandford waterlevel_offsets_tracked.csv'
    well_stage_path = f'D:/depressional_lidar/data/{site}/in_data/stage_data/{site}_wells_tracked_datum.csv'

    # 2.0 Read and clean the data
    footprint = gpd.read_file(basins_path)
    footprint = footprint[footprint['wetland_id'] == wetland_id]
    well_point = gpd.read_file(well_points_path)
    well_point = well_point[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
    """ 
    Clean well points gdf 
    """
    well_point.rename(
        columns={
            'rtk_elevat': 'rtk_elevation'
        },
        inplace=True
    )
    well_point = well_point[
        (well_point['type'] == 'core_well') | (well_point['type'] == 'wetland_well')
    ]
    well_point = well_point[well_point['wetland_id'] == wetland_id]


    # 3.0 Create the WetlandBasin object and visualize attributes
    basin = WetlandBasin(
        wetland_id=wetland_id, 
        source_dem_path=source_dem, 
        footprint=footprint, 
        well_point_info=well_point,
        transect_method='deepest',
        transect_n=10,
        transect_buffer=50
    )
    #basin.visualize_shape(show_deepest=True, show_centroid=True, show_well=True)


    # 4.0 Incorporate Well Data to Explore Dynamics
    well_stage = WellStageTimeseries.from_csv(
        well_stage_path,
        well_id=wetland_id,
        basin=basin,
        date_column='Date',
        water_level_column='revised_depth',
        well_id_column='Site_ID',
    )
    well_stage.plot()
    dynamics = BasinDynamics(basin=basin, well_stage=well_stage, well_to_dem_offset=0)
    dynamics.map_inundation_stacks()

    # 5.0 Implement Haki's BICY model
    wetland_model = WetlandModel(
        basin=basin,
        well_stage_timeseries=well_stage
    )

    wetland_model.plot_rET_and_Sy()