"""
Tests the WetlandBasin and WetlandDynamics classes.
"""

# 1.0 Libraries and file paths
import geopandas as gpd
import pandas as pd
from basin_attributes import WetlandBasin
from basin_dynamics import BasinDynamics, WellStageTimeseries

site = 'bradford'

if __name__ == '__main__':
    wetland_id = '13_410'
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
    print(well_point)

    # 3.0 Create the WetlandBasin object and visualize attributes

    basin = WetlandBasin(
        wetland_id=wetland_id, 
        source_dem_path=source_dem, 
        footprint=footprint, 
        well_point_info=well_point,
        transect_method='deepest',
        transect_n=10,
        transect_buffer=25
    )

    basin.visualize_shape(show_deepest=True, show_centroid=True, show_well=True)
    basin.plot_basin_hypsometry(plot_points=True)

    # basin.radial_transects_map(uniform=False)
    # basin.radial_transects_map(uniform = True)
    # basin.plot_individual_radial_transects(uniform=False)
    # basin.plot_individual_radial_transects(uniform=True)
    # basin.plot_aggregated_radial_transects(uniform=False)
    # basin.plot_aggregated_radial_transects(uniform=True)

    # basin.plot_hayashi_p(r0=2, r1=30, uniform=False)
    # basin.plot_hayashi_p(r0=1, r1=None, uniform=True)
    # basin.plot_hayashi_p(r0=10, r1=None, uniform=True)


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

    # ..
    dynamics = BasinDynamics(basin=basin, well_stage=well_stage, well_to_dem_offset=0)
    # 5.0 Visualize Inundation Dynamics

    dynamics.visualize_single_inundation_map(
        date=pd.Timestamp('2022-10-15', 
                          tz='UTC')
    )
    dynamics.visualize_single_tai_map(
        date=pd.Timestamp('2022-10-15', tz='UTC'),
        max_depth=0.10,
        min_depth=0
    )

    # dynamics.plot_inundated_area_timeseries()
    # dynamics.plot_tai_area_timeseries(max_depth=0.05, min_depth=-0.05)
    # dynamics.plot_inundated_area_histogram()
    # dynamics.plot_tai_area_histogram(max_depth=0.05, min_depth=-0.05)

    dynamics.map_tai_stacks(max_depth=0.05, min_depth=-0.05)
    dynamics.map_inundation_stacks()
