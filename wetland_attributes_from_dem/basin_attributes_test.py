"""
Tests the WetlandBasin and WetlandDynamics classes.
"""

# %% 1.0 Libraries and file paths
import geopandas as gpd
from basin_attributes import WetlandBasin
from basin_dynamics import BasinDynamics, WellStageTimeseries

if __name__ == '__main__':
    wetland_id = '15_409'
    source_dem = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
    basins_path = 'D:/depressional_lidar/data/bradford/in_data/basins_assigned_wetland_ids.shp'
    well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
    well_stage_path = 'D:/depressional_lidar/data/bradford/in_data/stage_data/waterlevel_offsets_tracked.csv'

    # %% 2.0 Read and clean the data

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

    # %% 3.0 Create the WetlandBasin object and visualize attributes

    basin = WetlandBasin(
        wetland_id=wetland_id, 
        source_dem_path=source_dem, 
        footprint=footprint, 
        well_point_info=well_point,
        transect_method='deepest',
        transect_n=10,
        transect_buffer=20
    )

    basin.visualize_shape(show_deepest=True, show_centroid=True, show_well=True)
    #basin.plot_basin_hypsometry(plot_points=True)
    basin.radial_transects_map()
    basin.plot_individual_radial_transects()
    basin.plot_aggregated_radial_transects()
    basin.plot_hayashi_p(r0=2, r1=20)

    #basin.plot_hayashi_p(r0=2, r1=None, max=True)

    # %% 4.0 Incorporate Well Data to Explore Dynamics
    well_stage = WellStageTimeseries.from_csv(
        well_stage_path,
        well_id=wetland_id,
        date_column='Date',
        water_level_column='revised_depth',
        well_id_column='Site_ID'
    )
    well_stage.plot()

    # %% ..
    dynamics = BasinDynamics(basin=basin, well_stage=well_stage, well_to_dem_offset=-0.03)
    # %% 5.0 Visualize Inundation Dynamics
    import pandas as pd

    dynamics.visualize_single_inundation_map(date=pd.Timestamp('2023-7-01', tz='UTC'))
    dynamics.plot_inundated_area_timeseries()

    # %%
