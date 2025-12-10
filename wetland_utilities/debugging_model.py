"""
Tests the WetlandBasin and WetlandDynamics classes.
"""

# 1.0 Libraries and file paths
import geopandas as gpd
import pandas as pd
from basin_attributes import WetlandBasin
from basin_dynamics import BasinDynamics, WellStageTimeseries
from wetland_utilities.wetland_water_balance_model import WetlandModel, ForcingData

site = 'bradford'
forcing = 'ERA-5'

if __name__ == '__main__':
    wetland_id = '3_173'
    source_dem = f'D:/depressional_lidar/data/{site}/in_data/{site}_DEM_cleaned_veg.tif'
    basins_path = f'D:/depressional_lidar/data/{site}/in_data/{site}_basins_assigned_wetland_ids_KG.shp'
    well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
    # osbs {site}_core_wells_tracked_datum.csv'
    # brandford waterlevel_offsets_tracked.csv'
    well_stage_path = f'D:/depressional_lidar/data/{site}/in_data/stage_data/waterlevel_Fall2025.csv'
    forcing_path = f'D:/depressional_lidar/data/{site}/in_data/hydro_forcings_and_LAI/{forcing}_daily_mean.csv'

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
        transect_buffer=20
    )
    basin.visualize_shape(show_deepest=True, show_centroid=True, show_well=True)
    basin.plot_basin_hypsometry(plot_points=True)


    # 4.0 Incorporate Well Data to Explore Dynamics
    well_stage = WellStageTimeseries.from_csv(
        well_stage_path,
        well_id=wetland_id,
        basin=basin,
        date_column='day',
        water_level_column='well_depth',
        well_id_column='well_id',
    )
    well_stage.plot()
    dynamics = BasinDynamics(basin=basin, well_stage=well_stage, well_to_dem_offset=0)
    dynamics.map_inundation_stacks()

    # 5.0 Implement Haki's BICY model
    forcing_data = ForcingData.from_csv(
        file_path=forcing_path, 
        data_source='ERA-5'
    )

    wetland_model = WetlandModel(
        basin=basin,
        well_stage_timeseries=well_stage,
        forcing_data=forcing_data,
        delta=0.1,
        well_flags = (2, 4),
        a=1, #NOTE why is my scale so far off...
        c=1.2,
        beta=1.5, 
        est_spill_depth=0.42
    )

    wetland_model.plot_filtered_timeseries()
    wetland_model.plot_rET_and_Sy()
    wetland_model.plot_Qh_A()
    wetland_model.plot_Q_timeseries()
    wetland_model.plot_rET_timeseries()
    wetland_model.plot_Sy_timeseries()
    wetland_model.plot_dh_dt_timseries(modeled=True)
    wetland_model.plot_dh_dt_timseries(modeled=False)
    wetland_model.plot_fluxes_timeseries()
    wetland_model.modeled_vs_actual_scatter_plot()
    wetland_model.difference_dh_dt_predictions_histogram(x_lims=(-0.15, 0.15))
    wetland_model.difference_dh_dt_predictions_onlogging(log_date='2023-01-07', x_lims=(-0.15, 0.15))