"""
Tests the WetlandBasin class.
"""
import geopandas as gpd

wetland_id = '13_267'
source_dem = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
basins_path = 'D:/depressional_lidar/data/bradford/in_data/basins_assigned_wetland_ids.shp'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'

footprint = gpd.read_file(basins_path)
footprint = footprint[footprint['wetland_id'] == wetland_id]

well_point = gpd.read_file(well_points_path)
well_point = well_point[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
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

from basin_attributes import WetlandBasin

if __name__ == "__main__":
    basin = WetlandBasin(wetland_id, source_dem, footprint, well_point)
    
    deepest_point = basin.find_deepest_point()
    print(f"Deepest point found at: {deepest_point.location} depth is {deepest_point.elevation} meters")
    basin.establish_well_point(basin.well_point_info)
    basin.visualize_shape(show_deepest=True, show_centroid=True, show_well=True)