"""
Tests the WetlandBasin class.
"""
import geopandas as gpd

wetland_id = '14_115'
source_dem = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
basins_path = 'D:/depressional_lidar/data/bradford/in_data/basins_assigned_wetland_ids.shp'

footprint = gpd.read_file(basins_path)
footprint = footprint[footprint['wetland_id'] == wetland_id]

from basin_attributes import WetlandBasin

if __name__ == "__main__":
    basin = WetlandBasin(wetland_id, source_dem, footprint)
    
    deepest_point = basin.find_deepest_point()
    print(f"Deepest point found at: {deepest_point.location} depth is {deepest_point.depth} meters")
    basin.visualize_shape(show_deepest=True, show_centroid=True)