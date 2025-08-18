# %% 2.0 Libraries and file paths

import os
import geopandas as gpd

os.chdir('D:/depressional_lidar/data/')
out_path = './bradford/in_data/basins_assigned_wetland_ids.shp'
depressions_path = './bradford/in_data/original_basins/well_basin_delineations.shp'
depressions = gpd.read_file(depressions_path)
print(depressions.crs)

well_points_path = './rtk_pts_with_dem_elevations.shp'
well_points = gpd.read_file(well_points_path)
print(well_points['type'].unique())
well_points = well_points[
    (well_points['type'] == 'wetland_well') |
    (well_points['type'] == 'core_well')
]

print(well_points.crs)

# %% Find the basin's centroid, then find the well_id closest to the basin's centroid

def assign_closest_wetland_id(
    depressions_gdf, 
    wells_gdf,
    max_distance
    ):
    """
    Assign the closest wetland_id to each depression based on centroid distance.

    Args:
    depressions_gdf: GeoDataFrame of depression polygons
    wells_gdf: GeoDataFrame of well points with 'wetland_id' column
    max_distance: Maximum distance threshold for assignment

    Returns:
    GeoDataFrame with added 'wetland_id' column
    """
    # Make a copy to avoid modifying original
    result = depressions_gdf.copy()
    
    # Calculate centroids of depressions
    centroids = result.geometry.centroid
    wetland_ids = []
    
    # For each depression, find closest well within max_distance
    for centroid in centroids:
        distances = wells_gdf.geometry.distance(centroid)
        closest_well_idx = distances.idxmin()
        min_distance = distances.loc[closest_well_idx]
        
        if min_distance <= max_distance:
            wetland_ids.append(wells_gdf.loc[closest_well_idx, 'wetland_id'])
        else:
            wetland_ids.append(None)

    result['wetland_id'] = wetland_ids
    return result

# %%

depressions = assign_closest_wetland_id(
    depressions, 
    well_points,
    250
)

out_depressions = depressions[depressions['wetland_id'].notna()]
print(len(out_depressions))
out_depressions = out_depressions[['wetland_id', 'geometry']]

# %%

out_depressions.to_file(out_path)

