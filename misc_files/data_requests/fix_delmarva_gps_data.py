"""
The original precision GPS files had some major issues, 
blaming my (James Maze's) ineptitude in 2022
This script fixes those files by...

1) Designating a proper projection for data points
2) Repairing invalid geometries and removing GPS noise from shoreline maps. 

"""

# %% 1.0 Libraries and directories
import os
import pandas as pd
import geopandas as gpd

os.chdir('/Users/jmaze/Documents/projects/depressional_lidar/')

orig_projection = 'EPSG:6488'
out_projection = 'EPSG:26917'

raw_well_pts1 = gpd.read_file('./delmarva/Generic_.shp')
raw_well_pts2 = gpd.read_file('./delmarva/Point_ge.shp')
raw_well_pts = gpd.GeoDataFrame(pd.concat([raw_well_pts1, raw_well_pts2]), geometry='geometry')

polygons = gpd.read_file('./delmarva/Area_gen.shp')
# %% 2.0 Set the projection (from Trimble unit), then change to NAD83 - UTM Zone 17N

raw_well_pts.set_crs(orig_projection, inplace=True)
raw_well_pts.to_crs(out_projection, inplace=True)
polygons.set_crs(orig_projection, inplace=True)
polygons.to_crs(out_projection, inplace=True)

# %% 3.0 Fix the inundation polygon's geometries

# NOTE: Inundated area estimates might be sensitive to these parameters.

polygons_fixed = polygons.copy()

def clean_inundation_polygons(gdf, buffer_dist=5, simplify_tolerance=2):
    """Combined approach to clean noisy polygons"""
    
    # Step 1: Fix invalid geometries (you're already doing this)
    cleaned = gdf.copy()
    cleaned['geometry'] = cleaned.make_valid()
    # Step 2: Small buffer to close gaps and smooth edges
    cleaned['geometry'] = cleaned.buffer(buffer_dist)
    # Step 3: Negative buffer to return to original size
    cleaned['geometry'] = cleaned.buffer(-buffer_dist)

    # Step 4: Fill holes which are an artifact of GPS point's noise
    def fill_holes_conditionally(geom):
        if len(geom.interiors) > 0:
            from shapely.geometry import Polygon
            return Polygon(geom.exterior)
        else:
            return geom
        
    cleaned['geometry'] = cleaned.geometry.apply(fill_holes_conditionally)

     # Step 5: Simplify and remove very small polygons (likely noise)
    cleaned['geometry'] = cleaned.simplify(simplify_tolerance)
    cleaned = cleaned[cleaned.area > 100]  # Adjust threshold as needed

    return cleaned

polygons_fixed2 = clean_inundation_polygons(polygons_fixed, buffer_dist=1, simplify_tolerance=2)

# %% 4.0 Write the ammended files

raw_well_pts.to_file('./delmarva/trimble_well_pts.shp', index=False)
polygons_fixed2.to_file('./delmarva/inundation_extent_Mar14_2022.shp', index=False)


# %%
