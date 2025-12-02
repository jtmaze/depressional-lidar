# %% 1.0 File paths and packages

import os 
import fiona
import geopandas as gpd

site_name = 'osbs'
nwi_path = f'D:/depressional_lidar/data/nwi_downloads/FL_Wetlands_Geopackage.gpkg'
site_boundary_path = f'D:/depressional_lidar/data/{site_name}/{site_name}_boundary.shp'
out_path = f'D:/depressional_lidar/data/{site_name}/in_data/original_basins/{site_name}_nwi_polygons.shp'

# %% 2.0 Clip NWI to the site boundary

site_boundary = gpd.read_file(site_boundary_path)
print(site_boundary.crs)

# Improve speeds by only reading NWI wetlands within the site bounds
with fiona.open(nwi_path, layer='FL_Wetlands') as src:
    nwi_crs = src.crs
reproj_site_bounds = site_boundary.to_crs(nwi_crs)
site_bound_box = tuple(reproj_site_bounds.total_bounds)

nwi = gpd.read_file(nwi_path, layer='FL_Wetlands', bbox=site_bound_box)
print(nwi.crs)

nwi_reproj = nwi.to_crs(site_boundary.crs)
site_wetlands = nwi_reproj[nwi_reproj.intersects(site_boundary.union_all())]

# %% 3.0 Remove Riverine Wetlands
print(site_wetlands['WETLAND_TYPE'].unique())
site_wetlands = site_wetlands[site_wetlands['WETLAND_TYPE'] != 'Riverine']


# %% 4.0 Write the site's wetlands to output

site_wetlands.to_file(out_path)

# %%
