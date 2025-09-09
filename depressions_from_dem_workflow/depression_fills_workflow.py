# %% 1.0 Import libraries, organize file paths, set parameters
# 
import os
import rasterio as rio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes


import whitebox_workflows as wbw
from shapely.geometry import Polygon, MultiPolygon

wbe = wbw.WbEnvironment()
wbe.verbose = True
site = 'bradford'
min_depth = 0.03
max_depth_percentile = 95
smoothed_depression_threshold = 0.35
depression_buffer_distance = 3  # meters
min_depression_area = 300  # m^2

os.chdir(f'D:/depressional_lidar/data/{site}')
vegetation_off_dem_path = f"./in_data/{site}_DEM_cleaned_veg.tif"
filled_path = f'./temp/{site}_depressions_filled.tif'
boundary_path = f'./{site}_boundary.shp'

# %% 2.0 Make spill depths raster based on depression filling

# 2.1 Use the WhiteboxTools depression filling algorithm to fill depressions in the DEM
input_raster = wbe.read_raster(vegetation_off_dem_path)
filled_raster = wbe.fill_depressions(input_raster, max_depth=3.0)
wbe.write_raster(filled_raster, filled_path)

# 2.2 Subtract the filled DEM from the original DEM to get the depth of depressions
depression_depth_raster = filled_raster - input_raster
depression_depth_path = f'./temp/{site}_depression_depth.tif'
wbe.write_raster(depression_depth_raster, depression_depth_path)

# %% 3.0 Make a binary raster of depression shapes (1=filled, 0=not filled),
depression_depth_path = f'./temp/{site}_depression_depth.tif'
with rio.open(depression_depth_path) as src:
    depression_depth_array = src.read(1)
    no_data = src.nodata
    depression_depth_array[depression_depth_array == no_data] = np.nan

# NOTE: Setting the minimum fill depth to > 0.05m due to DEM noise
# NOTE: Since the ditches are so deep, setting a max depth (95th percentile) is a hacky way to exclude them.
# The deep basin centers are captured by filling holes, but ditches are excluded by the max depth threshold.

max_depth = np.percentile(depression_depth_array[~np.isnan(depression_depth_array)], max_depth_percentile)
print(f'Max depth threshold set to {max_depth:.2f} m')
# 3.1 Plot a histogram of the depression depths to help set thresholds
plt.hist(depression_depth_array[~np.isnan(depression_depth_array) & (depression_depth_array > 0.03)], bins=100)
plt.xlabel('Depression Depth (m)')
plt.ylabel('Frequency')
plt.title('Histogram of Depression Depths')
plt.axvline(min_depth, color='r', linestyle='dashed', linewidth=1)
plt.axvline(max_depth, color='r', linestyle='dashed', linewidth=1)
plt.xlim(0.03, 1)
plt.show()

# 4.2 Assign values of 1 to pixels within the depth range, 0 otherwise
binary_depression_array = (
    (depression_depth_array >= min_depth) & (depression_depth_array <= max_depth)
    ).astype(np.float32)

# %% 4.0 Smooth the binary raster to make shapes more contiguous. Use a wide sigma to convolve the image. 
smoothed_array = ndimage.gaussian_filter(
    binary_depression_array,
    sigma=10,
    radius=25,
)
# 4.1 Write the smoothed raster file to see how it looks
smoothed_depression_depth_path = f'./temp/{site}_depression_depth_smoothed_v4.tif'
with rio.open(depression_depth_path) as src:
    profile = src.profile
    with rio.open(smoothed_depression_depth_path, 'w', **profile) as dst:
        dst.write(smoothed_array, 1)

# %% 5.0 Convert the smoothed depression raster to a vector

with rio.open(smoothed_depression_depth_path) as src:
    profile = src.profile
    crs = src.crs

# Round the smoothed raster to get a binary raster again
depression_mask = np.where(smoothed_array >= smoothed_depression_threshold, 1, 0).astype(np.uint8)
depression_shapes = list(
    shapes(
        depression_mask, 
        mask=depression_mask.astype(bool), 
        transform=profile['transform']
    )
)

geoms = [shape(geom) for geom, value in depression_shapes]
values = [value for geom, value in depression_shapes]

depression_gdf = gpd.GeoDataFrame({'value': values}, geometry=geoms, crs=src.crs)

# %% 6.0 Post-process the depression polygons

# 6.1 Clip to the polygons to the site boundary
boundary = gpd.read_file(boundary_path)
boundary = boundary.to_crs(crs)
depression_gdf = depression_gdf.clip(boundary)

# 6.2 Buffer the polygons by 5m to close gaps.
depression_gdf['geometry'] = depression_gdf['geometry'].buffer(3)

# 6.3 Dissolve all polygons into one, then explode back into individual polygons
collapsed = depression_gdf.geometry.union_all()
depression_gdf = gpd.GeoDataFrame(geometry=[collapsed], crs=crs)
depression_gdf = depression_gdf.explode(index_parts=False).reset_index(drop=True)

# 6.4 Remove holes in polygons
def remove_holes(geom):
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    elif isinstance(geom, MultiPolygon):
        return MultiPolygon([Polygon(part.exterior) for part in geom.geoms])
    else:
        return geom

depression_gdf['geometry'] = depression_gdf['geometry'].apply(remove_holes)

# %% 7.0 Filter polygons based on size thresholds

depression_gdf['area_m2'] = depression_gdf.geometry.area

# Histogram of depression areas
plt.hist(depression_gdf['area_m2'], bins=50)
plt.xlabel('Depression Area (m²)')
plt.ylabel('Frequency')
plt.title('Histogram of Depression Areas')
plt.axvline(min_depression_area, color='r', linestyle='dashed', linewidth=1)
plt.show()

# %%

depression_gdf = depression_gdf[depression_gdf['area_m2'] >= min_depression_area]


depression_gdf.to_file(f'./temp/{site}_depression_polygons_test.shp')

# %% 5.0 Convert the rounded raster to a vector, then filter based on size thresholds

