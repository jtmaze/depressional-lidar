# %% 1.0 Libraries and packages

import numpy as np
import rasterio as rio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import scipy.ndimage as ndi

site = 'delmarva'
data_dir = f'D:/depressional_lidar/data/{site}'

depression_prob_threshold = 0.5
size_thresh = 100        # minimum area in square meters
ditch_half_width = 3.0   # erosion radius in meters; ~half the typical ditch width
ditch_width_thresh = 5.0  # secondary mean-width filter (catches any ditches not severed by erosion)
compact_thresh = 0.8

src_path = f'{data_dir}/out_data/{site}_prob_depressions.tif'
out_path = f'{data_dir}/out_data/{site}_wetland_basins.shp'

# %% 2.0 Read the depression probability and vectorize

with rio.open(src_path) as src:
    prob = src.read(1)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    pixel_size = abs(transform.a)  # assumes pixels are in m
    pixel_area = pixel_size ** 2   

# Threshold to binary mask
mask = (prob >= depression_prob_threshold).astype(np.uint8)
if nodata is not None:
    mask[prob == nodata] = 0

# --- Morphological erosion to sever ditch connections ---
# Eroding by ditch_half_width collapses any feature narrower than a full ditch
# (2 * ditch_half_width) to zero, severing thin connections between basins.
erosion_radius_px = max(1, round(ditch_half_width / pixel_size))
print(f'Erosion radius: {erosion_radius_px} px ({erosion_radius_px * pixel_size:.1f} m)')
r = erosion_radius_px
y_idx, x_idx = np.ogrid[-r:r + 1, -r:r + 1]
disk_kernel = (x_idx ** 2 + y_idx ** 2) <= r ** 2

eroded = ndi.binary_erosion(mask.astype(bool), structure=disk_kernel)

# Label eroded components
struct8 = ndi.generate_binary_structure(2, 2)  # 8-connectivity
eroded_labeled, n_seeds = ndi.label(eroded, structure=struct8)
print(f'{n_seeds} seed components after erosion')

# Vectorize the eroded (pre-filter) shapes
polys = []
for geom, val in shapes(eroded_labeled.astype(np.int32), mask=(eroded_labeled > 0), transform=transform):
    polys.append({'geometry': shape(geom), 'label': int(val)})

gdf = gpd.GeoDataFrame(polys, crs=crs)

# %% 3.0 Compute shape metrics and apply filters

gdf['area_m2']   = gdf.geometry.area
gdf['perimeter'] = gdf.geometry.length
gdf['mean_width'] = 2 * gdf['area_m2'] / gdf['perimeter']
gdf['compact']    = (4 * np.pi * gdf['area_m2']) / (gdf['perimeter'] ** 2)

# Filter 1: minimum area (on eroded shapes — slightly smaller than true basins)
gdf = gdf[gdf['area_m2'] >= size_thresh].copy()
print(f'{len(gdf)} polygons after area filter (>= {size_thresh} m²)')

# Filter 2: ditch remnants — eroded ditches that weren't fully collapsed
is_ditch = (gdf['mean_width'] < ditch_width_thresh) & (gdf['compact'] < compact_thresh)
gdf = gdf[~is_ditch].copy()
print(f'{len(gdf)} polygons after ditch filter (mean_width < {ditch_width_thresh} m)')

# --- Buffer back out by the erosion radius to restore original footprints ---
# Two depressions closer than 2 * ditch_half_width may overlap after buffering.
# Dissolving and re-exploding merges only those that genuinely touch, keeping
# truly separate basins distinct.
gdf['geometry'] = gdf.geometry.buffer(ditch_half_width)
gdf = gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
print(f'{len(gdf)} polygons after buffer + dissolve')

# %% 4.0 Recompute metrics on final buffered shapes and calculate centroids

gdf['area_m2']    = gdf.geometry.area
gdf['perimeter']  = gdf.geometry.length
gdf['mean_width'] = 2 * gdf['area_m2'] / gdf['perimeter']
gdf['compact']    = (4 * np.pi * gdf['area_m2']) / (gdf['perimeter'] ** 2)
gdf['centroid_x'] = gdf.geometry.centroid.x
gdf['centroid_y'] = gdf.geometry.centroid.y

# %% 5.0 Write the output shapefile

gdf = gdf.reset_index(drop=True)
gdf.to_file(out_path)
print(f'Wrote {len(gdf)} basins to {out_path}')


# %%
