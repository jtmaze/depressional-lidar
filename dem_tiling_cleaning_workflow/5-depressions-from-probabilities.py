# %% 1.0 Libraries and packages

import numpy as np
import rasterio as rio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import scipy.ndimage as ndi
from shapely.geometry import Polygon, MultiPolygon


site = 'bradford'
data_dir = f'D:/depressional_lidar/data/{site}'

depression_prob_threshold = 0.3
seed_thresh = 50       # minimum area in square meters of binary seeds
ditch_half_width = 5   # erosion radius in meters; ~half the typical ditch width
ditch_width_thresh = 5.0  # secondary mean-width filter (catches any ditches not severed by erosion)
hole_fill = 50 # Removes any islands up-to 50 square meters
concavity_thresh = 10 # Fills concave chunks in basin shapes up to X meters
final_min_area = 1000 # The final minimum area of written depressions

src_path = f'{data_dir}/out_data/{site}_prob_depressions_SMOOTH_TEST_0.25_50_25.tif'
out_path = f'{data_dir}/out_data/{site}_wetland_basins_v10.shp'

# %% 2.0 Read the depression probability and vectorize

with rio.open(src_path) as src:
    prob = src.read(1)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    pixel_size = abs(transform.a)  # CRS is always projected meters
    pixel_area = pixel_size ** 2
    print(f'Pixel size: {pixel_size:.4f} m')

# %%

# Threshold depression probabilities to binary mask
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

# Label eroded components as basins without ditch connections
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

# Filter 1: minimum area seed_area
gdf = gdf[gdf['area_m2'] >= seed_thresh].copy()
print(f'{len(gdf)} polygons after seed area filter (>= {seed_thresh} m²)')

# Filter 2: ditch remnants — eroded ditches that weren't fully collapsed
is_ditch = gdf['mean_width'] < ditch_width_thresh
gdf = gdf[~is_ditch].copy()
print(f'{len(gdf)} polygons after ditch filter (mean_width < {ditch_width_thresh} m)')

# --- Buffer back out by the erosion radius to restore original footprints ---
# Two depressions closer than 2 * ditch_half_width may overlap after buffering.
# Dissolving and re-exploding merges only those that genuinely touch, keeping
# truly separate basins distinct.
gdf['geometry'] = gdf.geometry.buffer(ditch_half_width)
gdf = gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
print(f'{len(gdf)} polygons after buffer + dissolve')

# %% 4.0 Correct/simplify the morphologies

def remove_holes(geom, min_hole_area=None):
    """Remove interior rings (holes) from a polygon.
    If min_hole_area is None, removes ALL holes.
    Otherwise, only removes holes smaller than the threshold.
    """
    if geom.geom_type == 'Polygon':
        kept = [r for r in geom.interiors if Polygon(r).area >= min_hole_area]
        return Polygon(geom.exterior, kept)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([remove_holes(p, min_hole_area) for p in geom.geoms])
    return geom

def morphological_close(geom, distance=10):
    """Buffer outward then inward to fill concavities."""
    return geom.buffer(distance).buffer(-distance)

gdf['geometry'] = (
    gdf['geometry']
    .apply(lambda g: remove_holes(g, min_hole_area=hole_fill))   # drop small islands
    .apply(lambda g: morphological_close(g, distance=concavity_thresh))  # fill concavities
)

# %% 5.0 Recompute metrics on final buffered shapes and calculate centroids

gdf['area_m2']    = gdf.geometry.area
gdf['perimeter']  = gdf.geometry.length
gdf['mean_width'] = 2 * gdf['area_m2'] / gdf['perimeter']
gdf['compact']    = (4 * np.pi * gdf['area_m2']) / (gdf['perimeter'] ** 2)
gdf['centroid_x'] = gdf.geometry.centroid.x
gdf['centroid_y'] = gdf.geometry.centroid.y

gdf = gdf[gdf['area_m2'] >= final_min_area]

# %% 6.0 Write the output shapefile

gdf = gdf.reset_index(drop=True)
gdf.to_file(out_path)
print(f'Wrote {len(gdf)} basins to {out_path}')


# %%
