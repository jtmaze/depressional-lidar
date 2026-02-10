# %% 1.0 Libraries and file paths

import ee 
import geemap
import geopandas as gpd
import pandas as pd

ee.Authenticate()
ee.Initialize()

bradford_shapefile = gpd.read_file("D:/depressional_lidar/data/bradford/bradford_boundary.shp")
bradford_shapefile = bradford_shapefile.to_crs(epsg=4326)

# %% 2.0 Generate LAI map for a given date range cropped to bradford boundary

# Helpter functions

def apply_scale_factors(image):
    optical = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True)

def rename_ls8_bands(image):
    bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']
    new_names = ['B', 'G', 'R', 'NIR']
    return image.select(bands).rename(new_names)

def calc_lai(image):
        SR = image.select('NIR').divide(image.select('R'))
        LAI = image.expression('(0.332915 * SR) - 0.00212', {'SR': SR}).rename('LAI').clamp(0, 5.6)
        return image.addBands(LAI)

def mask_clouds(image):
    qa = image.select('QA_PIXEL')
    cloudsBitMask = (1 << 3)
    cloudShadowBitMask = (1 << 4)
    cirrusBitMask = (1 << 2)
    dilatedBitMask = (1 << 1)
    mask = (qa.bitwiseAnd(cloudsBitMask).eq(0)
            .And(qa.bitwiseAnd(cloudShadowBitMask).eq(0))
            .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            .And(qa.bitwiseAnd(dilatedBitMask).eq(0)))
    return image.updateMask(mask)

def make_ls8_collection(polygon, start_date, end_date):
    """Create Landsat 8 collection - clip deferred until after reduction."""
    ls8_full = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

    ls8 = (ls8_full
        .filterDate(start_date, end_date)
        .filterMetadata('CLOUD_COVER', 'less_than', 90)
        .filterBounds(polygon)
        .map(mask_clouds)
        .map(apply_scale_factors)
        .map(rename_ls8_bands)
    )
    return ls8

def calc_lai_composite(collection, polygon):
    """Calculate LAI and create composite - all server-side."""
    def add_lai(image):
        SR = image.select('NIR').divide(image.select('R'))
        LAI = SR.multiply(0.332915).subtract(0.00212).rename('LAI').clamp(0, 5.6)
        return LAI
    
    # Map LAI calculation, reduce to median, then clip once at the end
    lai_composite = (collection
        .map(add_lai)
        .median()
        .clip(polygon)
    )
    return lai_composite

# %% 3.0 Create LAI map

# Convert shapefile to ee.Geometry
polygon = geemap.geopandas_to_ee(bradford_shapefile).geometry()

# Set date range
start_date = '2025-06-01'
end_date = '2025-12-31'

# Create Landsat 8 collection and calculate LAI composite
ls8_collection = make_ls8_collection(polygon, start_date, end_date)
lai_composite = calc_lai_composite(ls8_collection, polygon)

# Check collection size (for debugging)
print(f"Number of images in collection: {ls8_collection.size().getInfo()}")

# %% 4.0 Display LAI map in geemap

Map = geemap.Map(basemap='Esri.WorldImagery')
Map.centerObject(polygon, zoom=12)

Map.addLayer(polygon, {'color': 'black'}, 'Study Area Boundary')

lai_vis = {
    'min': 0,
    'max': 5,
    'palette': ['#ff0000', '#ff8000', '#ffff00', '#80ff00', '#00ff00']
}

Map.addLayer(lai_composite, lai_vis, f'LAI ({start_date} to {end_date})')
Map.add_colorbar(lai_vis, label='Leaf Area Index (LAI)')

Map

# %% 5.0 Function to write composite image to Google Drive

drive_folder = 'landsat_lai_image_exports'

def export_composite_to_drive(image, region, description, folder, scale=30, crs='EPSG:4326'):
    """
    Quick export function
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"Export started: '{description}' -> Drive folder '{folder}'")
    return task

# Example usage:
task = export_composite_to_drive(
    lai_composite, polygon,
    description=f'LAI_composite_{start_date}_to_{end_date}',
    folder=drive_folder,
)

# %%
