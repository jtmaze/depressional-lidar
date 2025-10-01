# %% 1.0 Librarie, files paths, and authentication
import ee
import geopandas as gpd
from shapely.geometry import mapping
import json
import datetime
import os
import time

shapefile_path = "D:/depressional_lidar/data/bradford/bradford_boundary.shp"   
start_date = "2021-01-01"                       
end_date   = "2025-09-10"                       
drive_folder = "GEE_ERA5_EXPORTS"               
export_scale = 11132             

ee.Authenticate()  
ee.Initialize()

# %% 2.0 Read the shapefile

# Read shapefile, union geometries and convert to lat-lon GeoJSON
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(epsg=4326)               
geom = mapping(gdf.union_all())           

# Create an ee.Geometry from the GeoJSON mapping
region = ee.Geometry(geom)

# ERA5-Land collection and bands (hourly)
COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"
PRECIP_BAND = "total_precipitation_hourly"      
PET_BAND = "potential_evaporation_hourly"

era5 = ee.ImageCollection(COLLECTION_ID)

# %% 3.0 Functions to group hourly data to daily.

def make_daily_image(day_dt):
    """Create a single daily image (precip sum + PET sum) for a Python datetime.date"""
    start = ee.Date(day_dt.strftime("%Y-%m-%d"))
    end = start.advance(1, 'day')
    daily_precip = era5.filterDate(start, end).select(PRECIP_BAND).sum().rename("precip_m")
    daily_pet = era5.filterDate(start, end).select(PET_BAND).sum().rename("pet_m")
    img = daily_precip.addBands(daily_pet).set('system:time_start', start.millis()).toFloat()
    return img

def daterange(start_date_str, end_date_str):
    start = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    cur = start
    while cur < end:
        yield cur
        cur += datetime.timedelta(days=1)

# %% 4.0 Daily mean values and export as a table

fc_list = []
for day in daterange(start_date, end_date):
    img = make_daily_image(day)
    stats = img.reduceRegion(reducer=ee.Reducer.mean(),
                             geometry=region,
                             scale=export_scale,
                             bestEffort=True,
                             maxPixels=1e13)
    feat = ee.Feature(None, stats).set('date', day.strftime("%Y-%m-%d"))
    fc_list.append(feat)

if fc_list:
    table = ee.FeatureCollection(fc_list)
    csv_task = ee.batch.Export.table.toDrive(
        collection = table,
        description = "ERA5LAND_daily_mean_csv",
        folder = drive_folder,
        fileNamePrefix = "ERA5LAND_daily_mean",
        fileFormat = "CSV"
    )
    csv_task.start()
    print("Started CSV export (region mean) -> task id:", csv_task.id)

# %%
