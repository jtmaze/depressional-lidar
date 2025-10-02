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
time_zone = "America/New_York"               
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

def make_daily_image_local(date_str, timezone=time_zone):
    """
    date_str: 'YYYY-MM-DD' (local date in the provided timezone)
    timezone: IANA timezone name, e.g. 'America/New_York'
    Returns: ee.Image with bands precip_m and pet_m for that local calendar day
    """
    # Parse local midnight into an ee.Date (this yields an absolute instant in UTC)
    start = ee.Date.parse('YYYY-MM-dd', date_str, timezone)
    end = start.advance(1, 'day')
    daily_precip = era5.filterDate(start, end).select(PRECIP_BAND).sum().rename("precip_m")
    daily_pet   = era5.filterDate(start, end).select(PET_BAND).sum().rename("pet_m")
    img = daily_precip.addBands(daily_pet).set('system:time_start', start.millis()).toFloat()
    return img

def local_daterange_strings(start_date_str, end_date_str):
    """
    Yields local date strings 'YYYY-MM-DD' from start_date (inclusive)
    to end_date (exclusive). These are the *local* calendar dates you want to
    treat as the grouping unit.
    """
    start = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    cur = start
    while cur < end:
        yield cur.strftime("%Y-%m-%d")
        cur += datetime.timedelta(days=1)

# %% 4.0 Daily mean values and export as a table

fc_list = []
for date_str in local_daterange_strings(start_date, end_date):
    img = make_daily_image_local(date_str, timezone=time_zone)
    stats = img.reduceRegion(reducer=ee.Reducer.mean(),
                             geometry=region,
                             scale=export_scale,
                             bestEffort=True,
                             maxPixels=1e13)
    feat = ee.Feature(None, stats).set('date_local', date_str)
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
