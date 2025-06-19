# %% 
import os
import pandas as pd
import geopandas as gpd
from gw_interpolation import pykrige_constructor

os.chdir('./delmarva/')

well_timeseries = pd.read_csv('')
well_coords = pd.read_csv('')
catchment_boundary = gpd.read_file('')

# %% Generate well points for interpolation

summary_wl = pykrige_constructor.WellsWaterLevel(
    df=well_timeseries,
    begin_obs='2021-03-16',
    end_obs='2021-03-16'
)

print(f'{summary_wl.well_count} wells avaible for Kriging')

wl_points = summary_wl.merge_well_coords(
    well_coords,
    lat_col='Latitude',
    long_col='Longitude',
    epsg_code='26917'
)

# %% Attempt Kriging
pykrige_constructor.InterpolationResult(
    wl_points,
    catchment_boundary, 
)