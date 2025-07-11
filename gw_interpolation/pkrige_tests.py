# %% 
import os
import sys
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gw_interpolation import pykrige_constructor

os.chdir('/Users/jmaze/Documents/projects/depressional_lidar/')

catchment = 'BC'

# %%

well_timeseries = pd.read_csv('./delmarva/waterlevel_data/output_JM_2019_2022.csv').rename(
    columns={'Timestamp': 'Date', 'Site_Name': 'SiteID'}
)
# Convert Date column to datetime
well_timeseries['Date'] = pd.to_datetime(well_timeseries['Date'])

jl_sites = [
    'TS-CH', 'ND-UW3', 'ND-SW', 'DK-SW', 'DK-UW1', 'BD-CH', 'BD-SW',
    'DK-CH', 'DK-UW2', 'ND-UW1', 'ND-UW2', 'TS-SW', 'TS-UW1'
]

jl_elevation_dict = {
    'TS-UW1': 194.5,
    'TS-SW': 72.5,
    'TS-CH': 101.5,
    'BD-CH': 139.5,
    'BD-SW': 80.5,
    'ND-UW3': 165,
    'ND-UW2': 198,
    'ND-UW1': 217.5,
    'ND-SW': 75.5,
    'DK-UW1': 119.5,
    'DK-CH': 59.5,
    'DK-UW2': 89.5,
    'DK-SW': 0
}

jr_sites = [
    'OB-UW1', 'MB-SW', 'MB-CH', 'XB-UW1', 'XB-SW', 'XB-CH',
    'HB-OT', 'HB-SW', 'HB-UW1', 'HB-CH', 'TP-CH', 'OB-SW',
    'OB-CH', 'MB-UW1'
]

jr_elevation_dict = {
    'OB-UW1': 144.5,
    'OB-SW': 11.5,
    'OB-CH': 56.5,
    'MB-UW1': 130.5,
    'MB-SW': -40.5,
    'MB-CH': 50.5,
    'XB-UW1': 175.5,
    'XB-SW': 18.5,
    'XB-CH': 56.5,
    'HB-SW': -42,
    'HB-UW1': 103,
    'HB-CH': 19,
    'TP-CH': 0
}


if catchment == 'JL':
    sites = jl_sites
    boundary_path = './delmarva/site_boundries/JL_toy_bounds.shp'
    elevation_dict = jl_elevation_dict
else:
    sites = jr_sites
    boundary_path = './delmarva/site_boundries/JR_toy_bounds.shp'
    elevation_dict = jr_elevation_dict

well_timeseries = well_timeseries[well_timeseries['SiteID'].isin(sites)]
well_timeseries = well_timeseries[['SiteID', 'Flag', 'Date', 'waterLevel']]
# %%
well_coords = gpd.read_file('./delmarva/trimble_well_pts.shp').rename(
    columns={'Descriptio': 'SiteID'}
)
well_coords = well_coords[['SiteID', 'geometry']]
# Clean up SiteID by removing "well" and "WELL" suffixes
well_coords['SiteID'] = well_coords['SiteID'].str.replace(' well', '', case=False)
well_coords['SiteID'] = well_coords['SiteID'].str.replace(' WELL', '', case=False)

well_coords.to_crs('EPSG:4326', inplace=True)
well_coords['latitude'] = well_coords.geometry.y
well_coords['longitude'] = well_coords.geometry.x

# Add elevation to well_coords
well_coords['Elevation'] = well_coords['SiteID'].map(elevation_dict)

# %%

catchment_boundary = gpd.read_file(boundary_path)
print(catchment_boundary.crs)

# %% Generate well points for interpolation

summary_wl = pykrige_constructor.WellsWaterLevel(
    df=well_timeseries,
    begin_obs='2022-3-5',
    end_obs='2022-4-20',
    well_type='UW_CH'
)

print(f'{summary_wl.well_count} wells avaible for Kriging')


# %%
wl_points = summary_wl.merge_well_coords(
    well_coords,
    lat_col='latitude',
    lon_col='longitude',
    epsg_code='4326'
)

# %% Attempt Kriging

gridded_samples = pykrige_constructor.InterpolationResult(
    wl_points,
    catchment_boundary 
)
# %%

gridded_samples.ordinary_kriging(
    variogram_model='linear',
    nlags=4,
    plot_variogram=True
)

# %%

gridded_samples.plot_interpolation_result()
gridded_samples.plot_sigma_squared()
gridded_samples.plot_masked_result(sigma_squared_threshold=0.5)

# %%
