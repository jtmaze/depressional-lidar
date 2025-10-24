# %% 
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from basin_attributes import WetlandBasin
from basin_dynamics import BasinDynamics, WellStageTimeseries

site = 'bradford'
wetland_id = '13_271'

source_dem = f'D:/depressional_lidar/data/{site}/in_data/{site}_DEM_cleaned_veg.tif'
basins_path = f'D:/depressional_lidar/data/{site}/in_data/{site}_basins_assigned_wetland_ids_KG.shp'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
well_stage_path = f'D:/depressional_lidar/data/{site}/in_data/stage_data/daily_waterlevel_Fall2025.csv'

footprint = gpd.read_file(basins_path)
footprint = footprint[footprint['wetland_id'] == wetland_id]
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
    .rename(columns={'rtk_elevat': 'rtk_elevation'})
    .query("type in ['core_well', 'wetland_well'] and wetland_id == @wetland_id")
)

# %%

basin = WetlandBasin(
    wetland_id=wetland_id,
    source_dem_path=source_dem,
    footprint=footprint,
    well_point_info=well_point,
    transect_method=None,
    transect_n=None,
    transect_buffer=30
)
well_elevation = basin.well_point.elevation_dem
hypsometry = basin.calculate_hypsometry(method='pct_trim_pdf')
hypsometry_cdf = basin.calculate_hypsometry(method='pct_trim_cdf')
hypsometry = pd.DataFrame({'area': hypsometry[0], 'elevation': hypsometry[1]})
hypsometry_cdf = pd.DataFrame({'area': hypsometry_cdf[0], 'elevation': hypsometry_cdf[1]})
basin_low_elevation = hypsometry['elevation'].min()
hypsometry['depth'] = hypsometry['elevation'] - basin_low_elevation
hypsometry_cdf['depth'] = hypsometry_cdf['elevation'] - basin_low_elevation


def calc_tai_from_hypsometry(hypsometry_df, lower_step, upper_step):

    tai_areas = []
    tai_percents = []

    for idx, row in hypsometry_df.iterrows():
        wtr_elevation = row['elevation']
        lower = wtr_elevation + lower_step
        upper = wtr_elevation + upper_step

        msk_tai = ((hypsometry_df['elevation'] >= lower) &
               (hypsometry_df['elevation'] <= upper))
        tai_area = hypsometry_df.loc[msk_tai, 'area'].sum()
        flooded = hypsometry_df[hypsometry_df['elevation'] < wtr_elevation]
        total_flooded = flooded['area'].sum()
        tai_percent = tai_area / total_flooded * 100
        tai_areas.append(tai_area)
        tai_percents.append(tai_percent)

    out_df = hypsometry_df.copy()
    out_df['tai_area'] = tai_areas
    out_df['tai_percent'] = tai_percents
    out_df['tai_percent'].replace(np.inf, 0, inplace=True)

    return out_df

hypsometry = calc_tai_from_hypsometry(hypsometry, lower_step=-0.05, upper_step=0.05)

timeseries = WellStageTimeseries.from_csv(
    well_stage_path, 
    well_id=wetland_id,
    basin=basin,
    date_column='day',
    water_level_column='well_depth', 
    well_id_column='well_id'
)

ts = timeseries.timeseries_data

ts['water_depth'] = (ts['water_level'] + well_elevation - basin_low_elevation).round(2)

dynamics = BasinDynamics(
    basin=basin,
    well_stage=timeseries,
    well_to_dem_offset=0
)
dynamics.plot_tai_area_histogram(max_depth=0.05, min_depth=-0.05, as_pct=True)
dynamics.map_tai_stacks(max_depth=0.05, min_depth=-0.05, show_basin_footprint=True)
dynamics.map_inundation_stacks(show_basin_footprint=True)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7, 7))

# Simple CDF of basin elevation (spatial), weighted by area
hs = hypsometry[['depth', 'area']].dropna().sort_values('depth')
cdf = hs['area'].cumsum() / hs['area'].sum()
ax1.plot(hs['depth'], cdf, color="#2ca02c", label="Basin Elevation CDF (Spatial)")
ax1.set_ylabel("CDF")
ax1.legend()

sns.kdeplot(
    x=hypsometry['depth'],
    weights=hypsometry['area'],
    fill=True,
    color="#ff7f0e",
    bw_adjust=0.8,
    ax=ax2,
    label="Basin Elevation PDF (Spatial)"
)
ax2.set_ylabel("PDF (Not Normalized)")
ax2.legend()

water_depths = ts['water_depth'].dropna()
sns.kdeplot(
    x=water_depths,
    fill=True,
    color="#1f77b4",
    bw_adjust=0.8,
    ax=ax3,
    label="Water Depth (Time)"
)
ax3.legend()
ax3.set_ylabel("PDF (Not Normalized)")
plt.tight_layout()
plt.show()

# %%
