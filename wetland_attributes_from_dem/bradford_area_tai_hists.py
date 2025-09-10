# %%  1.0 Libraries and File Paths

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from basin_attributes import WetlandBasin
from basin_dynamics import BasinDynamics, WellStageTimeseries


site = 'bradford'
source_dem = f'D:/depressional_lidar/data/{site}/in_data/{site}_DEM_cleaned_veg.tif'
basins_path = f'D:/depressional_lidar/data/{site}/in_data/{site}_basins_assigned_wetland_ids.shp'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
well_stage_path = f'D:/depressional_lidar/data/{site}/in_data/stage_data/{site}_core_wells_tracked_datum.csv'

basin_footprints = gpd.read_file(basins_path)
well_points = gpd.read_file(well_points_path)
well_points = well_points[(well_points['site'] == 'osbs') & (well_points['type'] == 'core_well')]

# %% 1.1 Clean up the well points gdf

well_points.rename(
        columns={
            'rtk_elevat': 'rtk_elevation'
        },
        inplace=True
    )

# %% 1.2 Make a list of wetland_ids to calculate timeseries
wetland_ids = [
    'Ross', 'Brantley North', 'Devils Den', 
    'West Ford', 'Fish Cove'
]

area_ts_dict = {}
tai_ts_dict = {}

# %% 2.0 Run the BasinDynamics Class for each wetland_id

for i in wetland_ids:
    f = basin_footprints[basin_footprints['wetland_id'] == i]
    pt = well_points[well_points['wetland_id'] == i]
    b = WetlandBasin(
        wetland_id=i,
        source_dem_path=source_dem,
        footprint=f,
        well_point_info=pt,
        transect_buffer=25
    )

    well_stage = WellStageTimeseries.from_csv(
        well_stage_path,
        well_id=i,
        date_column='date',
        water_level_column='water_level',
        well_id_column='well_id'
    )

    dynamics = BasinDynamics(
        basin=b,
        well_stage=well_stage,
        well_to_dem_offset=0
    )

    area_ts = dynamics.calculate_inundated_area_timeseries()
    tai_ts = dynamics.calculate_tai_timeseries(min_depth=-0.05, max_depth=0.05)
    
    area_ts_dict[i] = area_ts
    tai_ts_dict[i] = tai_ts


# %% 3.0 PDF of inundated Area by Wetland

fig, ax = plt.subplots(figsize=(8, 6))
for wid, ats in area_ts_dict.items():
    area = ats.values
    area = area[area > 0]
    # Removing zero area for now
    sns.kdeplot(area, label=wid, ax=ax, fill=True, alpha=0.3)

plt.xlabel('Inundated Area (m2)')
plt.ylabel('Density')
plt.title('PDF of Inundated Area by Wetland')
plt.legend()
plt.tight_layout()

# %% 3.1 PDF of inundated area by Wetland rescaled 0-1

fig, ax = plt.subplots(figsize=(8, 6))
for wid, ats in area_ts_dict.items():
    area = ats.values
    area = area[area > 0]
    min_a = np.nanmin(area)
    max_a = np.nanmax(area)
    print(min_a, max_a)
    area_scaled = (area - min_a) / (max_a - min_a)
    # Removing zero area for now
    sns.kdeplot(area_scaled, label=wid, ax=ax, fill=True, alpha=0.3)

plt.xlabel('Inundated Area (rescaled 0-1)')
plt.ylabel('Density')
plt.title('PDF (rescaled 0-1)')
plt.legend()
plt.tight_layout()

# %% 4.1 PDF of TAI area by Wetland 

fig, ax = plt.subplots(figsize=(8, 6))
for wid, tts in tai_ts_dict.items():
    tai = tts.values
    #tai = tai[tai > 0]
    # Removing zero area for now
    sns.kdeplot(tai, label=wid, ax=ax, fill=True, alpha=0.3)

plt.xlabel('TAI Area (5cm to -5cm) (m2)')
plt.ylabel('Density')
plt.title('PDF of TAI Area by Wetland')
plt.legend()
plt.tight_layout()

# %% 4.2 Wetland TAI PDFs rescaled 0-1

fig, ax = plt.subplots(figsize=(8, 6))
for wid, tts in tai_ts_dict.items():
    tai = tts.values
    #tai = tai[tai > 0]
    min_tai = np.nanmin(tai)
    max_tai = np.nanmax(tai)
    tai_scaled = (tai - min_tai) / (max_tai - min_tai)

    sns.kdeplot(tai_scaled, label=wid, ax=ax, fill=True, alpha=0.3)

plt.xlabel('TAI (Rescaled 0-1)')
plt.ylabel('Density')
plt.title('PDF of TAI (Rescaled 0-1)')
plt.legend()
plt.tight_layout()

# %%
