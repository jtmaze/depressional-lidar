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
well_stage_path = f'D:/depressional_lidar/data/{site}/in_data/stage_data/bradford_wells_tracked_datum.csv'

basin_footprints = gpd.read_file(basins_path)
well_points = gpd.read_file(well_points_path)
print(well_points['type'].unique())
well_points = well_points[(well_points['site'] == site) & 
                          ((well_points['type'] == 'core_well') | (well_points['type'] == 'wetland_well'))
]

well_id = '13_267'

# %% 1.1 Clean up the well points gdf

well_points.rename(
        columns={
            'rtk_elevat': 'rtk_elevation'
        },
        inplace=True
)

# %% 2.0 Set up the wetland basin

footprint = basin_footprints[basin_footprints['wetland_id'] == well_id]
pt = well_points[well_points['wetland_id'] == well_id]
basin = WetlandBasin(
    wetland_id=well_id,
    source_dem_path=source_dem,
    footprint=footprint,
    well_point_info=pt,
    transect_buffer=30
)

well_stage = WellStageTimeseries.from_csv(
    well_stage_path,
    well_id=well_id,
    date_column='Date',
    water_level_column='revised_depth',
    well_id_column='Site_ID'
)

# %% 3.0 Iterate to test sensitivity to illustrate elevation error

vals = np.arange(-0.2, 0.25, 0.1)
print(vals)

area_ts_dicts = {}
tai_ts_dicts = {}

for v in vals:
    dynamics = BasinDynamics(
        basin=basin,
        well_stage=well_stage,
        well_to_dem_offset=v
    )
    print(v)
    area_ts = dynamics.calculate_inundated_area_timeseries()
    tai_ts = dynamics.calculate_tai_timeseries(max_depth=0.05, min_depth=-0.05)

    dynamics.map_inundation_stacks()
    dynamics.map_tai_stacks(max_depth=0.05, min_depth=-0.05)
    print(v)
    area_ts_dicts[v] = area_ts
    tai_ts_dicts[v] = tai_ts


# %% 4.0 Plot the Area PDFs to illustrate sensitivity

fig, ax = plt.subplots(figsize=(8, 6))
for offset, ats in area_ts_dicts.items():
    area = ats.values

    sns.kdeplot(area, label=f"{offset:.2f}", ax=ax, fill=True, alpha=0.3)
    plt.xlabel('Inundated Area (m2)')
    plt.ylabel('Density')
    plt.title(f'{well_id} PDF of Inundated Area by Well Elevation Bias')
    plt.legend()
    plt.tight_layout()

# %%

fig, ax = plt.subplots(figsize=(8, 6))
for offset, ats in tai_ts_dicts.items():
    tai_values = ats.values

    sns.kdeplot(tai_values, label=f"{offset:.2f}", ax=ax, fill=True, alpha=0.3)
    plt.xlabel('TAI (m2)')
    plt.ylabel('Density')
    plt.title(f'{well_id} PDF of TAI by Well Elevation Bias')
    plt.legend()
    plt.tight_layout()
# %%
