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
well_stage_path = f'D:/depressional_lidar/data/{site}/in_data/stage_data/waterlevel_offsets_tracked.csv'

basin_footprints = gpd.read_file(basins_path)
well_points = gpd.read_file(well_points_path)
print(well_points['type'].unique())
well_points = well_points[(well_points['site'] == site) & 
                          ((well_points['type'] == 'core_well') | (well_points['type'] == 'wetland_well'))]
                           

# %% 1.1 Clean up the well points gdf

well_points.rename(
        columns={
            'rtk_elevat': 'rtk_elevation'
        },
        inplace=True
    )

# %% 1.2 Make a list of wetland_ids to calculate timeseries
wetland_ids = [
    '14_612', '15_409', '14_500', '13_410', '13_267'
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
        transect_buffer=20
    )

    well_stage = WellStageTimeseries.from_csv(
        well_stage_path,
        well_id=i,
        date_column='Date',
        water_level_column='revised_depth',
        well_id_column='Site_ID'
    )

    dynamics = BasinDynamics(
        basin=b,
        well_stage=well_stage,
        well_to_dem_offset=0
    )

    area_ts = dynamics.calculate_inundated_area_timeseries()
    tai_ts = dynamics.calculate_tai_timeseries(min_depth=-0.05, max_depth=0.05)
    dynamics.map_tai_stacks(max_depth=0.05, min_depth=-0.05)
    
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

# %% 5.1 Make a boxplot's showing 'TAI area' as a fraction of wetland's max area
tai_fraction_data = []
wetland_labels = []

for wid, ats in area_ts_dict.items():
    area = ats.values
    tts = tai_ts_dict[wid]
    tai = tts.values

    max_a = np.nanmax(area)
    tai_adj = tai / max_a * 100
    
    # Add to lists for plotting
    tai_fraction_data.extend(tai_adj)
    wetland_labels.extend([wid] * len(tai_adj))

# Create DataFrame for easier plotting
boxplot_df = pd.DataFrame({
    'tai_fraction': tai_fraction_data,
    'wetland_id': wetland_labels
})

# Create the boxplot
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=boxplot_df, x='wetland_id', y='tai_fraction', ax=ax)

plt.xlabel('', fontsize=16)
plt.ylabel('TAI Area % Relative to Max Wetland Area', fontsize=16)
plt.title('TAI Area % of Maximum Wetland Area')
plt.xticks(rotation=15, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# %% 5.2 Boxplot showing TAI area relative to wetland area at time (t)

tai_fraction_data = []
wetland_labels = []

for wid, ats in area_ts_dict.items():
    area = ats.values
    tts = tai_ts_dict[wid]
    tai = tts.values

    max_a = np.nanmax(area)
    tai_adj = tai / area * 100
    
    # Add to lists for plotting
    tai_fraction_data.extend(tai_adj)
    wetland_labels.extend([wid] * len(tai_adj))

# Create DataFrame for easier plotting
boxplot_df = pd.DataFrame({
    'tai_fraction': tai_fraction_data,
    'wetland_id': wetland_labels
})

# Create the boxplot with outliers removed
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=boxplot_df, x='wetland_id', y='tai_fraction', ax=ax, color='orange', showfliers=False)

plt.ylabel('TAI Area % Relative Wetland A(t)')
plt.title('TAI Area % Relative to Wetland A(t)')
plt.xticks(rotation=15)
plt.xlabel('')

plt.tight_layout()
plt.show()

# %% 5.3 Scatter plot of TAI area and wetland area for each wetland

fig, ax = plt.subplots(figsize=(10, 8))

# Create scatter plot for each wetland
for wid in wetland_ids:
    area = area_ts_dict[wid].values
    tai = tai_ts_dict[wid].values

    min_a = np.nanmin(area)
    max_a = np.nanmax(area)
    print(min_a, max_a)
    area_scaled = (area - min_a) / (max_a - min_a)

    min_tai = np.nanmin(tai)
    max_tai = np.nanmax(tai)
    tai_scaled = (tai -  min_tai) / (max_tai - min_tai)
    # Filter out zero area values if desired
    
    ax.scatter(area_scaled, tai_scaled, label=wid, alpha=0.6, s=30)

plt.xlabel('Wetland Area (Scaled 0-1)')
plt.ylabel('TAI Area (Scaled 0-1)')
plt.title('TAI Area vs Wetland Area by Wetland ID')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
