# %% Libraries and file paths

import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from basin_attributes import WetlandBasin
from basin_dynamics import WellStageTimeseries


wetlands_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_stage_path = f'D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv'

wetlands = gpd.read_file(wetlands_path)
well_points = gpd.read_file(well_points_path)
well_points.rename(
    columns={'rtk_elevat': 'rtk_elevation'},
    inplace=True,
)
well_points = well_points[(well_points['type'] == 'core_well') | (well_points['type'] == 'wetland_well')]

unique_ids = wetlands['wetland_id'].unique()

ids_list = [
    '14_115', '14_418', '15_409', '5_597', '3_638', '3_34', '6_93', '6a_530',
    '9_508', '13_410', '13_271', '13_267', '13_263', '13_274', '5_510', '5_161',
     '14_610', '14_616'
]

# %% 

curves = []

for i in ids_list:

    well_pt = well_points[well_points['wetland_id'] == i]
    footprint = wetlands[wetlands['wetland_id'] == i]

    basin = WetlandBasin(
        wetland_id=i,
        source_dem_path=dem_path,
        footprint=footprint,
        well_point_info=well_pt,
        transect_method=None,
        transect_n=None,
        transect_buffer=50
    )

    cum_area_m2, bin_centers = basin.calculate_hypsometry(method='pct_trim_cdf')

    df_curve = pd.DataFrame({
        'wetland_id': i,
        'elevation': bin_centers,
        'cum_area_m2': cum_area_m2
    })

    curves.append(df_curve)

out_df = pd.concat(curves, ignore_index=True)

# %% Rescale the elevation values to min and max for each wetland

out_df['depth'] = out_df.groupby('wetland_id')['elevation'].transform(
    lambda x: (x - x.min())
)

out_df['area_rescaled'] = out_df.groupby('wetland_id')['cum_area_m2'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# %% Plot the hypsometric curves for all wetlands

fig, ax = plt.subplots(figsize=(6, 6))

unique_wetlands = sorted(out_df['wetland_id'].unique())
n = len(unique_wetlands)

# Use a palette that can generate many distinct colors
if n <= 20:
    palette = sns.color_palette('tab20', n)
else:
    palette = sns.color_palette('husl', n)

color_map = dict(zip(unique_wetlands, palette))

for wetland, group in out_df.groupby('wetland_id'):
    group = group.sort_values('depth')
    ax.plot(group['depth'], group['area_rescaled'], label=wetland, linewidth=2, color=color_map[wetland])

ax.set_xlabel('Depth (m)', fontsize=12)
ax.set_title('Bradford Wetland Hypsometric Curves', fontsize=14)
ax.grid(True, alpha=0.3)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=min(len(unique_wetlands), 4), frameon=False, framealpha=0.3)

fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
plt.show()

# %% Collect the stage data

stages = []

for i in ids_list:

    well_pt = well_points[well_points['wetland_id'] == i]
    footprint = wetlands[wetlands['wetland_id'] == i]

    basin = WetlandBasin(
        wetland_id=i,
        source_dem_path=dem_path,
        footprint=footprint,
        well_point_info=well_pt,
        transect_method=None,
        transect_n=None,
        transect_buffer=50
    )
    

    stage = WellStageTimeseries.from_csv(
        file_path=well_stage_path,
        well_id=i, 
        basin=basin,
        date_column='day',
        water_level_column='well_depth',
        well_id_column='well_id'
    )

    data = stage.timeseries_data
    data['well_id'] = i
    stages.append(data)


# %%

all_stages = pd.concat(stages, ignore_index=True)

# Choose a smaller subset to reduce clutter
plot_ids = [
    '14_115', '14_418', '15_409', '5_597', '3_638', '3_34', '6_93', '6a_530',
    '9_508', '13_410', '13_271', '13_267'
]
max_wells = 8  # adjust to plot fewer/more wells
selected_ids = [wid for wid in plot_ids if wid in all_stages['well_id'].unique()][:max_wells]

# Single plot with overlapping KDEs for selected wells
fig, ax = plt.subplots(figsize=(10, 10))
palette = sns.color_palette('tab10', len(selected_ids)) if len(selected_ids) <= 10 else sns.color_palette('husl', len(selected_ids))

for i, well_id in enumerate(selected_ids):
    well_data = all_stages.loc[all_stages['well_id'] == well_id, 'water_level'].dropna()
    if len(well_data) > 1 and well_data.nunique() > 1:
        sns.kdeplot(x=well_data, label=well_id, color=palette[i], linewidth=2, fill=False, ax=ax)
    elif len(well_data) >= 1:
        ax.axvline(well_data.iloc[0], color=palette[i], linestyle='--', linewidth=2, label=well_id)

ax.set_xlabel('Stage at Well (meters - not adjusted to basin low)')
ax.set_ylabel('Density')
ax.grid(True, alpha=0.3)
ax.legend(title='Well', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
fig.tight_layout()
plt.show()

# %%
