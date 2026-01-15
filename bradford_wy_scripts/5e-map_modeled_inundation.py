# %% 1.0 Libraries and file paths

import sys
import numpy as np
import pandas as pd
import geopandas as gpd

tgt_id = '14_418'

distributions_path = 'D:/depressional_lidar/data/bradford/out_data/logging_hypothetical_distributions.csv'
source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
shift_results_path = 'D:/depressional_lidar/data/bradford/out_data/logging_hypothetical_shift_results.csv'

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wetland_utilities.basin_attributes import WetlandBasin
from wetland_utilities.basin_dynamics import BasinDynamics, WellStageTimeseries

# %% 2.0 Establish the Wetland BasinClass

well_pt = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
    .rename(columns={'rtk_elevat': 'rtk_elevation'})
    .query("type in ['core_well', 'wetland_well']")
)
well_pt = well_pt[well_pt['wetland_id'] == tgt_id]

basin = WetlandBasin(
    wetland_id=tgt_id,
    source_dem_path=source_dem_path,
    footprint=None,
    well_point_info=well_pt,
    transect_buffer=100
)

# %% 3.0 Figure out the proportion of days the well was bottomed out

dry_days = pd.read_csv(shift_results_path)
dry_days = dry_days[
    #(dry_days['log_id'] == tgt_id) &
    (dry_days['data_set'] == 'full') &
    (dry_days['model_type'] == 'ols') 
][['log_id', 'ref_id', 'total_obs', 'n_bottomed_out']].copy()

dry_days['dry_proportion'] = dry_days['n_bottomed_out'] / dry_days['total_obs']

print(dry_days.groupby('log_id')['dry_proportion'].mean())
print(dry_days.groupby('ref_id')['dry_proportion'].mean())

dry_proportion = dry_days['dry_proportion'].mean()

print(dry_proportion)
# %% 3.0 Read the modeled distributions and establish the wetland basin class
distributions = pd.read_csv(distributions_path)
distributions_clean = distributions[
        (distributions['pre'] >= -1) & (distributions['pre'] <= 1) &
        (distributions['post'] >= -1) & (distributions['post'] <= 1) &
        (distributions['log_id'] == tgt_id)
].copy()

# TODO: Figure out if a gaussian KDE is better than just sampling
pre_dist = np.random.choice(
    distributions_clean['pre'].values, 
    size=1_000, 
    replace=True
)

# Set proportion of values to account for bottomed out days not in the model
pre_dist_dry = pre_dist.copy()
n_dry = int(len(pre_dist_dry) * dry_proportion)
dry_indices = np.random.choice(len(pre_dist_dry), size=n_dry, replace=False)
pre_dist_dry[dry_indices] = -2.0

pre_data = pd.DataFrame({'stage': pre_dist_dry})

# Add a dummy column for "date", even though data will not resemble a timeseries
pre_data['date'] = pd.date_range(
    start='2000-01-01', 
    periods=len(pre_data), 
    freq='D'
)
post_dist = np.random.choice(
    distributions_clean['post'].values,
    size=1_000,
    replace=True
)
post_dist_dry = post_dist.copy()
n_dry = int(len(post_dist_dry) * dry_proportion)
dry_indices = np.random.choice(len(post_dist_dry), size=n_dry, replace=False)
post_dist_dry[dry_indices] = -2.0

post_data = pd.DataFrame({'stage': post_dist_dry})
post_data['date'] = pd.date_range(
    start='2000-01-01',
    periods=len(post_data),
    freq='D'
)
# %% 3.0 Generate the BasinDynamics class for pre & post modeled hydrology

# NOTE: The the stage models are already converted to the lowest point with DEM radius
# However, the BasinDynamics class exepects the timeseries as water_level at the 
# well. I convert the modeled stage back to well depth

offset = basin.well_point.elevation_dem - basin.deepest_point.elevation
pre_data['water_level'] = pre_data['stage'] - offset
post_data['water_level'] = post_data['stage'] - offset

pre_ts = WellStageTimeseries(
    well_id=tgt_id,
    timeseries_data=pre_data,
    basin=basin
)
pre_dynamics = BasinDynamics(
    basin=basin,
    well_stage=pre_ts, 
    well_to_dem_offset=0.1
)
pre_dynamics.map_inundation_stacks(
    inundation_frequency=None,
    show_basin_footprint=False,
    cbar_min=0,
    cbar_max=100
)

post_ts = WellStageTimeseries(
    well_id=tgt_id,
    timeseries_data=post_data,
    basin=basin
)
post_dynamics = BasinDynamics(
    basin=basin,
    well_stage=post_ts,
    well_to_dem_offset=0.1
)
post_dynamics.map_inundation_stacks(
    inundation_frequency=None,
    show_basin_footprint=False,
    cbar_min=0,
    cbar_max=100
)
                                   

# %% 