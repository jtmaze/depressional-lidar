# %% 1.0 Libraries and file paths

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

tgt_id = '14_500'
lai_buffer_dist = 150
inundation_mapping_dist = 200
data_set = 'no_dry_days'

data_dir = data_dir = "D:/depressional_lidar/data/bradford/"

distributions_path = data_dir + f'/out_data/modeled_logging_stages/all_wells_hypothetical_distributions_LAI{lai_buffer_dist}m_domain_{data_set}.csv'
wetland_pairs_path = data_dir + f'out_data/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv'

source_dem_path = data_dir + '/in_data/bradford_DEM_cleaned_veg.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
shift_results_path = data_dir + f'/out_data/modeled_logging_stages/all_wells_shift_results_LAI{lai_buffer_dist}m_domain_{data_set}.csv'

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wetland_utilities.basin_attributes import WetlandBasin
from wetland_utilities.basin_dynamics import BasinDynamics, WellStageTimeseries

# %% 2.0 Establish the Wetland BasinClass

well_pt = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)
well_pt = well_pt[well_pt['wetland_id'] == tgt_id]

basin = WetlandBasin(
    wetland_id=tgt_id,
    source_dem_path=source_dem_path,
    footprint=None,
    well_point_info=well_pt,
    transect_buffer=inundation_mapping_dist
)

# %% 3.0 Figure out the proportion of days the well was bottomed out

dry_days = pd.read_csv(shift_results_path)
print(dry_days)
dry_days = dry_days[
    (dry_days['log_id'] == tgt_id) &
    (dry_days['data_set'] == data_set) &
    (dry_days['model_type'] == 'ols') 
][['log_id', 'ref_id', 'total_obs', 'n_bottomed_out', 'filtered_domain_days']].copy()

dry_days['dry_proportion'] = 1 - (dry_days['filtered_domain_days'] / dry_days['total_obs'])

dry_proportion = dry_days['dry_proportion'].mean()

print(dry_proportion)
# %% 3.0 Read the modeled distributions and establish the wetland basin class

distributions = pd.read_csv(distributions_path)
distributions_clean = distributions[
        (distributions['pre'] >= -1) & (distributions['pre'] <= 1) &
        (distributions['post'] >= -1) & (distributions['post'] <= 1) &
        (distributions['log_id'] == tgt_id)
].copy()

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
    well_to_dem_offset=0.5
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
    well_to_dem_offset=0.5
)
post_dynamics.map_inundation_stacks(
    inundation_frequency=None,
    show_basin_footprint=False,
    cbar_min=0,
    cbar_max=100
)                                

# %% 4.0 Make Nominal and Relative Inundation Change Map

# Pre-logging inundation frequency
pre_stacks = pre_dynamics.calculate_inundation_stacks()
pre_stack = np.stack(list(pre_stacks.values())).astype(np.float32)
pre_map = np.nansum(pre_stack, axis=0) / pre_stack.shape[0]

# Post-logging inundation frequency
post_stacks = post_dynamics.calculate_inundation_stacks()
post_stack = np.stack(list(post_stacks.values())).astype(np.float32)
post_map = np.nansum(post_stack, axis=0) / post_stack.shape[0]

# Nominal change (post - pre), as percentage points
nominal_change = (post_map - pre_map) * 100

# Relative change (post - pre) / pre, guard against division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    relative_change = np.where(
        pre_map > 0,
        ((post_map - pre_map) / pre_map) * 100,
        np.nan
    )

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

# Create single colormap with grey for NaN values
cmap = plt.cm.RdBu.copy()
cmap.set_bad('white')

nom_lim = np.nanmax(np.abs(nominal_change))
im0 = axes[0].imshow(nominal_change, cmap=cmap, vmin=-nom_lim, vmax=nom_lim)
axes[0].set_title(f'Nominal Change (%)')
axes[0].set_facecolor('white')
fig.colorbar(im0, ax=axes[0], label='Δ Inundation Freq (%)')

rel_lim = np.nanpercentile(np.abs(relative_change[np.isfinite(relative_change)]), 95)
im1 = axes[1].imshow(relative_change, cmap=cmap, vmin=-rel_lim, vmax=rel_lim)
axes[1].set_title(f'Relative Change (%)')
axes[1].set_facecolor('white')
fig.colorbar(im1, ax=axes[1], label='Relative to Pre')

for ax in axes:
    ax.set_axis_off()

plt.tight_layout()
plt.show()

# %%
