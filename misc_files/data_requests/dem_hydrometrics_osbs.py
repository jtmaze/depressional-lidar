# %% 1.0 Libraries and file paths

import os
import datetime as dt
import pandas as pd
import numpy as np
import geopandas as gpd

os.chdir('D:/depressional_lidar/data/')

points_master = gpd.read_file('./rtk_pts_with_dem_elevations.shp')



core_wells = ['Devils Den', 'West Ford', 'Brantley North', 'Ross Pond', 'Fish Cove', 'Surprise']

osbs_core_wells = pd.read_csv('./osbs/in_data/stage_data/daily_well_depth_Fall2025.csv')
osbs_core_wells = osbs_core_wells[
    osbs_core_wells['well_id'].isin(core_wells)
]

# NOTE: only using post 2022 data due to coverage discrepancies
osbs_core_wells = osbs_core_wells[osbs_core_wells['date'] >= '2022-03-09']
print(osbs_core_wells['flag'].unique())

# Quick check on osbs core well timeseries:
# import plotly.express as px

# # Create interactive timeseries
# fig = px.line(
#     osbs_core_wells.sort_values('date'),
#     x='date',
#     y='well_depth_m',
#     color='well_id',
#     title='Well Depth Over Time',
#     markers=True
# )

# fig.show()

# %% 2.0 List out osbs wetlands, types and locations

points_osbs = points_master[points_master['site'] == 'osbs']
wetlands_osbs = points_osbs['wetland_id'].unique()
print(wetlands_osbs)
types = points_osbs['type'].unique()
locations = points_osbs['location'].unique()
print(types)

# Remove wetland and core_well types, only interested in sampling locations
remove_types = ['core_well', 'wetland_well']
types = [t for t in types if t not in remove_types]
print(types)

# %% Line plot of well depth at Ross Pond

import matplotlib.pyplot as plt

ross_pond = osbs_core_wells[osbs_core_wells['well_id'] == 'Ross Pond'].copy()
ross_pond['date'] = pd.to_datetime(ross_pond['date'])
plt.figure(figsize=(12, 6))
plt.plot(ross_pond['date'], ross_pond['well_depth_m'], linewidth=1.5, color='#006994')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Ground surface')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Well Depth (m)', fontsize=12)
plt.title('Ross Pond - Well Depth', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% 3.0 Calculate sensor hydrographs based on difference in DEM elevations

out_dfs = []
point_elevations_dfs = []

for i in core_wells:

    stage = osbs_core_wells[osbs_core_wells['well_id'] == i].copy()
    stage = stage[['date', 'well_depth_m', 'well_id']]
    well_z = points_osbs[
        (points_osbs['type'] == 'core_well') & 
        (points_osbs['wetland_id'] == i)
    ]['elevation_']

    well_z = float(well_z.iloc[0])

    for t in types:
        for l in locations:

            location_pt = points_osbs[
                (points_osbs['wetland_id'] == i) &
                (points_osbs['type'] == t) & 
                (points_osbs['location'] == l)]
            
            location_z = location_pt['elevation_']
            
            if len(location_z) == 0:
                print(f"Warning -- no data type {t} at location {l} for wetland {i}")
                continue
            else:
                location_z = float(location_z.iloc[0])

            print(i, l)
            elevation_diff = location_z - well_z

            stage_location = stage.copy()
            stage_location['location'] = l
            stage_location['type'] = t
            stage_location['location_well_diff_m'] = elevation_diff
            stage_location['location_depth_m'] = stage_location['well_depth_m'] - elevation_diff
            
            # Convert the stage location
            stage_location['binary_inundated'] = (stage_location['location_depth_m'] >= 0).astype(int)
            stage_location['binary_10cm_under'] = (stage_location['location_depth_m'] >= -0.10).astype(int)

            def calc_run_col(df: pd.DataFrame, inundated: bool, tgt_col: str, new_col: str):
                """
                Counts the number of consecutive days inundated (tgt_val=1) or dry (tgt_val=0) in 
                the tgt_col and creates a new_col in the df with the count.
                """
                run_count = 0
                run_counts = []
                tgt_val = 1 

                # Switch target val to dry if not inundated
                if not inundated:
                    tgt_val = 0
                
                for val in df[tgt_col]:
                    if val == tgt_val:
                        run_count += 1
                    else:
                        run_count = 0
                    run_counts.append(run_count)
                
                df[new_col] = run_counts
                return df
            
            stage_location = calc_run_col(stage_location, inundated=True, tgt_col='binary_inundated', new_col='consec_inundated_days')
            stage_location = calc_run_col(stage_location, inundated=False, tgt_col='binary_inundated', new_col='consec_dry_days')
            stage_location = calc_run_col(stage_location, inundated=True, tgt_col='binary_10cm_under', new_col='consec_above_10cm_under')
            stage_location = calc_run_col(stage_location, inundated=False, tgt_col='binary_10cm_under', new_col='consec_below_10cm_under')

            pt_summary = pd.DataFrame({
                'location_dem_z': [location_z],
                'well_dem_z': [well_z],
                'location': [l],
                'type': [t],
                'wetland_id': [i],
            })

            out_dfs.append(stage_location)
            point_elevations_dfs.append(pt_summary)

# %% 4.0 Concatonate the sampling specific hydrographs into single dataframes

hydrographs = pd.concat(out_dfs, ignore_index=True)
point_elevations = pd.concat(point_elevations_dfs, ignore_index=True)

# %% 5.0 Compute the hydro summary for each sensor

summary_dfs = []

wetlands = hydrographs['well_id'].unique()
locations = hydrographs['location'].unique()
types = hydrographs['type'].unique()

for i in wetlands:
    for t in types:
        for l in locations:
            ts = hydrographs[(hydrographs['well_id'] == i) &
                             (hydrographs['type'] == t) &
                            (hydrographs['location'] == l)
            ].dropna(subset=['location_depth_m'])
            
            if len(ts) == 0:
                continue
            
            df = ts.copy()

            mean_depth = df['location_depth_m'].mean()
            flooded_days = (df['binary_inundated'] == 1).sum()
            total_days = len(df)
            pti = flooded_days / total_days * 100

            def wet_dry_events(binary, wet_dry):
                """
                Count transitions in binary series.
                - 'wet': transitions from dry (0) to wet (1)
                - 'dry': transitions from wet (1) to dry (0)
                """
                diff = np.diff(np.concatenate([[0], binary]))
                if wet_dry == 'wet':
                    events = np.sum(diff == 1)
                elif wet_dry == 'dry':
                    events = np.sum(diff == -1)
                else:
                    events = -9999
                return events
            
            def inundated_durations(binary):
                """
                Compute statistics on consecutive inundated periods (runs of 1s).
                Returns dict with max, min, median, mean durations (in days).
                """
                runs = []
                current_run = 0
                for val in binary:
                    if val == 1:
                        current_run += 1
                    else:
                        if current_run > 0:
                            runs.append(current_run)
                            current_run = 0
                if current_run > 0:
                    runs.append(current_run)

                if not runs:
                    return {
                        'max_inundated': np.nan,
                        'min_inundated': np.nan,
                        'median_inundated': np.nan,
                        'mean_inundated': np.nan,
                    }
                return {
                    'max_inundated': max(runs),
                    'min_inundated': min(runs),
                    'median_inundated': np.median(runs),
                    'mean_inundated': np.mean(runs),
                }
            
            wet_events = wet_dry_events(df['binary_inundated'].values, 'wet')
            dry_events = wet_dry_events(df['binary_inundated'].values, 'dry')
            durations = inundated_durations(df['binary_inundated'].values)

            result = {
                'well_id': i,
                'type': t,
                'location': l,
                'mean_depth': mean_depth,
                'pti': pti, 
                'max_inundated_duration': durations['max_inundated'],
                'min_inundated_duration': durations['min_inundated'], 
                'median_inundated_duration': durations['median_inundated'],
                'mean_inundated_duration': durations['mean_inundated'],
                'wetup_event_count': wet_events,
                'drydown_event_count': dry_events
            }

            summary_dfs.append(result)

summary = pd.DataFrame(summary_dfs)
    

# %% 6.0 Write the files

# hydrographs_path = './osbs/out_data/misc_stuff/osbs_hydrographs_for_Sunita_Audrey.csv'
# summary_path = './osbs/out_data/misc_stuff/osbs_hydro_summary_for_Sunita_Audrey.csv'

# hydrographs.to_csv(hydrographs_path, index=False)
# summary.to_csv(summary_path, index=False)

# %% 7.0 Quick plot 
import matplotlib.pyplot as plt
import pandas as pd

# Filter for Ross Pond and convert date to datetime
plot_df = hydrographs[hydrographs['well_id'] == 'Ross Pond'].copy()
plot_df = plot_df[plot_df['location'] != 'U']
plot_df['date'] = pd.to_datetime(plot_df['date'])

# Create the timeseries plot
fig, ax = plt.subplots(figsize=(14, 6))

# Define color scheme: deep blue (wetland) to warm brown (upland)
colors = {
    'W1': '#006994',  # Deep ocean blue (wettest)
    'W2': '#2E8B57',  # Sea green (transitional)
    'W3': '#8B6914',  # Dark goldenrod (drier)
    'W4': '#A0522D',  # Sienna (dry)
    'U': '#654321'    # Dark brown (upland/driest)
}

# Plot each location with assigned colors
for location in ['W1', 'W2', 'W3', 'W4', 'U']:
    if location in plot_df['location'].unique():
        location_data = plot_df[plot_df['location'] == location].sort_values('date')
        ax.plot(location_data['date'], location_data['location_depth_m'], 
                label=f'{location}', linewidth=1.5, alpha=0.8, color=colors[location])

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Ground surface')
ax.set_ylabel('Depth at Location (m)', fontsize=12)
ax.set_title('Ross Pond - Water Depth Time Series by Location', fontsize=16, fontweight='bold')
ax.legend(loc='best', ncol=2, fontsize='large')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
