# %% 1.0 Libraries and file paths

import os
import pandas as pd
import numpy as np
import geopandas as gpd

os.chdir('D:/depressional_lidar/data/')

points_master = gpd.read_file('./rtk_pts_with_dem_elevations.shp')
metric_wells = points_master[points_master['type'].isin(['main_doe_well', 'sunita_transect'])].copy()
metric_wells = metric_wells[metric_wells['site'] == 'Bradford']
metric_wells = metric_wells['wetland_id'].unique()

bradford_core_wells = pd.read_csv('./bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv')
bradford_core_wells = bradford_core_wells[
    bradford_core_wells['wetland_id'].isin(metric_wells)
]

print(bradford_core_wells['flag'].unique())

# %%
import plotly.express as px
plot_df = bradford_core_wells[bradford_core_wells['flag'] == 0].copy()
# # Create interactive timeseries
fig = px.line(
    plot_df.sort_values('date'),
    x='date',
    y='well_depth_m',
    color='wetland_id',
    title='Well Depth Over Time',
    markers=True
)

# Add a horizontal line for the mean of each well
for well in bradford_core_wells['wetland_id'].unique():
    mean_depth = bradford_core_wells[bradford_core_wells['wetland_id'] == well]['well_depth_m'].mean()
    fig.add_hline(y=mean_depth, line_dash="dot",
                  annotation_text=f"Mean {well}", 
                  annotation_position="bottom right")


fig.show()

# %% 2.0 List out bradford wetlands, types and locations

points_bradford = points_master[points_master['site'] == 'Bradford']
wetlands_bradford = points_bradford['wetland_id'].unique()

types = points_bradford['type'].unique()
locations = points_bradford['location'].unique()
print(locations)

# Remove wetland and core_well types, only interested in sampling locations
remove_types = ['main_doe_well', 'aux_wetland_well', 'sunita_well']
types = [t for t in types if t not in remove_types]
print(types)

# %% 3.0 Calculate sensor hydrographs based on difference in DEM elevations

out_dfs = []
point_elevations_dfs = []

for i in metric_wells:

    stage = bradford_core_wells[bradford_core_wells['wetland_id'] == i].copy()
    stage = stage[['date', 'well_depth_m', 'wetland_id', 'flag']]
    well_z = points_bradford[
        (points_bradford['type'].isin(['main_doe_well', 'sunita_well'])) & 
        (points_bradford['wetland_id'] == i)
    ]['z_dem']

    well_z = float(well_z.iloc[0])

    for t in types:
        for l in locations:

            location_pt = points_bradford[
                (points_bradford['wetland_id'] == i) &
                (points_bradford['type'] == t) & 
                (points_bradford['location'] == l)]
            
            location_z = location_pt['z_dem']
            
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

print(hydrographs)

# %% 5.0 Compute the hydro summary for each sensor

summary_dfs = []

wetlands = hydrographs['wetland_id'].unique()
locations = hydrographs['location'].unique()
types = hydrographs['type'].unique()

for i in wetlands:
    for t in types:
        for l in locations:
            ts = hydrographs[(hydrographs['wetland_id'] == i) &
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
                'wetland_id': i,
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

hydrographs_path = './bradford/out_data/misc_stuff/bradford_hydrographs_for_Sunita_Audrey.csv'
summary_path = './bradford/out_data/misc_stuff/bradford_hydro_summary_for_Sunita_Audrey.csv'

hydrographs.to_csv(hydrographs_path, index=False)
summary.to_csv(summary_path, index=False)


# %%
