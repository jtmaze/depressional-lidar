# %% 1.0 Libraries and file paths

# NOTE: processing Ordway and Bradford wells with differently until the Bradford wells are
# resurveyed, we are not confident in those RTK measurements. 

import os
import datetime as dt
import pandas as pd
import numpy as np
import geopandas as gpd

os.chdir('D:/depressional_lidar/data/')

points_master = gpd.read_file('./rtk_pts_with_dem_elevations.shp')

core_wells = ['13_267', '14_500', '15_409', '6_93', '14_612', '5a_582']

bradford_core_wells = pd.read_csv('./bradford/in_data/stage_data/bradford_daily_stage_Winter2025.csv')
bradford_core_wells = bradford_core_wells[
    bradford_core_wells['well_id'].isin(core_wells)
]

print(bradford_core_wells['flag'].unique())

# %% 2.0 List out bradford wetlands, types and locations

points_bradford = points_master[points_master['site'] == 'bradford']
wetlands_bradford = points_bradford['wetland_id'].unique()
print(wetlands_bradford)
types = points_bradford['type'].unique()
locations = points_bradford['location'].unique()
print(types)

# Remove wetland and core_well types, only interested in sampling locations
remove_types = ['core_well', 'wetland_well']
types = [t for t in types if t not in remove_types]
print(types)

# %% 3.0 Calculate sensor hydrographs based on difference in DEM elevations

out_dfs = []
point_elevations_dfs = []

for i in core_wells:

    stage = bradford_core_wells[bradford_core_wells['well_id'] == i].copy()
    stage = stage[['date', 'well_depth_m', 'well_id']]
    well_z = points_bradford[
        (points_bradford['type'] == 'core_well') & 
        (points_bradford['wetland_id'] == i)
    ]['elevation_']

    well_z = float(well_z.iloc[0])

    for t in types:
        for l in locations:

            location_pt = points_bradford[
                (points_bradford['wetland_id'] == i) &
                (points_bradford['type'] == t) & 
                (points_bradford['location'] == l)]
            
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

            def calc_run_col(df: pd.DataFrame, tgt_col: str, new_col: str):
                """
                Counts the number of consecutive days inundated (=1) in the tgt_col and 
                creates a new_col in the df with the count
                """
                run_count = 0
                run_counts = []
                
                for val in df[tgt_col]:
                    if val == 1:
                        run_count += 1
                    else:
                        run_count = 0
                    run_counts.append(run_count)
                
                df[new_col] = run_counts
                return df
            
            stage_location = calc_run_col(stage_location, 'binary_inundated', 'consec_inundated_days')
            stage_location = calc_run_col(stage_location, 'binary_10cm_under', 'consec_10cm_under')

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

summary_sensors = summary[summary['type'] == 'soil_moisture_sensor']
hydrographs_sensors = hydrographs[hydrographs['type'] == 'soil_moisture_sensor']
summary_sensors.to_csv('./misc/bradford_sensor_metrics_for_AudreyG.csv', index=False)
hydrographs_sensors.to_csv('./misc/bradford_sensor_hydrographs_for_AudreyG.csv', index=False)

summary_litter = summary[summary['type'] == 'litter_trap']
hydrographs_litter = hydrographs[hydrographs['type'] == 'litter_trap']
summary_litter.to_csv('./misc/bradford_litter_metrics_for_SunitaS.csv', index=False)
hydrographs_litter.to_csv('./misc/bradford_litter_stage_for_SunitaS.csv', index=False)

summary_flux = summary[summary['type'] == 'ghg_flux']
hydrographs_flux = hydrographs[hydrographs['type'] == 'ghg_flux']
summary_flux.to_csv('./misc/bradford_flux_metrics_for_SunitaS.csv', index=False)
hydrographs_flux.to_csv('./misc/bradford_flux_stage_for_SunitaS.csv', index=False)

# %% 6.0 Quick plot to see how uncertianty translates to metrics. 

import seaborn as sns
import matplotlib.pyplot as plt

plot = summary[summary['type'] == 'litter_trap'].copy()

# Convert location to categorical with desired order
location_order = ['W4', 'W3', 'W2', 'W1', 'U']
plot['location'] = pd.Categorical(plot['location'], categories=location_order, ordered=True)

plt.figure(figsize=(20, 12))

# Create subplot grid
g = sns.FacetGrid(plot, col='well_id', col_wrap=3, height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x='location', y='mean_inundated_duration', hue='uncert', s=100, alpha=0.7)
g.set_axis_labels('Location', '# continous days')
g.set_titles('{col_name}')

g.fig.legend(g._legend_data.values(), g._legend_data.keys(), 
             loc='lower center', bbox_to_anchor=(0.5, -0.05), 
             ncol=3, title='Uncertainty Level')

for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()
# %%
