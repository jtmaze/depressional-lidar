# %% 1.0 Libraries and file paths

# NOTE: processing Ordway and Bradford wells with differently until the Bradford wells are
# resurveyed, we are not confident in those RTK measurements. 

import os
import datetime as dt
import pandas as pd
import numpy as np
import geopandas as gpd

os.chdir('D:/depressional_lidar/data/')

pt_master = gpd.read_file('./rtk_pts_with_dem_elevations.shp')
osbs_core_wells = pd.read_csv('./osbs/in_data/stage_data/osbs_core_wells_tracked_datum.csv')

# %% 2.0 List out OSBS wetlands, types and locations

pt_osbs = pt_master[pt_master['site'] == 'osbs']
wetlands_osbs = pt_osbs['wetland_id'].unique()
print(wetlands_osbs)
types = pt_osbs['type'].unique()
locations = pt_osbs['location'].unique()
print(types)

# Convert OSBS well timeseries to daily
osbs_core_wells['date'] = pd.to_datetime(osbs_core_wells['date'])
osbs_core_wells['calendar_date'] = osbs_core_wells['date'].dt.date
osbs_core_wells_daily = osbs_core_wells.groupby(by=['well_id', 'calendar_date']).mean().reset_index()
osbs_core_wells_daily.drop(columns=['date'], inplace=True)

# %% 3.0 Calculate sensor hydrographs based on difference in RTK elevations
out_dfs = []

for i in wetlands_osbs:

    stage = osbs_core_wells_daily[osbs_core_wells_daily['well_id'] == i].copy()
    stage = stage[['calendar_date', 'water_level', 'well_id']]
    well_z = pt_osbs[
        (pt_osbs['type'] == 'core_well') & 
        (pt_osbs['wetland_id'] == i)
    ]['rtk_elevat']

    well_z = float(well_z.iloc[0])

    for l in locations:

        location_pt = pt_osbs[
            (pt_osbs['wetland_id'] == i) &
            (pt_osbs['type'] == 'sensor_transect') & 
            (pt_osbs['location'] == l)]
        
        location_z = location_pt['rtk_elevat']
        
        if len(location_z) == 0:
            print(f"Warning -- no sensor location {l} at wetland {i}")
            continue
        else:
            location_z = float(location_z.iloc[0])

        print(i, l)
        elevation_diff = location_z - well_z
        vertical_error = location_pt['rtk_vert_a'].iloc[0]

        # Handle missing values for vertical error and convert mm to meters. 
        if vertical_error == 'Missing':
            vertical_error_m = 0.5
        else:
            vertical_error_m = float(vertical_error) / 1_000
        print(elevation_diff, vertical_error_m)

        stage_location = stage.copy()
        stage_location['location'] = l
        stage_location['rtk_vertical_accuracy'] = vertical_error_m
        stage_location['location_well_diff'] = elevation_diff
        stage_location['location_depth'] = stage_location['water_level'] - elevation_diff
        stage_location['location_depth_lower'] = stage_location['water_level'] - elevation_diff - vertical_error_m
        stage_location['location_depth_upper'] = stage_location['water_level'] - elevation_diff + vertical_error_m

        out_dfs.append(stage_location)


# %% 4.0 Compute the hydro summary for each sensor

hydrographs = pd.concat(out_dfs)

def calc_summary_stats(df: pd.DataFrame):

    summary_dfs = []

    wetlands = hydrographs['well_id'].unique()
    locations = hydrographs['location'].unique()

    for i in wetlands:
        for l in locations:
            ts = hydrographs[(hydrographs['well_id'] == i) &
                             (hydrographs['location'] == l)
            ].dropna(subset=['location_depth'])
            
            def calc_ts_summary(df: pd.DataFrame, depth_name: str):
                
                df = df.copy()
                df['binary'] = (df[depth_name] >= 0).astype(int)

                mean_depth = df[depth_name].mean()
                flooded_days = (df['binary'] == 1).sum()
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
                            'max_inundated': 0,
                            'min_inundated': 0,
                            'median_inundated': 0,
                            'mean_inundated': 0
                        }
                    return {
                        'max_inundated': max(runs),
                        'min_inundated': min(runs),
                        'median_inundated': np.median(runs),
                        'mean_inundated': np.mean(runs)
                    }
                
                wet_events = wet_dry_events(df['binary'].values, 'wet')
                dry_events = wet_dry_events(df['binary'].values, 'dry')
                durations = inundated_durations(df['binary'].values)

                summary = {
                    'uncert': depth_name,
                    'mean_depth': mean_depth,
                    'pti': pti, 
                    'max_inundated_duration': durations['max_inundated'],
                    'min_inundated_duration': durations['min_inundated'], 
                    'median_inundated_duration': durations['median_inundated'],
                    'mean_inundated_duration': durations['mean_inundated'],
                    'wetup_event_count': wet_events,
                    'drydown_event_count': dry_events
                }

                return summary
            
            low = calc_ts_summary(ts, depth_name='location_depth_lower')
            med = calc_ts_summary(ts, depth_name='location_depth')
            high = calc_ts_summary(ts, depth_name='location_depth_upper')

            result = pd.DataFrame([low, med, high])
            result['well_id'] = i
            result['location'] = l

            summary_dfs.append(result)

    return pd.concat(summary_dfs, ignore_index=True)
        
# %% 4.1 Run the function

summary = calc_summary_stats(hydrographs)

# %% 5.0 Write the files

summary.to_csv('./misc/osbs_sensor_metrics_for_AudreyG.csv', index=False)
hydrographs.to_csv('./misc/osbs_sensor_hydrographs_for_AudreyG.csv', index=False)

# %% 6.0 Quick plot to see how uncertianty translates to metrics. 

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 12))

# Create subplot grid
g = sns.FacetGrid(summary, col='well_id', col_wrap=3, height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x='location', y='wetup_event_count', hue='uncert', s=100, alpha=0.7)
g.set_axis_labels('Location', 'Wetup Event Count')
g.set_titles('{col_name}')


g.fig.legend(g._legend_data.values(), g._legend_data.keys(), 
             loc='lower center', bbox_to_anchor=(0.5, -0.05), 
             ncol=3, title='Uncertainty Level')

for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

    
# %%
