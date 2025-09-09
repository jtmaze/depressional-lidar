# %% 1.0 Libraries and file paths

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon

os.chdir('D:/depressional_lidar/data/')

site = 'bradford'
out_path = f'./{site}/in_data/{site}_basins_assigned_wetland_ids.shp'
depressions_path = f'./{site}/in_data/original_basins/{site}_depression_polygons.shp'
depressions = gpd.read_file(depressions_path)
print(depressions.crs)

well_points_path = f'./rtk_pts_with_dem_elevations.shp'
well_points = gpd.read_file(well_points_path)
well_points = well_points[
    (well_points['type'] == 'wetland_well') |
    (well_points['type'] == 'core_well') &
    (well_points['site'] == 'site')
]
print(well_points.crs)

# %% 2.0 Match wells to basins where well points are contained within the basin polygon
matched_list = []

for idx, pt in well_points.iterrows():
    matched_depressions = depressions.copy()
    pt_geom = pt.geometry
    matched_depressions['msk'] = matched_depressions.geometry.apply(lambda poly: poly.contains(pt_geom))
    matched_depressions = matched_depressions[matched_depressions['msk'] == True]
    # Append to out_gdf if matched
    if not matched_depressions.empty:
        matched_depressions['wetland_id'] = pt['wetland_id']
        matched_depressions.index.name = 'orig_idx'
        matched_depressions['rtk_el'] = pt['rtk_elevat']
        matched_depressions['core_well'] = pt['type'] == 'core_well'
        if len(matched_depressions) > 1:
            print('Warning duplicate depressions for well')
        else:
            matched_list.append(matched_depressions)

out_gdf = gpd.GeoDataFrame(pd.concat(matched_list, ignore_index=True), crs=depressions.crs)
out_gdf.drop(columns=['msk'], inplace=True)      

# %% 3.0 Find wells not contained in depressions, but nearby

unmatched_wells = well_points[~well_points['wetland_id'].isin(out_gdf['wetland_id'])]
print(unmatched_wells['wetland_id'])

matched_list = []

for idx, row in unmatched_wells.iterrows():
    search_area = row.geometry.buffer(200)
    candidate_depressions = depressions[depressions['area_m2'] >= 1000].copy()
    candidate_depressions = candidate_depressions[candidate_depressions.intersects(search_area)]
    candidate_depressions['boundary_dist'] = candidate_depressions.boundary.distance(row.geometry)
    nearest_depression = candidate_depressions.sort_values(by='boundary_dist', ascending=True).iloc[[0]]

    # Quick plot of the original point, nearest, and candidate depressions
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot the original point
    gpd.GeoSeries([row.geometry]).plot(ax=ax, color='red', markersize=100, label='Original Point')
    # Plot candidate depressions
    candidate_depressions.plot(ax=ax, color='blue', alpha=0.5, label='Candidate Depressions')
    # Plot the nearest depression in orange
    if not nearest_depression.empty:
        nearest_depression.plot(ax=ax, color='orange', alpha=0.7, label='Nearest Depression')
    ax.set_title(f'Well {row["wetland_id"]}: Point and Candidate Depressions')
    ax.legend()
    plt.show()

    if not nearest_depression.empty:
        nearest_depression.drop(columns=['boundary_dist'], inplace=True)
        nearest_depression['wetland_id'] = row['wetland_id']
        nearest_depression.index.name = 'orig_idx'
        nearest_depression['rtk_el'] = row['rtk_elevat']
        nearest_depression['core_well'] = pt['type'] == 'core_well'
        matched_list.append(nearest_depression)

# Add the other matched depressions to the out_gdf
concatonated = pd.concat(matched_list, ignore_index=True)
out_gdf = gpd.GeoDataFrame(pd.concat([concatonated, out_gdf], ignore_index=True), crs=depressions.crs)

# %% 4.0 Write the output

out_gdf.to_file(out_path)

# %%
