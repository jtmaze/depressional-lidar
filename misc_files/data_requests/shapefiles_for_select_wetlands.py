# %% 1.0 File paths and libraries

import geopandas as gpd

basins_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'
well_pts_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'

basins = gpd.read_file(basins_path)
basins = basins[['wetland_id', 'Shape_Area', 'geometry']]
well_pts = (
        gpd.read_file(well_pts_path)[['wetland_id', 'type', 'rtk_elevat', 'geometry']]
        .rename(columns={'rtk_elevat': 'rtk_elevation'})
        .query("type in ['core_well', 'wetland_well']")
    )
well_pts = well_pts[['wetland_id', 'geometry']]

# %% 2.0 Select the WY analysis wetlands

wy_log = [
    '15_268', '14_418', '14_500', '14_610', '5_597',
    '5_510', '9_439', '9_508', '13_271', '7_243',
    '3_311', '3_173', '3_244'
]

wy_ref = [
    '15_409', '14_538', '14_616', '5a_550', '5_573', 
    '5_546', '9_332', '13_410', '13_274', '6a_17',
    '6_93', '6_300', '3_34', '3_368'
]
wy_list = wy_log + wy_ref

wy_points = well_pts[well_pts['wetland_id'].isin(wy_list)]
print(len(wy_points))
wy_basins = basins[basins['wetland_id'].isin(wy_list)]
print(len(wy_basins))

# %% 2.1 Write WY wetlands to a file

pts_out_path = 'D:/depressional_lidar/data/bradford/bradford_wtr_yield_well_points.shp'
basins_out_path = 'D:/depressional_lidar/data/bradford/bradford_wtr_yield_wetland_basins.shp'

wy_points.to_file(pts_out_path, index=False)
wy_basins.to_file(basins_out_path, index=False)

# %% 3.0 Select the DOE analysis wetlands

doe_list = ['13_267', '5a_582', '15_409', '6_93', '14_500', '14_612']

doe_points = well_pts[well_pts['wetland_id'].isin(doe_list)]
doe_basins = basins[basins['wetland_id'].isin(doe_list)]

# %% 3.1 Write DOE wetlands to a file. 

pts_out_path = 'D:/depressional_lidar/data/bradford/bradford_doe_well_points.shp'
basins_out_path = 'D:/depressional_lidar/data/bradford/bradford_doe_wetland_basins.shp'

doe_points.to_file(pts_out_path, index=False)
print(len(doe_points))
doe_basins.to_file(basins_out_path, index=False)
print(len(doe_basins))
# %% 4.0 File with unused wetlands

omit_wetlands = [
    "14_15", "14.9_527", "14.9_168", "14.9_601",
    "5_560", "5_321", "9_77", "7_622", "3_21", "3_23",
    "9_609", "6a_530", "6_629", "7_341"
]

unused_basins = basins[basins['wetland_id'].isin(omit_wetlands)]
unused_points = well_pts[well_pts['wetland_id'].isin(omit_wetlands)]

pts_out_path = 'D:/depressional_lidar/data/bradford/bradford_unused_well_points.shp'
basins_out_path = 'D:/depressional_lidar/data/bradford/bradford_unused_wetland_basins.shp'

unused_points.to_file(pts_out_path, index=False)
print(len(unused_points))
unused_basins.to_file(basins_out_path, index=False)
print(len(unused_basins))


# %%
