# %% 1.0 Libraries and filepaths

import os
import rasterio as rio
import whitebox_workflows as wbw
wbe = wbw.WbEnvironment()

wbe.verbose = True
site = 'bradford'
print(wbe)

os.chdir(f'D:/depressional_lidar/data/{site}')
vegetation_off_dem_path = f"./in_data/{site}_dem_smoothed_5.tif"

rmse = 0.25
error_range = 50
iterations = 25

# %% 2.0 Run the tool

input_raster = wbe.read_raster(vegetation_off_dem_path)
print('read the raster')
prob_depression = wbe.stochastic_depression_analysis(
    input_raster,
    rmse=rmse,
    range=error_range,
    iterations=iterations
)
print('done assigning probabilities')
output_path = f"./out_data/{site}_prob_depressions_SMOOTH_TEST_{rmse}_{error_range}_{iterations}.tif"
wbe.write_raster(prob_depression, output_path)

# %%
