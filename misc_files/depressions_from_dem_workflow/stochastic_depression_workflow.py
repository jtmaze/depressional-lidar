# %% 1.0 Libraries and filepaths

import os
import rasterio as rio
import whitebox_workflows as wbw
wbe = wbw.WbEnvironment()
wbe.verbose = True
site = 'bradford'
print(wbe)

os.chdir(f'D:/depressional_lidar/data/{site}')
vegetation_off_dem_path = f"./in_data/{site}_DEM_cleaned_veg.tif"

rmse = 0.1
error_range = 50

input_raster = wbe.read_raster(vegetation_off_dem_path)
print('read the raste')
prob_depression = wbe.stochastic_depression_analysis(
    input_raster,
    rmse=rmse,
    range=error_range,
    iterations=50,
)
print('done assigning probabilities')
output_path = f"./out_data/{site}_prob_depressions_{rmse}_{error_range}.tif"
wbe.write_raster(prob_depression, output_path)

# %%
