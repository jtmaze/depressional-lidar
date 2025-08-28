# %% 1.0 Import libraries, organize paths, set parameters
# 
import os
import rasterio as rio
import whitebox_workflows as wbw
wbe = wbw.WbEnvironment()
wbe.verbose = True
site = 'bradford'
print(wbe)

os.chdir(f'D:/depressional_lidar/data/{site}')
vegetation_off_dem_path = f"./in_data/{site}_DEM_cleaned_veg.tif"
filled_path = f'./temp/{site}_depressions_filled.tif'

# %% 
input_raster = wbe.read_raster(vegetation_off_dem_path)
filled_raster = wbe.fill_depressions(input_raster, max_depth=3.0)
wbe.write_raster(filled_raster, filled_path)

 #%% Subtract the filled DEM from the original DEM to get the depth of depressions
depression_depth_raster = filled_raster - input_raster
depression_depth_path = f'./temp/{site}_depression_depth.tif'
wbe.write_raster(depression_depth_raster, depression_depth_path)


# %%