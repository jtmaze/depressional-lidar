# %% 1.0
import os, whitebox
import rasterio as rio
import geopandas as gpd
from pathlib import Path
import random

random.seed(42)

basin = "all_basins"
gaussian_sigma = 6
off_ter_filter = 8
off_ter_slope = 0.15
error_range = 40
rmse = 0.25

proj_root = Path.cwd()                 

dem    = proj_root / "out_data" / f"dem_mosaic_basin_{basin}.tif"
filled = proj_root / "temp"     / f"dem_mosaic_filled_{basin}.tif"
off_terrain = proj_root / "temp" / f"dem_mosaic_filled_off_terrain_{basin}_filter{off_ter_filter}_slope{off_ter_slope}.tif"
gaussian = proj_root / "temp" / f"dem_mosaic_gaussian_{basin}_sigma{gaussian_sigma}.tif"
depresions = proj_root / "temp" / f"depression_probabilities_{basin}_sigma{gaussian_sigma}_range{error_range}_rmse{rmse}.tif"

wbt = whitebox.WhiteboxTools()
wbt.verbose = True
wbt.set_whitebox_dir(os.path.join(os.environ["CONDA_PREFIX"], "bin"))
wbt.set_working_dir(str(proj_root))   

# %% Run the fill pits algorithm

ret = wbt.fill_single_cell_pits(
    str(dem.resolve()),
    str(filled.resolve())
)

# %% Remove off-terrain objects (i.e., trees missed by original LiDAR processing)

ret = wbt.remove_off_terrain_objects(
    str(filled.resolve()),
    str(off_terrain.resolve()),
    filter=off_ter_filter, # Filter size X*Y
    slope=off_ter_slope, # Slope theshold in degrees
)

# %% Apply a Gaussian filter to the DEM

wbt.gaussian_filter(
    str(off_terrain.resolve()),
    str(gaussian.resolve()),
    sigma=gaussian_sigma
)

# %% Run the stochastic depresion algorithm

wbt.stochastic_depression_analysis(
    str(gaussian.resolve()),
    str(depresions.resolve()),
    rmse=rmse, # Not sure if this should be 0.1 or 0.2?
    range=error_range,#Error field's correlation range xy-units #TODO: Figure out what happens as I change this
    iterations=50
)
# %%

with rio.open(depresions) as src:
    print(src.meta)
    depressions = src.read(1)


