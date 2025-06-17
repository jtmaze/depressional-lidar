# %% 1.0 Libraries and paths

import os, whitebox
from pathlib import Path
import random

random.seed(42)

proj_root = Path.cwd()
wbt = whitebox.WhiteboxTools()
wbt.verbose = True
# NOTE Conda had issues with wbt installation. Needed to hard-code this. 
wbt.set_whitebox_dir(os.path.join(os.environ['CONDA_PREFIX'], 'bin'))
wbt.set_working_dir(str(proj_root))

# %% 2.0 Set the smoothing parameters designate sites and basins

"""
NOTE: Don't waist too much time fiddling with these. Here's general observations:
-
"""
off_terrain_filter = 12 # Number of cells
off_terrain_slope = 0.25 # Max slope to smooth 
gaussian_sigma = 3 # lager sigma creates more smoothing

site_name = 'bradford'
basin = 'all_basins'

# %% 3.0 Directories for reading and writing files

dem = proj_root / site_name / "out_data" / "basin_clipped_DEMs" / f"dem_mosaic_basin_{basin}.tif"
filled = proj_root / site_name / "temp" / f"dem_mosaic_filled_{basin}.tif"
veg_off = proj_root / site_name / "temp" / f"dem_mosaic_filled_off_terrain_{basin}.tif"
gaussian = proj_root / site_name / "out_data" / "smoothed_dems" / f"dem_smoothed_{basin}.tif"

# %% 4.0 Fill the single cell pits

wbt.fill_single_cell_pits(
    str(dem.resolve()),
    str(filled.resolve())
)

# %% 5.0 Remove off-terrain objects

wbt.remove_off_terrain_objects(
    str(filled.resolve()), 
    str(veg_off.resolve()),
    filter=off_terrain_filter,
    slope=off_terrain_slope
)
# %% 6.0 Gaussian smoothing

wbt.gaussian_filter(
    str(veg_off.resolve()),
    str(gaussian.resolve()),
    sigma=gaussian_sigma
)

# %% 7.0 Clean up workspace

os.remove(filled)
os.remove(veg_off)


# %%
