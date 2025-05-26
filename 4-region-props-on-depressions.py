# %% 1.0 Libraries and file paths

import rasterio as rio
import pandas as pd
import skimage as ski

from skimage.measure import regionprops

basin = '3'
dem_path = f'./temp/dem_smoothed_{basin}.tif'



# %% 2.0 