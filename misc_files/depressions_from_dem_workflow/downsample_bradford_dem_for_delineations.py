# %%

import os
import rasterio as rio
from rasterio.enums import Resampling

os.chdir('D:/depressional_lidar/data/osbs/')

dem_in_path = "./in_data/osbs_DEM_cleaned_veg.tif"
dem_resampled_path = "./in_data/osbs_DEM_resampled_cleaned.tif"

# %%

scale_factor = 3.0

with rio.open(dem_in_path) as src:
    # Calculate new dimensions
    new_width = int(src.width / scale_factor)
    new_height = int(src.height / scale_factor)
    
    # Calculate new transform
    transform = src.transform * src.transform.scale(
        (src.width / new_width),
        (src.height / new_height)
    )
    
    # Read and resample the data
    data = src.read(
        out_shape=(src.count, new_height, new_width),
        resampling=Resampling.bilinear
    )
    
    # Update metadata
    profile = src.profile.copy()
    profile.update({
        'height': new_height,
        'width': new_width,
        'transform': transform
    })
    
    # Write the resampled DEM
    with rio.open(dem_resampled_path, 'w', **profile) as dst:
        dst.write(data)

print(f"Resampled DEM saved to: {dem_resampled_path}")
# %%
