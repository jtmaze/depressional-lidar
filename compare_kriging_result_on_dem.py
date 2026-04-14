# %% 1.0 Libraries
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

kriging_result_path = 'D:/depressional_lidar/data/bradford/out_data/well_wse_interpolations/interpolated_median_WSE_optimized_model.tif'
dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_USGS.tif'
out_path = 'D:/depressional_lidar/data/bradford/out_data/kriging_inundation_optimized.tif'

# %% 2.0 Read rasters and reproject kriging to DEM grid
with rasterio.open(dem_path) as dem_src:
    dem = dem_src.read(1)
    dem_profile = dem_src.profile.copy()

    # Reproject kriging WSE (band 1) and uncertainty (band 2) to DEM grid
    wse = np.empty_like(dem, dtype=np.float32)
    uncertainty = np.empty_like(dem, dtype=np.float32)

    with rasterio.open(kriging_result_path) as krig_src:
        reproject(
            source=krig_src.read(1),
            destination=wse,
            src_transform=krig_src.transform,
            src_crs=krig_src.crs,
            dst_transform=dem_src.transform,
            dst_crs=dem_src.crs,
            resampling=Resampling.bilinear,
            src_nodata=krig_src.nodata,
            dst_nodata=np.nan,
        )
        reproject(
            source=krig_src.read(2),
            destination=uncertainty,
            src_transform=krig_src.transform,
            src_crs=krig_src.crs,
            dst_transform=dem_src.transform,
            dst_crs=dem_src.crs,
            resampling=Resampling.bilinear,
            src_nodata=krig_src.nodata,
            dst_nodata=np.nan,
        )

wse[uncertainty > 1.25] = np.nan

# %% 3.0 Compute flooded zones and write output
flooded = (wse > dem).astype(np.uint8)
flooded[np.isnan(wse) | np.isnan(dem)] = 255  # nodata

out_profile = dem_profile.copy()
out_profile.update(dtype='uint8', count=1, nodata=255)

with rasterio.open(out_path, 'w', **out_profile) as dst:
    dst.write(flooded, 1)



# %%
