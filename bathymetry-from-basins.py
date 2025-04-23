# %% 1.0 Read the depression shapes

import pandas as pd
import numpy as np 
import geopandas as gpd
from shapely.geometry import mapping
import rasterio as rio
from rasterio.mask import mask
import matplotlib.pyplot as plt

target_basin_ids = {
    7: '15_409',
    2: '6_93',
    51: '13_267',
    36: '14_500',
    45: '5a_582',
    23: '6_629'
}

well_basins_path = './out_data/well_delineations.shp'
mosaic_dem_path = './out_data/dem_mosaic.tif'

# 2.0 Select basins of interest reproject the shape file into the DEM's crs
with rio.open(mosaic_dem_path) as src:
    dem_crs = src.crs

basins = gpd.read_file(well_basins_path)
basins_crs = basins.crs

target_basins = basins[basins['OBJECTID'].isin(target_basin_ids.keys())].copy()
# Create buffered geometry (store original first)
# target_basins['geometry_original'] = target_basins.geometry.copy()
# buffered_geoms = target_basins.geometry.buffer(-5)

# # Set the buffered geometry as the active geometry
# target_basins = target_basins.set_geometry(buffered_geoms)
target_basins = target_basins.to_crs(dem_crs)
target_basins['well_id'] = target_basins['OBJECTID'].map(target_basin_ids)
print(target_basins.head(10))
target_basins = target_basins[['OBJECTID', 'BUFF_DIST', 'geometry', 'well_id']]

# %% 3.0 Clip the DEM to the basin extent

out_data = []

for idx, row in target_basins.iterrows():
    # NOTE: Raster cells are 2.5x2.5 ft = 6.25 sqft
    # NOTE: 1 sqft = 0.092903 m2

    row_geom = [mapping(row.geometry)]
    well_id = row.well_id

    with rio.open(mosaic_dem_path) as src:

        out_img, out_trans = mask(src, row_geom, crop=True)
        out_meta = src.meta.copy()
        nodata = src.nodata
        out_meta.update({
            'height': out_img.shape[1],
            'width': out_img.shape[2],
            'transform': out_trans
        })

        out_path = f'./out_data/basin_clipped_DEMs/{well_id}_basin_dem.tif'
        # with rio.open(out_path, 'w', **out_meta) as dst:
        #     dst.write(out_img)

        elevations = out_img[0].flatten()

        elevations_clean = elevations[elevations != nodata]
        total_area = len(elevations_clean) * 6.25 * 0.092903 # convert to m2

        print(f'Total basin area = {total_area:.2f} sqm')

        def calc_bathymetric_curve(data, bin_width=0.05, total_area=total_area):
            
            # NOTE: bin_width is in feet, covert to meters later

            min_elev = np.floor(data.min())
            min_elev = min_elev + (0.3 * data.std())
            max_elev = np.ceil(data.max())
            print(min_elev, max_elev)
            bins = np.arange(min_elev, max_elev + bin_width, bin_width)

            hist, bin_edges = np.histogram(data, bins=bins)
            cum_counts = np.cumsum(hist) # cumulative number of pixels
            area = cum_counts * 6.25 * 0.092003 / total_area # raster grid cells are 2.5 sqm
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


            return area, bin_centers
        
        def calc_stage_volume_curve(data, bin_width=0.05):

            min_elev = np.floor(data.min())
            min_elev = min_elev + (0.3 * data.std())
            max_elev = np.ceil(data.max())
            bins = np.arange(min_elev, max_elev + bin_width, bin_width)

            # Calculate histogram
            hist, bin_edges = np.histogram(data, bins=bins)
            bin_width_m = bin_width * 0.3048 # Convert bin width to meters. 
            
            # Calculate volume of each bin (i.e, elevation range)
            cell_area = 6.25 * 0.092903 # area of each raster cell in m2
            bin_volumes = hist * cell_area * bin_width_m # volume of each bin in m3

            # Calculate cumulative volumes
            cum_volumes = np.cumsum(bin_volumes) # cumulative volume of each bin
            total_basin_volume = cum_volumes[-1] if len(cum_volumes) > 0 else 0

            # Replace values equal to total basin volume with np.nan
            cum_volumes[cum_volumes + (total_basin_volume * 0.02) >= total_basin_volume] = np.nan


            print(f'Total basin volume = {total_basin_volume:.2f} m3')

            # Calculate bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            return cum_volumes, bin_centers

        area, stage = calc_bathymetric_curve(elevations_clean)
        volume, stage_bins_v = calc_stage_volume_curve(elevations_clean)

        out_data.append({
            'well_id': well_id,
            'area': area,
            'stage': stage, 
            'stage_from_lowest': stage - stage.min(),
            'volume': volume,
        })

# %%

combined_df = pd.DataFrame(out_data)

expanded_rows = []
for idx, row in combined_df.iterrows():
    well_id = row['well_id']
    area = row['area']
    stage = row['stage']
    stage_from_lowest = row['stage_from_lowest']
    volume = row['volume']

    # Create a DataFrame for the current row
    for a, s, s_lowest, v in zip(area, stage, stage_from_lowest, volume):
        expanded_rows.append({
            'well_id': well_id,
            'area': a,
            'stage': s,
            'stage_from_lowest': s_lowest,
            'volume': v
        })

expanded_df = pd.DataFrame(expanded_rows)
expanded_df.to_csv('./out_data/bathymetric_curves.csv', index=False)

# %% Plot the bathymetric curves together

plt.figure(figsize=(10, 6))

for idx, row in combined_df.iterrows():
    well_id = row['well_id']
    area = row['area'] * 100 # convert to percentage
    stage = row['stage_from_lowest'] * 0.3048 # convert to meters

    plt.plot(area, stage, label=well_id)

plt.xlabel('Inundated Area (%) of Total Basin Area')
plt.ylabel('Stage (meters from basin lowest elevation)')
plt.ylim(0, 1.75)
plt.title('Bathymetric Curves for Intensive Basins')
plt.legend()
plt.tight_layout()
plt.show()

# %% Plot the stage-volume curves together

plt.figure(figsize=(10, 6))

for idx, row in combined_df.iterrows():
    well_id = row['well_id']
    volume = row['volume'] * 100
    stage = row['stage_from_lowest'] * 0.3048 # convert to meters

    plt.plot(volume, stage, label=well_id)

plt.xlabel('Volume (m3) - log scale')
plt.ylabel('Stage (meters from basin lowest elevation)')
plt.ylim(0, 1.55)
plt.xscale('log')
plt.title('Stage-Volume Curves for Intensive Basins')
plt.legend()

plt.tight_layout()
plt.show()

# %%

