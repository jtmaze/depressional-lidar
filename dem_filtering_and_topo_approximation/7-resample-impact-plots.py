# %% 1.0 Libraries and file paths
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio

site = 'bradford'

os.chdir(f'D:/depressional_lidar/data/{site}')

results_files = glob.glob('./out_data/bradford_region_props_on_depressions*.csv')
print(results_files)

# %% 2.0 Read the data 

def calculate_dAdh(df: pd.DataFrame,
                   step_size_m: float,
                   y_variable: str = 'inundated_area_m2'):
    
    temp = df.copy()
    # Convert step size from meters to kilometers for consistency
    step_size_km = step_size_m / 1000

    forward_area = temp[y_variable].shift(-1)  # f(x+h)
    backward_area = temp[y_variable].shift(1)  # f(x-h) 
    central_difference = (forward_area - backward_area) / (2 * step_size_km)

    # Return the central difference
    return central_difference

US_SURVEY_FOOT_TO_METER = 0.304800609601219

resample_to_cell_size_map_meters = {
    'native': 2.5 * US_SURVEY_FOOT_TO_METER,
    '2': 2.5 * 2 * US_SURVEY_FOOT_TO_METER, 
    '5': 2.5 * 5 * US_SURVEY_FOOT_TO_METER,
    '10': 2.5 * 10 * US_SURVEY_FOOT_TO_METER,
    '12': 2.5 * 12 * US_SURVEY_FOOT_TO_METER,
    '15': 2.5 * 15 * US_SURVEY_FOOT_TO_METER,
    '20': 2.5 * 20 * US_SURVEY_FOOT_TO_METER,
    '25': 2.5 * 25 * US_SURVEY_FOOT_TO_METER,
    '30': 2.5 * 30 * US_SURVEY_FOOT_TO_METER,
    '40': 2.5 * 40 * US_SURVEY_FOOT_TO_METER,
    '50': 2.5 * 50 * US_SURVEY_FOOT_TO_METER,
}

def fetch_DEM_histogram(
    res_factor,
    dem_dir: str = './in_data/resampled_DEMs/'
):
    if res_factor == 'native':
        dem_path = f'{dem_dir}/detrended_dem_all_basins_size1000.tif'
    else:
        dem_path = f'{dem_dir}/detrended_dem_all_basins_resampled{res_factor}_size1000.tif'

    with rio.open(dem_path) as src:
        data = src.read(1, masked=True)
        data = data * US_SURVEY_FOOT_TO_METER
        flat = data.compressed()

        total_pix = len(flat)
        trimmed = flat[(flat >= -3) & (flat <= 3)]
        
        # Create histogram with 0.02 intervals
        bins = np.arange(-3, 3.02, 0.02)  # Add 0.02 to include upper bound
        dem_dist = np.histogram(trimmed, bins=bins)

    return dem_dist, total_pix

rescale_factors = []
results_dfs = []

for i in results_files:
    scale_factor = i.split('_')[-1].replace('.csv', '')
    
    rescale_factors.append(scale_factor)

    hist, total_pix = fetch_DEM_histogram(
        scale_factor,
        f'./in_data/resampled_DEMs/'
    )

    hist_counts, hist_bins = hist
    hist_df = pd.DataFrame({
        'threshold': hist_bins, 
        'dem_pixel_counts': hist_counts
    })
    hist_df['threshold'] = hist_df['threshold'].round(2)

    cell_size_meters = resample_to_cell_size_map_meters.get(scale_factor, 0)

    df = pd.read_csv(i)
    df['scale_factor'] = scale_factor
    df['cell_size_m'] = cell_size_meters
    df['total_pixels'] = total_pix
    # Calculate dA/dh using the function with the appropriate step size
    df['dA/dh_region_props'] = calculate_dAdh(df, step_size_m=0.02, y_variable='inundated_area_m2')

    df_merged = pd.merge(df, hist_df, on='threshold', how='left')
    
    results_dfs.append(df_merged)

# %% 3.0 
results = pd.concat(results_dfs)

# %% 4.0 Make a CDF? to ensure resampling doesn't impact detrended DEM distributions