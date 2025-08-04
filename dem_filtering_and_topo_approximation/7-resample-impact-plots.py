# %% 1.0 Libraries and file paths
import os
import glob
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import rasterio as rio

import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

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
    step_size = step_size_m 

    forward_area = temp[y_variable].shift(-1) # f(x+h)
    backward_area = temp[y_variable].shift(1)  # f(x-h) 
    central_difference = (forward_area - backward_area) / (2 * step_size)

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
    hist_bins = hist_bins[:-1]

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

    df['threshold'] = df['threshold'].round(2)

    df_merged = pd.merge(df, hist_df, on='threshold', how='left')
    df_merged['dem_hist_area_m2'] = df_merged['dem_pixel_counts'].cumsum() * cell_size_meters**2
    df_merged['dA/dh_dem_hist'] = calculate_dAdh(df_merged, 
                                                 step_size_m=0.02, 
                                                 y_variable='dem_hist_area_m2')

    results_dfs.append(df_merged)

# %% 3.0 
results = pd.concat(results_dfs)

# %% 4.1 Plot dA_dh versus the relative gw depth colored by cell size

plt.figure(figsize=(10, 6))

# Get unique cell sizes and create a color mapping
unique_cell_sizes = sorted(results['cell_size_m'].unique())
norm = Normalize(vmin=min(unique_cell_sizes), vmax=max(unique_cell_sizes))
cmap = cm.viridis

for cell_size in unique_cell_sizes:

    ds = results[results['cell_size_m'] == cell_size]
    
    ds = ds.sort_values('threshold')

    plt.plot(ds['threshold'], ds['dA/dh_dem_hist'], 
             #label=f'Cell size: {round(cell_size, 2)} m', 
             linewidth=2, color=cmap(norm(cell_size)))

plt.xlabel('z$_{gw}$ (meters - relative to detrended DEM)', fontsize=12)
plt.ylabel('dA/dh (m²/m)', fontsize=12)
plt.title('DEM Inudated Area Derrivative', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)


sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())  # Added ax=plt.gca()
cbar.set_label('Cell Size (m)')

plt.legend()
plt.tight_layout()
plt.show()

# %% 4.2 Plot the summed perimeter as a function of gw depth colored by cell size

plt.figure(figsize=(10, 6))

for cell_size in unique_cell_sizes:

    ds = results[results['cell_size_m'] == cell_size]

    ds = ds.sort_values('threshold')

    plt.plot(ds['threshold'], ds['total_perimeter_m'], 
             #label=f'Cell size: {round(cell_size, 2)} m', 
             linewidth=2, color=cmap(norm(cell_size)))

plt.xlabel('z$_{gw}$ (meters - relative to detrended DEM)', fontsize=12)
plt.ylabel('Total Perimeter (m)', fontsize=12)
plt.title('Summed Raster Perimeters', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)


sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Cell Size (m)')
plt.legend()
plt.tight_layout()
plt.show()

# %% See if max summed perimeter's threshold is impacted by DEM resolution

max_perimeter_numbers = results.groupby('scale_factor').apply(
    lambda g: pd.Series({
        'cell_size_m': g['cell_size_m'].iloc[0],
        'max_perimeter': g['total_perimeter_m'].max(),
        'threshold': g.loc[g['total_perimeter_m'].idxmax(), 'threshold']
    })
).reset_index()

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=max_perimeter_numbers,
    x='max_perimeter',
    y='threshold',
    hue='cell_size_m',
    palette='viridis',
    s=100
)

plt.title('Max Perimeter vs. z$_{gw}$ threshold')
plt.xlabel('Max Perimeter (m)')
plt.ylabel('z$_{gw}$ threshold')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Q-Q plot comparing summed perimeter values across cell sizes to a 
# Calculate how many cell sizes we have and determine a good grid size
fig, ax = plt.subplots(figsize=(12, 10))


# Create Q-Q plots for each cell size
for cell_size in unique_cell_sizes:
    
    cell_data = results[results['cell_size_m'] == cell_size]
    
    clean_data = cell_data.dropna(subset=['total_perimeter_m', 'dA/dh_dem_hist'])
        
    # Get perimeter and dA/dh data
    perimeter_data = clean_data['total_perimeter_m'].values
    dadh_data = clean_data['dA/dh_dem_hist'].values
    
    # Sort both datasets
    perimeter_sorted = np.sort(perimeter_data)
    dadh_sorted = np.sort(dadh_data)
    
    # If datasets have different lengths, interpolate to match
    if len(perimeter_sorted) != len(dadh_sorted):
        # Create evenly spaced quantiles
        quantiles = np.linspace(0, 1, min(len(perimeter_sorted), len(dadh_sorted)))
        
        
        perimeter_quantiles = np.quantile(perimeter_sorted, quantiles)
        dadh_quantiles = np.quantile(dadh_sorted, quantiles)
        print("Interpolated some stuff")

    else:
        perimeter_quantiles = perimeter_sorted
        dadh_quantiles = dadh_sorted
    
    # Plot the Q-Q points with color based on cell size
    plt.scatter(perimeter_quantiles, dadh_quantiles, 
                label=f'Cell size: {round(cell_size, 2)} m', 
                color=cmap(norm(cell_size)),
                alpha=0.7, s=40)

# Add a reference line for visual guidance (45-degree line)
# Determine appropriate limits for the reference line
min_perimeter = results['total_perimeter_m'].min()
max_perimeter = results['total_perimeter_m'].max()
min_dadh = results['dA/dh_dem_hist'].min()
max_dadh = results['dA/dh_dem_hist'].max()

# Scale the reference line to match the data ranges
# This is a visual guide, not a strict 1:1 line since the units are different
scale_factor = (max_dadh - min_dadh) / (max_perimeter - min_perimeter)
midpoint_perimeter = (min_perimeter + max_perimeter) / 2
midpoint_dadh = (min_dadh + max_dadh) / 2

# Create reference line based on scaled ranges
x_ref = np.linspace(min_perimeter, max_perimeter, 100)
y_ref = (x_ref - midpoint_perimeter) * scale_factor + midpoint_dadh

plt.plot(x_ref, y_ref, 'r--', label='Reference Line')

plt.xlabel('Total Perimeter (m)', fontsize=14)
plt.ylabel('dA/dh (m²/m)', fontsize=14)
plt.title('Q-Q Plot: Total Perimeter vs. dA/dz Distribution by Cell Size', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)

# Add color bar for cell size
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Cell Size (m)', fontsize=14)

# Add a legend (optional, might be crowded with many cell sizes)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()
# %% 4.3 Check that summed pond area curves vs zgw are invariant with DEM resolution

plt.figure(figsize=(10, 6))

for cell_size in unique_cell_sizes:

    ds = results[results['cell_size_m'] == cell_size]

    ds = ds.sort_values('threshold')

    plt.plot(ds['threshold'], ds['inundated_area_m2'], 
             #label=f'Cell size: {round(cell_size, 2)} m', 
             linewidth=2, color=cmap(norm(cell_size)))

plt.yscale('log')
plt.xlabel('z$_{gw}$ (meters - relative to detrended DEM)', fontsize=12)
plt.ylabel('Inundate Area (sqm - log scale)', fontsize=12)
plt.title('Summed Area vs z$_{gw}$ (m)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)


sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Cell Size (m)')
plt.legend()
plt.tight_layout()
plt.show()


# %% Plot the distinct pond distributions colored by cell size

plt.figure(figsize=(10, 6))

for cell_size in unique_cell_sizes:

    ds = results[results['cell_size_m'] == cell_size]

    ds = ds.sort_values('threshold')

    plt.plot(ds['threshold'], ds['n_ponds'], 
             #label=f'Cell size: {round(cell_size, 2)} m', 
             linewidth=2, color=cmap(norm(cell_size)))

plt.xlabel('z$_{gw}$ (meters - relative to detrended DEM)', fontsize=12)
plt.ylabel('Distinct Ponds', fontsize=12)
plt.title('Distinct pond counts by z$_{gw}$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)


sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Cell Size (m)')
plt.legend()
plt.tight_layout()
plt.show()

# %%

max_pond_numbers = results.groupby('scale_factor').apply(
    lambda g: pd.Series({
        'cell_size_m': g['cell_size_m'].iloc[0],
        'max_ponds': g['n_ponds'].max(),
        'threshold': g.loc[g['n_ponds'].idxmax(), 'threshold']
    })
).reset_index()

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=max_pond_numbers,
    x='cell_size_m',
    y='max_ponds',
    palette='viridis',
    s=100
)

plt.title('Max Ponds vs. DEM Cell Size')
plt.xlabel('DEM Cell Size (m)')
plt.ylabel('Max Ponds from z$_{gw}$ curve')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

reference_domain = (min(results['threshold']), max(results['threshold']))
reference_z = np.arange(reference_domain[0], reference_domain[1], 0.02)
print(reference_z)
# %%
