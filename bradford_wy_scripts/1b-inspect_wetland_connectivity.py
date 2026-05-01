# %% 1.0 Imports, directories and file paths
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wetland_utilities.basin_attributes import WetlandBasin
dem_buffer = 5

source_dem_path = 'D:/depressional_lidar/data/bradford/in_data/bradford_DEM_cleaned_USGS.tif'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
footprints_path = 'D:/depressional_lidar/data/bradford/out_data/bradford_well_basins.shp'
wetland_connectivity_path = 'D:/depressional_lidar/data/bradford/bradford_wetland_connect_logging_key.xlsx'

footprints = gpd.read_file(footprints_path)
well_point = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'rtk_z', 'geometry', 'site']]
    .rename(columns={'rtk_z': 'rtk_z'})
    .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)

wetland_ids = well_point['wetland_id'].unique().tolist()

connectivity = pd.read_excel(wetland_connectivity_path)

# %% 2.0 Visualize the wetland's DEM

results = []
cdfs = []

for i in wetland_ids:

    fp = footprints[footprints['wetland_id'] == i]
    delineated_basin = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path, 
        footprint=fp,
        transect_buffer=dem_buffer
    )
    connectivity_class = connectivity[connectivity['wetland_id'] == i].iloc[0]['connectivity']
    print(f'Well ID: {i}, Connectivity: {connectivity_class}')

    """
    Test delineations on rough basin shapes
    """

    # delineated_basin.visualize_shape(
    #     show_shape=True, 
    #     show_well=True, 
    #     show_deepest=True, 
    #     show_spill=True, 
    #     show_smoothed_spill=True
    # )
    delineated_basin.plot_basin_hypsometry(
        plot_points=True,
        plot_spill=True, 
        plot_smoothed_spill=True, 
        plot_contiguous_spill=True,
    )

    well_elev = delineated_basin.well_point.elevation_dem
    min_elev = delineated_basin.deepest_point.elevation

    max_fill_delineated, fill_dem_z = delineated_basin.max_fill_depth()
    max_fill_elev = max_fill_delineated + fill_dem_z


    elev_cdf = delineated_basin.calculate_hypsometry(method='total_cdf') # returned as a tuple

    for area, elev in zip(elev_cdf[0], elev_cdf[1]):
        cdfs.append({
            'wetland_id': i,
            'inundated_area': area,
            'elev_bin_center': elev
        })

    # Test simple spill on the undelineated basins
    # at 150m and 250m

    basin150 = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=150
    )
    max_fill150, fill150_dem_z = basin150.max_fill_depth()
    basin150_min = basin150.deepest_point.elevation

    basin200 = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=200
    )
    max_fill200, fill200_dem_z = basin200.max_fill_depth()
    basin200_min = basin200.deepest_point.elevation
    

    basin250 = WetlandBasin(
        wetland_id=i,
        well_point_info=well_point[well_point['wetland_id'] == i],
        source_dem_path=source_dem_path, 
        footprint=None,
        transect_buffer=250
    )
    max_fill250, fill250_dem_z = basin250.max_fill_depth()
    basin250_min = basin250.deepest_point.elevation

    # Compile the results

    r = {
        'wetland_id': i,
        'min_elev': min_elev,
        'well_elev': well_elev,

        'max_fill_delineated': max_fill_delineated,
        'max_fill_elev': max_fill_elev,
        # Agnostic of basin shape
        'max_fill150': max_fill150, 
        'basin150_dem_z': fill150_dem_z,
        'basin150_min': basin150_min,
        'max_fill200': max_fill200,
        'basin200_dem_z': fill200_dem_z,
        'basin200_min': basin200_min,
        'max_fill250': max_fill250,
        'basin250_dem_z': fill250_dem_z,
        'basin250_min': basin250_min
    }
    
    results.append(r)

# %% 2.0 Plot the wetland spill depths as a histogram

results_df = pd.DataFrame(results)

results_df.to_csv('D:/depressional_lidar/data/bradford/out_data/bradford_estimated_basin_spills.csv', index=False)

# Save hypsometry curves as a flat tidy CSV (one row per bin)
cdf_df = pd.DataFrame(cdfs)
cdf_df.to_csv('D:/depressional_lidar/data/bradford/out_data/bradford_hypsometry_curves.csv', index=False)

# %%
"""
# %% 3.0 Plot the wetland spill depths as a histogram
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(results_df['spill_depth'], bins=15, color='steelblue', edgecolor='black', alpha=0.6, label='Standard Spill')
ax.hist(results_df['smoothed_spill_depth'], bins=15, color='cyan', edgecolor='black', alpha=0.6, label='Smoothed Spill')
ax.hist(results_df['contiguous_spill_depth'], bins=15, color='green', edgecolor='black', alpha=0.6, label='Contiguous Spill')
ax.set_xlabel('Spill Depth (m)', fontsize=12)
ax.set_ylabel('Number of Wetlands', fontsize=12)
ax.set_title('Distribution of Wetland Spill Depths (Standard vs. Smoothed vs. Contiguous)', fontsize=14)
ax.grid(axis='y', alpha=0.3)

# Add summary stats
mean_spill = results_df['spill_depth'].mean()
median_spill = results_df['spill_depth'].median()
mean_spill_smooth = results_df['smoothed_spill_depth'].mean()
median_spill_smooth = results_df['smoothed_spill_depth'].median()
mean_spill_contig = results_df['contiguous_spill_depth'].mean()
median_spill_contig = results_df['contiguous_spill_depth'].median()

ax.axvline(mean_spill, color='red', linestyle='--', linewidth=2, label=f'Mean Standard: {mean_spill:.2f}m')
ax.axvline(median_spill, color='darkred', linestyle=':', linewidth=2, label=f'Median Standard: {median_spill:.2f}m')
ax.axvline(mean_spill_smooth, color='orange', linestyle='--', linewidth=2, label=f'Mean Smoothed: {mean_spill_smooth:.2f}m')
ax.axvline(median_spill_smooth, color='darkorange', linestyle=':', linewidth=2, label=f'Median Smoothed: {median_spill_smooth:.2f}m')
ax.axvline(mean_spill_contig, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean Contiguous: {mean_spill_contig:.2f}m')
ax.axvline(median_spill_contig, color='forestgreen', linestyle=':', linewidth=2, label=f'Median Contiguous: {median_spill_contig:.2f}m')
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()

# print(f"\nSpill Depth Summary:")
# print(f"  Mean: {mean_spill:.2f}m")
# print(f"  Median: {median_spill:.2f}m")
# print(f"  Min: {results_df['spill_depth'].min():.2f}m")
# print(f"  Max: {results_df['spill_depth'].max():.2f}m")
# print(f"  Std Dev: {results_df['spill_depth'].std():.2f}m")
# print(f"  N wetlands: {len(results_df)}")

# %% 4.0 Box plot of spill depth by connectivity class

# Merge results with connectivity data
results_with_connectivity = results_df.merge(
    connectivity[['wetland_id', 'connectivity']], 
    on='wetland_id', 
    how='left'
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Standard spill depth boxplot
results_with_connectivity.boxplot(column='spill_depth', by='connectivity', ax=ax1)
ax1.set_xlabel('Connectivity Class', fontsize=12)
ax1.set_ylabel('Spill Depth (m)', fontsize=12)
ax1.set_title('Standard Spill Depth by Connectivity Class', fontsize=12)
ax1.get_figure().suptitle('')  # Remove automatic title on first subplot

# Smoothed spill depth boxplot
results_with_connectivity.boxplot(column='smoothed_spill_depth', by='connectivity', ax=ax2)
ax2.set_xlabel('Connectivity Class', fontsize=12)
ax2.set_ylabel('Spill Depth (m)', fontsize=12)
ax2.set_title('Smoothed Spill Depth by Connectivity Class', fontsize=12)

# Contiguous spill depth boxplot
results_with_connectivity.boxplot(column='contiguous_spill_depth', by='connectivity', ax=ax3)
ax3.set_xlabel('Connectivity Class', fontsize=12)
ax3.set_ylabel('Spill Depth (m)', fontsize=12)
ax3.set_title('Contiguous Spill Depth by Connectivity Class', fontsize=12)

fig.suptitle('Spill Depth Comparison by Connectivity Class', fontsize=14, y=1.00)

plt.tight_layout()
plt.show()

print("\nStandard Spill Depth by Connectivity Class:")
print(results_with_connectivity.groupby('connectivity')['spill_depth'].describe())

print("\nSmoothed Spill Depth by Connectivity Class:")
print(results_with_connectivity.groupby('connectivity')['smoothed_spill_depth'].describe())

"""
