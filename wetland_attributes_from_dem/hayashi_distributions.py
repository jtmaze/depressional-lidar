# %%

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from basin_attributes import WetlandBasin

site = 'bradford'

# %%

source_dem = f'D:/depressional_lidar/data/{site}/in_data/{site}_DEM_cleaned_veg.tif'
assigned_basins_path = f'D:/depressional_lidar/data/{site}/in_data/{site}_basins_assigned_wetland_ids.shp'
all_basins_path = f'D:/depressional_lidar/data/{site}/in_data/original_basins/{site}_depression_polygons.shp'

wetlands = gpd.read_file(all_basins_path)
wetlands['wetland_id'] = wetlands.index
print(len(wetlands))

# %%
# wetlands_plot = wetlands[wetlands['area_m2'] <= 10_000]
# plt.hist(wetlands_plot['area_m2'], bins=30, alpha=0.7)
# plt.title('Distribution of Wetland Areas')
# plt.xlabel('Area (m²)')
# plt.ylabel('Frequency')
# plt.show()

# %%
print(len(wetlands))
wetlands = wetlands[wetlands['area_m2'] >= 500]
print(len(wetlands))

wetland_ids = wetlands['wetland_id'].sample(n=10, random_state=42).tolist()




# %%
results = []
for i in wetland_ids:

    footprint = wetlands[wetlands['wetland_id'] == i]
    basin = WetlandBasin(
        wetland_id=i,
        source_dem_path=source_dem,
        footprint=footprint,
        well_point_info=None,
        transect_method='deepest',
        transect_n=12,
        transect_buffer=10
    )

    #basin.visualize_shape(show_centroid=True, show_deepest=True)
    basin.radial_transects_map(uniform=False)
    #basin.radial_transects_map(uniform=True)
    #basin.plot_individual_radial_transects(uniform=False)
    basin.plot_individual_radial_transects(uniform=True)
    #basin.plot_hayashi_p(r0=1, r1=25, uniform=False)
    #basin.plot_hayashi_p(r0=1, r1=None, uniform=True)

    hayashi_uniform = basin.calc_hayashi_p_uniform_z(r0=1)
    mean_uniform = hayashi_uniform['p'].mean()
    hayashi_incorrect = basin.calc_hayashi_p_defined_r(r0=1, r1=25)
    mean_incorrect = hayashi_incorrect['p'].mean()

    results.append({
            'wetland_id': i,
            'mean_uniform': mean_uniform,
            'mean_incorrect': mean_incorrect,
            'hayashi_uniform_p': hayashi_uniform['p'].tolist(),
            'hayashi_incorrect_p': hayashi_incorrect['p'].tolist()
    })

# %%
results_df = pd.DataFrame(results)

# %%

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

mean_uniform_clean = results_df['mean_uniform'].replace([np.inf, -np.inf], np.nan).dropna()
mean_incorrect_clean = results_df['mean_incorrect'].replace([np.inf, -np.inf], np.nan).dropna()

all_uniform_p = [p for sublist in results_df['hayashi_uniform_p'] for p in sublist]
all_uniform_p_clean = [p for p in all_uniform_p if np.isfinite(p)]

all_incorrect_p = [p for sublist in results_df['hayashi_incorrect_p'] for p in sublist]
all_incorrect_p_clean = [p for p in all_incorrect_p if np.isfinite(p)]

all_values = list(mean_uniform_clean) + list(mean_incorrect_clean) + all_uniform_p_clean + all_incorrect_p_clean
x_min, x_max = min(all_values), max(all_values)

# Calculate statistics
mean_uniform_stats = (mean_uniform_clean.mean(), mean_uniform_clean.std())
mean_incorrect_stats = (mean_incorrect_clean.mean(), mean_incorrect_clean.std())
all_uniform_stats = (np.mean(all_uniform_p_clean), np.std(all_uniform_p_clean))
all_incorrect_stats = (np.mean(all_incorrect_p_clean), np.std(all_incorrect_p_clean))

axes[0, 0].hist(mean_uniform_clean, bins=30, alpha=0.7, color='blue')
axes[0, 0].set_title(f'Distribution of Mean Uniform Hayashi P (n={len(mean_uniform_clean)})\nμ={mean_uniform_stats[0]:.3f}, σ={mean_uniform_stats[1]:.3f}')
axes[0, 0].set_xlabel('Hayashi Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_xlim(x_min, x_max)

axes[0, 1].hist(mean_incorrect_clean, bins=30, alpha=0.7, color='red')
axes[0, 1].set_title(f'Distribution of Mean Incorrect Hayashi P (n={len(mean_incorrect_clean)})\nμ={mean_incorrect_stats[0]:.3f}, σ={mean_incorrect_stats[1]:.3f}')
axes[0, 1].set_xlabel('Hayashi Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_xlim(x_min, x_max)

axes[1, 0].hist(all_uniform_p_clean, bins=30, alpha=0.7, color='blue')
axes[1, 0].set_title(f'All Uniform Hayashi P Values (n={len(all_uniform_p_clean)})\nμ={all_uniform_stats[0]:.3f}, σ={all_uniform_stats[1]:.3f}')
axes[1, 0].set_xlabel('Hayashi Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xlim(x_min, x_max)

axes[1, 1].hist(all_incorrect_p_clean, bins=30, alpha=0.7, color='red')
axes[1, 1].set_title(f'All Incorrect Hayashi P Values (n={len(all_incorrect_p_clean)})\nμ={all_incorrect_stats[0]:.3f}, σ={all_incorrect_stats[1]:.3f}')
axes[1, 1].set_xlabel('Hayashi Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_xlim(x_min, x_max)

plt.tight_layout()
plt.show()

# Print summary of filtered data
print(f"Original uniform means: {len(results_df)}, After filtering: {len(mean_uniform_clean)}")
print(f"Original incorrect means: {len(results_df)}, After filtering: {len(mean_incorrect_clean)}")
print(f"Original uniform p values: {len(all_uniform_p)}, After filtering: {len(all_uniform_p_clean)}")
print(f"Original incorrect p values: {len(all_incorrect_p)}, After filtering: {len(all_incorrect_p_clean)}")

# %%
# %%
