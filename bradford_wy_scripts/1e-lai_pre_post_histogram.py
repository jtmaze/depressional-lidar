# %% 1.0 Libraries and filepaths

import rasterio as rio
import numpy as np 
import matplotlib.pyplot as plt

data_dir = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/lai_images/'

early_path = f'{data_dir}/LAI_composite_2019-06-01_to_2019-12-31.tif'
late_path = f'{data_dir}/LAI_composite_2025-06-01_to_2025-12-31.tif'


# %% 2.0 Read the LAI rasters

with rio.open(early_path) as src:
    early_lai = src.read(1)  
    
with rio.open(late_path) as src:
    late_lai = src.read(1)

# Flatten arrays and remove NaN values
early_flat = early_lai.flatten()
early_flat = early_flat[~np.isnan(early_flat)]
print(np.median(early_flat))

late_flat = late_lai.flatten()
late_flat = late_flat[~np.isnan(late_flat)]
print(np.median(late_flat))

print(f"Early LAI: {len(early_flat)} valid pixels")
print(f"Late LAI: {len(late_flat)} valid pixels")

# %% 3.0 Make a simple box plot comparing pre and post LAI
fig, ax = plt.subplots(figsize=(7, 7))


bp = ax.boxplot([early_flat, late_flat],
                labels=['2019 (c)', '2025 (d)'],
                patch_artist=True,
                widths=0.6,                    
                boxprops=dict(linewidth=3),     
                whiskerprops=dict(linewidth=3),
                capprops=dict(linewidth=3),
                medianprops=dict(linewidth=3, color='black'))

for patch in bp['boxes']:
    patch.set_facecolor('lightgrey')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)

ax.set_ylabel('LAI', fontsize=34, fontweight='bold')
#ax.set_title('LAI Comparison: Pre vs Post Logging', fontsize=18)
ax.tick_params(labelsize=34)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.show()


# %% 4.0 Print pre and post medians
