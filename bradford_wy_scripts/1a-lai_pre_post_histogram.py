# %% 1.0 Libraries and filepaths

import rasterio as rio
from rasterio import features
import geopandas as gpd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

data_dir = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/lai_images/'

early_path = f'{data_dir}/LAI_composite_2019-06-01_to_2019-12-31.tif'
late_path = f'{data_dir}/LAI_composite_2025-06-01_to_2025-12-31.tif'
bradford_boundary_path = 'D:/depressional_lidar/data/bradford/bradford_boundary.shp'
well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
nwi_wetlands_path = 'D:/depressional_lidar/data/bradford/in_data/original_basins/bradford_nwi_polygons.shp'


nwi_wetlands = gpd.read_file(nwi_wetlands_path)
bounds = gpd.read_file(bradford_boundary_path)

well_points = (
    gpd.read_file(well_points_path)[['wetland_id', 'type', 'site', 'geometry']]
    .query("type in ['main_doe_well', 'aux_wetland_well']")
)
def read_lai_raster(path: str):
    """Read a single-band LAI raster and convert all masked cells to NaN."""
    with rio.open(path) as src:
        arr = src.read(1, masked=True).astype('float32').filled(np.nan)
        transform = src.transform
        crs = src.crs
    return arr, transform, crs

# %% 2.0 Read the LAI rasters

early_lai, early_transform, early_crs = read_lai_raster(early_path)
late_lai, late_transform, late_crs = read_lai_raster(late_path)

bounds.to_crs(early_crs, inplace=True) # Covert bounds to match LAI crs
nwi_wetlands = nwi_wetlands.to_crs(early_crs)
well_points = well_points.query("site == 'Bradford'")
well_points = well_points.to_crs(early_crs)

area_crs = bounds.estimate_utm_crs()
bounds_area = bounds.to_crs(area_crs)
nwi_wetlands_area = nwi_wetlands.to_crs(area_crs)

boundary_geom = bounds_area.geometry.union_all()
nwi_within_boundary = nwi_wetlands_area.geometry.intersection(boundary_geom)
nwi_within_boundary_area = nwi_within_boundary.area.sum()
boundary_area = boundary_geom.area
nwi_wetland_proportion = nwi_within_boundary_area / boundary_area

print(
    f'NWI wetlands within boundary: {nwi_within_boundary_area:,.2f} '
    f'[{nwi_wetland_proportion:.3%} of boundary area]'
)

# Flatten arrays and remove NaN values
early_flat = early_lai[np.isfinite(early_lai)]
print(np.median(early_flat))

late_flat = late_lai[np.isfinite(late_lai)]
print(np.median(late_flat))

print(f"Early LAI: {len(early_flat)} valid pixels")
print(f"Late LAI: {len(late_flat)} valid pixels")

# Rasterize wetlands to split wetland/upland pixels.
wetland_mask = features.rasterize(
    [(geom, 1) for geom in nwi_wetlands.geometry if geom is not None and not geom.is_empty],
    out_shape=early_lai.shape,
    transform=early_transform,
    fill=0,
    all_touched=True,
    dtype='uint8'
).astype(bool)

early_wetland = early_lai[wetland_mask & np.isfinite(early_lai)]
early_upland = early_lai[(~wetland_mask) & np.isfinite(early_lai)]

late_wetland = late_lai[wetland_mask & np.isfinite(late_lai)]
late_upland = late_lai[(~wetland_mask) & np.isfinite(late_lai)]

# %% 3.0 Make a simple box plot comparing pre and post LAI
fig, ax = plt.subplots(figsize=(10, 10))

box_data = [
    early_upland, late_upland,
    early_wetland, late_wetland,
]

box_labels = [
    '2019', '2025',
    '2019', '2025',
]

bp = ax.boxplot(box_data,
                labels=box_labels,
                patch_artist=True,
                widths=0.6,
                whis=(5, 95),
                showfliers=False,
                boxprops=dict(linewidth=3),     
                whiskerprops=dict(linewidth=3),
                capprops=dict(linewidth=3),
                medianprops=dict(linewidth=3, color='black'))

# Apply colors: upland and wetland same color within group, but hatching for 2019
upland_color = '#E1CCA4'
wetland_color = '#70A800'

# 2019 = hatched, 2025 = solid
for i, patch in enumerate(bp['boxes']):
    if i < 2:  # Upland
        patch.set_facecolor(upland_color)
        if i == 1:  # 2025
            patch.set_hatch('///')
    else:  # Wetland
        patch.set_facecolor(wetland_color)
        if i == 3:  # 2025
            patch.set_hatch('///')

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# ax.spines['bottom'].set_linewidth(2.5)
# ax.spines['left'].set_linewidth(2.5)

ax.set_ylabel('LAI', fontsize=28, fontweight='bold')
ax.tick_params(labelsize=18)

# Add group labels
ax.set_xticks([1.5, 3.5])
ax.set_xticklabels(['Upland', 'Wetland'], fontsize=20, fontweight='bold')

# Customize x-axis for year labels
ax.set_xticks([1, 2, 3, 4], minor=True)
ax.xaxis.set_tick_params(which='minor', labelsize=14)

# Create custom legend with colors and patterns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=upland_color, edgecolor='black', hatch='///', label='Upland 2019'),
    Patch(facecolor=upland_color, edgecolor='black', label='Upland 2025'),
    Patch(facecolor=wetland_color, edgecolor='black', hatch='///', label='Wetland 2019'),
    Patch(facecolor=wetland_color, edgecolor='black', label='Wetland 2025'),
]

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
          ncol=2, fontsize=16, framealpha=0.9, edgecolor='black')

for label in ax.get_yticklabels():
    label.set_fontweight('bold')

#plt.tight_layout()
plt.show()

# %% 3.1 Print values associated with boxplots
for label, values in zip(
    ['Early upland', 'Late upland', 'Early wetland', 'Late wetland'],
    box_data,
):
    mean_val = np.mean(values)
    median_val = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    iqr_val = q75 - q25
    sd_val = np.std(values)

    print(
        f'{label}: mean={mean_val:.3f}, median={median_val:.3f}, '
        f'IQR={iqr_val:.3f}, sd={sd_val:.3f}'
    )


# %% 4.0 Make an LAI change map

lai_change = late_lai - early_lai
valid_change = lai_change[np.isfinite(lai_change)]

# # Use robust percentile clipping so outliers do not dominate the map colors.
abs_lim = np.nanpercentile(np.abs(valid_change), 98)

norm = TwoSlopeNorm(vmin=-abs_lim, vcenter=0.0, vmax=abs_lim)

height, width = lai_change.shape
left, bottom, right, top = rio.transform.array_bounds(height, width, early_transform)

fig, ax = plt.subplots(figsize=(10, 16), facecolor='#ffffff')
ax.set_facecolor('#ffffff')


im = ax.imshow(
    lai_change,
    cmap="RdBu", 
    norm=norm,
    extent=(left, right, bottom, top),
    origin='upper',
    interpolation='nearest'
)

bounds.boundary.plot(ax=ax, color='#1e5bd8', linewidth=2.2, zorder=4)
well_points.plot(ax=ax, color='black', markersize=75, zorder=5)

cbar = fig.colorbar(
    im,
    ax=ax,
    orientation='horizontal',
    fraction=0.1,   # thickness
    pad=0.02,        # spacing from map
    shrink=0.65      # length of colorbar
)
cbar.set_label('LAI change (2025 - 2019)', fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=16)

gain_pct = np.mean(valid_change > 0) * 100
loss_pct = np.mean(valid_change < 0) * 100
stable_pct = np.mean(np.abs(valid_change) <= 0.10) * 100
med_change = np.nanmedian(valid_change)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

xpad = (right - left) * 0.05
ypad = (top - bottom) * 0.01

ax.set_xlim(left - xpad, right + xpad)
ax.set_ylim(bottom - ypad, top + ypad)

for side in ['top', 'right', 'bottom', 'left']:
    ax.spines[side].set_visible(False)

plt.show()

# %% LAI Change Timeseries


