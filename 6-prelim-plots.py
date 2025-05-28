# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./out_data/region_props_on_depressions.csv')
print(df.columns)

# %% 2.0 Some quick exploratory plots

plt.figure(figsize=(7, 7))
plt.plot(df['threshold'], df['n_ponds'], marker='o', label='Number of Depressions')
plt.xlabel('Relative Groundwater Depth (m)')
plt.ylabel('Number of Depressions')
plt.show()

# %% 7.1

plt.figure(figsize=(7, 7))
plt.plot(df['threshold'], df['mean_feature_area_m2'], marker='o', label='Mean Pond Area')
plt.xlabel('Relative Groundwater Depth (m)')
#plt.xlim(-1.5, 0.0)
plt.ylabel('Mean Pond Area (m²)')
plt.show()

# %%
plt.figure(figsize=(7, 7))
plt.plot(df['threshold'], df['mean_feature_area_m2'], marker='o', label='Mean Pond Area')
plt.xlabel('Relative Groundwater Depth (m)')
plt.xlim(-1.5, 0.0)
plt.ylim(0, 100_000)
plt.ylabel('Mean Pond Area (m²)')
plt.show()

# %% Total perimeter

plt.figure(figsize=(7, 7))
plt.plot(df['threshold'], df['total_perimeter_m'], marker='o', label='Total Perimeter')
#plt.plot(df['threshold'], df['total_perimeter_crofton_m'], marker='x', label='Total Crofton Perimeter')
plt.xlabel('Relative Groundwater Depth (m)')
plt.ylabel('Perimeter (meters)')
plt.title('Total Perimeter vs. GW Depth')
plt.show()


# %% Total perimeter versus number of ponds

fig, ax1 = plt.subplots(figsize=(8, 7))

# Plot perimeter data on left axis
color1 = 'tab:blue'
ax1.set_xlabel('Relative Groundwater Depth (m)')
ax1.set_ylabel('Perimeter (meters)', color=color1)
ax1.plot(df['threshold'], df['total_perimeter_m'], marker='o', label='Total Perimeter', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# Create second y-axis for number of ponds
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Number of Ponds', color=color2)
ax2.plot(df['threshold'], df['n_ponds'], marker='x', label='Number of Ponds', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Create third y-axis for inundated area
# Make room for the third axis
fig.subplots_adjust(right=0.75)  
ax3 = ax1.twinx()
# Move the third axis to the right
ax3.spines['right'].set_position(('outward', 60))
color3 = 'tab:green'
ax3.set_ylabel('Inundated Area (%)', color=color3)
ax3.plot(df['threshold'], df['inundated_frac'], marker='^', label='Inundated Area', color=color3)
ax3.tick_params(axis='y', labelcolor=color3)

# Add legend with all three lines
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

fig.tight_layout()
plt.show()
# %%
