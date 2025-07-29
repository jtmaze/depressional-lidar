# %% 1.0 Libraries and file paths
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('D:/depressional_lidar/data/')
site = 'bradford'

#da_dz = pd.read_csv(f'./{site}/simple_inundated_area.csv')
results = pd.read_csv(f'./{site}/out_data/bradford_region_props_on_depressions_native.csv')

df = results
print(df.columns)

# %% 2.0 Some quick exploratory plots

plt.figure(figsize=(7, 7))
plt.plot(df['threshold'], df['n_ponds'], marker='o', label='Number of Depressions')
plt.xlabel('Relative Groundwater Depth (m)')
plt.ylabel('Number of Depressions')
plt.show()

# %% Total perimeter

plt.figure(figsize=(7, 7))
plt.plot(df['threshold'], df['total_perimeter_m'], marker='o', label='Total Perimeter')
#plt.plot(df['threshold'], df['total_perimeter_crofton_m'], marker='x', label='Total Crofton Perimeter')
plt.xlabel('Relative Groundwater Depth (m)')
plt.ylabel('Perimeter (meters)')
plt.title('Total Perimeter vs. GW Depth')
plt.show()

# %% Calculate dA/dS max with central difference
# Calculate the central difference for dA/dS
# Formula: f'(x) = (f(x+h) - f(x-h)) / (2h) where h is the step size
step_size = 0.02
step_size_km = step_size * 1_000
forward_area = df['inundated_area_m2'].shift(-1) / 1_000_000  # f(x+h) 
backward_area = df['inundated_area_m2'].shift(1) / 1_000_000  # f(x-h) 
central_difference = (forward_area - backward_area) / (2 * step_size)

# Assign the result to the dataframe
df['dA/dS'] = central_difference

# %%
fig, ax1 = plt.subplots(figsize=(12, 7))

color1 = 'tab:blue'
ax1.set_xlabel('Relative Groundwater Depth (m)')
ax1.set_ylabel('dA/dz (perimeter proxy km²/km)', color=color1)
ax1.plot(df['threshold'], df['dA/dS'], marker='o', color=color1, label='dA/dz (perimeter proxy km²/km)')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Summed Perimeter (kilometers)', color=color2)
ax2.plot(df['threshold'], df['total_perimeter_crofton_m'] / 1_000, marker='+',
         color='tab:orange', label='Summed Perimeter (Crofton, km)')
ax2.plot(df['threshold'], df['total_perimeter_m'] / 1_000, marker='x',
         color='red', label='Summed Perimeter (m, km)')
ax2.tick_params(axis='y', labelcolor=color2)

fig.subplots_adjust(right=0.65)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
color3 = 'tab:green'
ax3.set_ylabel('Inundated Area (km²)', color=color3)
ax3.plot(df['threshold'], df['inundated_area_m2'] / 1_000_000, marker='^',
         color=color3, label='Inundated Area (km²)')
ax3.tick_params(axis='y', labelcolor=color3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
           bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)

fig.tight_layout()
plt.subplots_adjust(right=0.65, bottom=0.2)
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
