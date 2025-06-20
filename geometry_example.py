# %%

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

example = gpd.read_file('./bradford/temp/example_polygon.shp')
print(example.crs)
example_projected = example.to_crs('EPSG:26916')
example_projected = example_projected.iloc[0]
example_projected = gpd.GeoDataFrame(geometry=[example_projected.geometry], crs='EPSG:26916')

# %%

example_projected.plot()
# %%

buffers = range(0, 200, 2)

results = []

for i in buffers:

    temp = example_projected.buffer(i)
    area = temp.area
    perimeter = temp.boundary.length
    
    r = {
        'buffer': i,
        'area': area,
        'perimeter': perimeter}
    
    results.append(r)

    if i % 25 == 0: # Mod function
        temp.plot()

# %% 

df = pd.DataFrame(results)
step = 2
forward_area = df['area'].shift(-1)
backward_area = df['area'].shift(1)
central_diff = (forward_area - backward_area) / (2 * step)
df['dA/db'] = central_diff


# %%

# Plot perimeter and dA/db on same plot with different y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot perimeter on primary y-axis with log scale
ax1.plot(df['buffer'], df['perimeter'], 'b-', label='Perimeter')
ax1.set_xlabel('Buffer Distance')
ax1.set_ylabel('Perimeter', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create second y-axis and plot dA/db with log scale
ax2 = ax1.twinx()
ax2.plot(df['buffer'], df['dA/db'], 'r-', label='dA/db')
ax2.set_ylabel('dA/db', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add title and adjust layout
plt.title('Perimeter and Rate of Area Change vs Buffer Distance')
plt.tight_layout()

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.show()

# %%
