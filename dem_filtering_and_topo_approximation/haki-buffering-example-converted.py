# %%

import numpy as np
import matplotlib.pyplot as plt

# Compute effective length of (possibly fractal) wetland perimeter based on
# relevant horizontal length scale (e.g., average roaming distance)

plt.close('all')

## GENERATE RANDOM WETLAND

n = 101    # n by n grid cells
a = 10     # for moving average (spatial correlation)

z1 = np.random.randn(n+a, n+a)

z = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        z[i, j] = np.mean(z1[i:i+a, j:j+a])   # create some spatial correlation

mz = np.min(z)
w = np.ones_like(z)
w[z > mz/4] = 0      # cutoff for wet vs dry

x, y = np.meshgrid(np.arange(n), np.arange(n))

## WET-DRY MATRIX ANALYSIS

s = 2      # distance scale

# Get coordinates of wet and dry cells
wet_indices = np.where(w == 1)
dry_indices = np.where(w == 0)

xw = x[wet_indices]   # x vector of wet cells
yw = y[wet_indices]   # y vector of wet cells
xd = x[dry_indices]   # x vector of dry cells
yd = y[dry_indices]   # y vector of dry cells

# Create coordinate matrices for distance calculation
xxw = xw[:, np.newaxis]  # wet x coordinates as column vector
xxd = xd[np.newaxis, :]  # dry x coordinates as row vector
yyw = yw[:, np.newaxis]  # wet y coordinates as column vector
yyd = yd[np.newaxis, :]  # dry y coordinates as row vector

# Calculate distance matrix between all wet and dry cells
dist = np.sqrt((xxw - xxd)**2 + (yyw - yyd)**2)

# Find cells within distance s
nearwet = np.min(dist, axis=0) <= s  # dry cells near wet cells
neardry = np.min(dist, axis=1) <= s  # wet cells near dry cells

## PLOTS

# Plot the wetland surface
plt.figure()
plt.imshow(w, extent=[0, n-1, 0, n-1], origin='lower', cmap='viridis')
plt.axis('equal')
plt.tight_layout()
plt.title('Wetland Surface')

# Plot near-boundary cells
plt.figure()
plt.plot(xw[neardry], yw[neardry], '.b', label='Wet cells near dry', markersize=2)
plt.plot(xd[nearwet], yd[nearwet], '.g', label='Dry cells near wet', markersize=2)
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Boundary Analysis')
plt.tight_layout()

plt.show()