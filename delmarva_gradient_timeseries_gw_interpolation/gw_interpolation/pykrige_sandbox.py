# %% 1.0 Packages 

import numpy as np
import matplotlib.pyplot as plt
import pykrige as pkr

# %% 2.0 Toy data for testing

# Define grid
nx, ny, = 101, 101
x = np.linspace(0, 100, nx)
y = np.linspace(0, 100, ny)
X, Y = np.meshgrid(x, y)

# Populate the arrays with example data
gradient = X + (Y / 2)
sine_pattern = np.sin(X / 20) * np.cos(Y / 20) * 20
random_noise = np.random.RandomState(42).rand(nx, ny) * 5
combo = gradient + sine_pattern + random_noise

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
patterns = [gradient, sine_pattern, random_noise, combo]
titles = ["Gradient (X+Y/2)", "Sine-Cosine Pattern", 'Noise', 'Combined']

for ax, pattern, title in zip(axes.flat, patterns, titles):
    im = ax.imshow(pattern, origin='lower')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# %% 3.0 Process for pyKrig inputs must be flat
coords = np.column_stack((X.flatten(), Y.flatten()))
values_gradient = gradient.flatten()
values_sine = sine_pattern.flatten()
values_random_noise = random_noise.flatten()
values_combo = combo.flatten()

# %% 4.0 Sample the data

sample_coords = np.array([
    (8, 8), (8, 50), (10, 75),
    (40, 15), (37, 37), (42, 90),
    (69, 5), (72, 42), (70, 87),
    (93, 12), (97, 40), (88, 95)
])
sample_x = sample_coords[:, 0]
sample_y = sample_coords[:, 1]

sample_count = 20

sample_x = np.random.randint(0, 101, size=sample_count)
sample_y = np.random.randint(0, 101, size=sample_count)
sample_coords = np.array(
    [(sample_x[i], sample_y[i]) for i in range(sample_count)]
)

sample_vals = combo[
    sample_coords[:, 1], # y index (row)
    sample_coords[:, 0] # x index (column)
]
# %% Test pykrige

ok = pkr.ok.OrdinaryKriging(
    sample_x, 
    sample_y,
    sample_vals,
    variogram_model='gaussian',
    nlags=8,
    verbose=True,
    enable_plotting=True,
)

# %% Plot the interpolated z-array

z_result, ss = ok.execute("grid", x, y)

# %%
plt.figure(figsize=(16, 6))

# Left subplot: Original 'combo' array
ax1 = plt.subplot(1, 2, 1)
cf1 = ax1.contourf(x, y, combo, cmap="viridis")
plt.colorbar(cf1, ax=ax1, label="Combo Value")
ax1.scatter(sample_x, sample_y, c='red', s=50, label='Sample Points')
ax1.set_title("Original Combo Array")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()

# Right subplot: Ordinary Kriging result (z_result)
ax2 = plt.subplot(1, 2, 2)
cf2 = ax2.contourf(x, y, z_result, cmap="viridis")
plt.colorbar(cf2, ax=ax2, label="Kriged Z Value")
ax2.scatter(sample_x, sample_y, c='red', s=50, label='Sample Points')
ax2.set_title("Ordinary Kriging Output")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()

plt.tight_layout()
plt.show()
# %%
