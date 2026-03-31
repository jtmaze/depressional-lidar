
# %% 1.0 Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# %% 2.0 Report results to dictionaries
# Nueman 1972 model
n72 = {
    0: -6.75,
    50: -3.2,
    100: -2.4,
    200: -1.6,
    500: -0.65,
    700: -0.45,
    1000: -0.28,
    1200: -0.22,
    1500: -0.16,
    1700: -0.13,
    2000: -0.10
}

# Papadopulas-Cooper 1967
# Demand: 105,633.00 gpd, Time: 20 days
pc_20d = {
    0: -1.1,
    50: -0.57,
    100: -0.47,
    200: -0.39,
    500: -0.27,
    700: -0.22,
    1000: -0.18,
    1200: -0.16,
    1500: -0.13,
    1700: -0.12,
    2000: -0.1
}
# Demand: 116,196.00 gpd, Time: 30 days
pc_30d = {
    0: -1.64,
    50: -0.89,
    100: -0.74,
    200: -0.61,
    500: -0.43,
    700: -0.37,
    1000: -0.3,
    1200: -0.27,
    1500: -0.28,
    1700: -0.2,
    2000: -0.18
}

# Thiem steady-state method
# New scenarios (from latest attachments)
thiem_avgd = {
    800: -1.9,
    1000: -1.6,
    1500: -1.0,
    2000: -0.6,
    2700: -0.2,
}

# use midpoints where ranges were given
thiem_lt = {
    800: -0.7,
    1000: -0.55,
    1500: -0.35,
    2000: -0.2,
    2700: -0.05,
}

# %% 3.0 Combine scenarios
# pack scenarios
scenarios = {
    'n72': n72,
    'pc_20d': pc_20d,
    'pc_30d': pc_30d,
    'thiem_avgd': thiem_avgd,
    'thiem_lt': thiem_lt,
}

all_distances = sorted({d for s in scenarios.values() for d in s.keys()})
df = pd.DataFrame(index=all_distances)
for name, dct in scenarios.items():
    df[name] = [dct.get(d, np.nan) for d in all_distances]
df.index.name = 'distance_ft'

# %% 4.0 Plot points with power-fitting

scenarios = {
    'n72': n72,
    'pc_20d': pc_20d,
    'pc_30d': pc_30d,
    'thiem_avgd': thiem_avgd,
    'thiem_lt': thiem_lt,
}

all_distances = sorted({d for s in scenarios.values() for d in s.keys()})
df = pd.DataFrame(index=all_distances)
for name, dct in scenarios.items():
    df[name] = [dct.get(d, np.nan) for d in all_distances]
df.index.name = 'distance_ft'


# %% 5.0 Plot points with simple shifted-power fit

def shifted_power(x, A, x0, p):
    return -A / (x + x0) ** p

def fit_shifted_power(x_arr, y_arr):
    mask = ~np.isnan(y_arr)

    x_fit = x_arr[mask].astype(float)
    y_fit = y_arr[mask].astype(float)

    popt, _ = curve_fit(
        shifted_power,
        x_fit,
        y_fit,
        p0=(50, 100, 1),          # simple starting guess
        bounds=([0, 0, 0], [1e6, 1e5, 5]),
        maxfev=10000
    )
    return popt


# Map short names to long names
scenario_labels = {
    'n72': 'Neumann 1972',
    'pc_20d': 'Papadopulos-Cooper 1967 (105,633 gpd, 20 days)',
    'pc_30d': 'Papadopulos-Cooper 1967 (116,196 gpd, 30 days)',
    'thiem_avgd': 'Thiem (time-averaged equivalent)',
    'thiem_lt': 'Thiem (long-term adjustec values)',
}

fig, ax = plt.subplots(figsize=(11, 6))
x_plot = np.linspace(1, max(all_distances), 500)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, name in enumerate(df.columns):
    x = np.array(df.index.values, dtype=float)
    y = df[name].values

    color = colors[i % len(colors)]
    label = scenario_labels.get(name, name)
    ax.scatter(x, y, label=label, color=color, s=50, zorder=3)

    params = fit_shifted_power(x, y)

    A, x0, p = params
    yfit = shifted_power(x_plot, A, x0, p)
    ax.plot(x_plot, yfit, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    print(f'{label}: y = -{A:.4g} / (x + {x0:.4g})^{p:.4g}')

plt.xlim(0, 3000)
plt.ylim(-5, 0)
plt.xlabel('Distance (ft)', fontsize=12)
plt.ylabel('Drawdown (ft)', fontsize=12)
plt.title('Aquifer Drawdown Models', fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
