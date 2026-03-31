# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


well_depth_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
out_summary_path = "D:/depressional_lidar/data/bradford/out_data/bradford_wetland_well_summary.csv"
well_data = pd.read_csv(well_depth_path)

well_ids = well_data['wetland_id'].unique()

# %% 2.0 Plot the well distributions

well_dry_days = [] 

for i in well_ids:

    temp = well_data[well_data['wetland_id'] == i].copy()
    data_full = temp["well_depth_m"].dropna()
    n_full = len(data_full)
    data_clean = temp.loc[temp["flag"] != 2, "well_depth_m"].dropna()
    n_clean = len(data_clean)

    bottom_threshold = data_clean.min()

    combined = pd.concat([data_full, data_clean], ignore_index=True)
    bins = np.histogram_bin_edges(combined, bins=50)

    # KDE plot for each well
    plt.figure(figsize=(5, 5))

    mean_full = data_full.mean()
    plt.hist(
        data_full,
        bins=bins,
        density=True,
        histtype='step',
        color="steelblue",
        label=f"All data (n={len(data_full)})"
    )
    plt.axvline(
        mean_full,
        color="steelblue",
        linestyle="--",
        linewidth=2,
        label=f"All mean: {mean_full:.2f}"
    )

    mean_clean = data_clean.mean()
    plt.hist(
        data_clean,
        bins=bins,
        density=True,
        histtype='step',
        color="darkorange",
        label=f"Excludes bottomed out (n={len(data_clean)})"
    )
    plt.axvline(
        mean_clean,
        color="darkorange",
        linestyle="-.",
        linewidth=2,
        label=f"Clean mean: {mean_clean:.2f}"
    )

    plt.axvline(
        bottom_threshold,
        color="maroon",
        linestyle="-",
        linewidth=2,
        label=f"Bottom-out Threshold"
    )

    plt.title(f"Wetland {i} well depth histogram")
    plt.xlabel("well_depth_m")
    plt.ylabel("Density")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.close()

    median_full = data_full.median()
    p75_full = np.percentile(data_full, 75)

    dry_status = {
        'wetland_id': i, 
        'median': median_full,
        'p75': p75_full,
        'bottom_threshold': bottom_threshold,
        'n_dry': n_full - n_clean,
        'prop_dry': (n_full - n_clean) / n_full
    }

    well_dry_days.append(dry_status)


# %% 3.0 Concatonate and write the output.

dry_df = pd.DataFrame(well_dry_days)
print(dry_df.sort_values('prop_dry'))
dry_df.to_csv(out_summary_path, index=False)


# %%

plot_df = dry_df.copy()
plot_df["bottom_threshold"] = pd.to_numeric(plot_df["bottom_threshold"], errors="coerce")
plot_df["prop_dry"] = pd.to_numeric(plot_df["prop_dry"], errors="coerce")
plot_df = plot_df.dropna(subset=["bottom_threshold", "prop_dry"])

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(
    plot_df["bottom_threshold"],
    plot_df["prop_dry"],
    s=45,
    alpha=0.8,
    color="teal",
    edgecolors="black",
    linewidths=0.4
)
ax.set_xlabel("bottom_threshold (m)")
ax.set_ylabel("prop_dry")
ax.set_title("Proportion Dry vs Bottom-Out Threshold")
ax.grid(alpha=0.25)
ax.legend()

# %%
