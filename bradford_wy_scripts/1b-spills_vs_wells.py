# %% 1.0 Libraries and file paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import f_oneway
#from matplotlib.lines import Line2D

data_dir = "D:/depressional_lidar/data/bradford/"

est_spills_path = f"{data_dir}/out_data/bradford_estimated_basin_spills.csv"
well_data_path = f"{data_dir}/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
connectivity_path = f"{data_dir}/bradford_wetland_connect_logging_key.xlsx"

est_spills = pd.read_csv(est_spills_path)
well_data = pd.read_csv(well_data_path)
connect = pd.read_excel(connectivity_path)

# %% 2.0 Convert delineated spill to basin-min depth and compute modal well depth

"""
NOTE:
Due to ditching, the elevation with the deepest spill depth (i.e., lowest depression point)
might not be the delineated area's absolute lowest point. So we convert delineated spill
depth to a basin-minimum depth reference.
"""

est_spills["well_to_min_delineated"] = est_spills["well_elev"] - est_spills["min_elev"]
est_spills["spill_min_to_basin_min"] = (
    est_spills["max_fill_elev"] - est_spills["max_fill_delineated"]
) - est_spills["min_elev"]
est_spills["delineated_spill_h_min"] = (
    est_spills["max_fill_delineated"] + est_spills["spill_min_to_basin_min"]
)


def compute_modal_depth(depths, bin_width=0.05):
    vals = pd.Series(depths).dropna().to_numpy()

    edges = np.arange(vals.min(), vals.max() + bin_width, bin_width)

    counts, _ = np.histogram(vals, bins=edges)
    idx = np.argmax(counts)
    modal = (edges[idx] + edges[idx + 1]) / 2
    median_depth = np.median(vals)

    # Safeguard against low-water oscillation modes.
    if modal < median_depth:
        above_median_vals = vals[vals >= median_depth]

        edges = np.arange(
            above_median_vals.min(),
            above_median_vals.max() + bin_width,
            bin_width,
        )

        counts, _ = np.histogram(above_median_vals, bins=edges)
        idx = np.argmax(counts)
        modal = (edges[idx] + edges[idx + 1]) / 2

    return float(modal)


records = []
for wetland_id, sp in est_spills.groupby("wetland_id"):
    w = well_data[
        (well_data["wetland_id"] == wetland_id) & (well_data["flag"] == 0)
    ]
    if w.empty:
        continue

    offset = sp["well_to_min_delineated"].iloc[0]
    modal_depth = compute_modal_depth(w["well_depth_m"] + offset)

    records.append(
        {
            "wetland_id": wetland_id,
            "modal_depth_delineated": modal_depth,
            "delineated_spill_h_min": sp["delineated_spill_h_min"].iloc[0],
        }
    )

modal_df = pd.DataFrame(records).dropna(
    subset=["delineated_spill_h_min", "modal_depth_delineated"]
)

print(modal_df['delineated_spill_h_min'].mean())
print(modal_df['modal_depth_delineated'].mean())
print(modal_df['modal_depth_delineated'].std())
# %% 3.0 2x2 panel plot (all wetlands + connectivity classes)

modal_with_connect = modal_df.merge(
    connect[["wetland_id", "connectivity"]],
    on="wetland_id",
    how="left",
)

connectivity_config = {
    "first order": {"color": "#6C5B7B", "label": "Ditch connected"},
    "giw": {"color": "#1B7F79", "label": "Unditched"},
    "flow-through": {"color": "#C46A1A", "label": "Flow-through connected"},
}


plot_df = modal_with_connect.copy()
plot_df["connectivity_key"] = plot_df["connectivity"].astype(str).str.strip().str.lower()

global_vals = np.concatenate(
    [
        modal_df["delineated_spill_h_min"].to_numpy(dtype=float),
        modal_df["modal_depth_delineated"].to_numpy(dtype=float),
    ]
)

lim = (global_vals.min() - 0.05, global_vals.max() + 0.05)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

panel_order = [
    ("all", "All wetlands", None),
    ("class", "first order", connectivity_config["first order"]),
    ("class", "giw", connectivity_config["giw"]),
    ("class", "flow-through", connectivity_config["flow-through"]),
]

for ax, (panel_kind, key, cfg) in zip(axes, panel_order):
    if panel_kind == "all":
        class_df = modal_df.copy()
        color = "black"
        panel_label = "All wetlands"
    else:
        class_df = plot_df[plot_df["connectivity_key"] == key]
        color = cfg["color"]
        panel_label = cfg["label"]

    x = class_df["delineated_spill_h_min"].to_numpy(dtype=float)
    y = class_df["modal_depth_delineated"].to_numpy(dtype=float)

    ax.scatter(
        x,
        y,
        color=color,
        marker='.',
        s=150,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
        label=panel_label,
    )

    if panel_kind == "class":
        for xi, yi, wid in zip(x, y, class_df["wetland_id"].values):
            ax.annotate(
                str(wid),
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    mask = np.isfinite(x) & np.isfinite(y)

    m, b = np.polyfit(x[mask], y[mask], 1)
    y_pred = m * x[mask] + b
    ss_res = np.sum((y[mask] - y_pred) ** 2)
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    trend_label = (
        f"Trend (slope={m:.2f}, r^2={r2:.3f})"
        if np.isfinite(r2)
        else f"Trend (slope={m:.2f})"
    )
    ax.plot(x_line, m * x_line + b, color=color, linewidth=2, label=trend_label)

    ax.plot(lim, lim, "k--", linewidth=1.5, label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11)

fig.supxlabel("Delineated spill depth (cm)", fontsize=18)
fig.supylabel("Modal water depth (cm)", fontsize=18)

fig.subplots_adjust(wspace=0.08, hspace=0.10)
plt.tight_layout(rect=(0.04, 0.04, 1, 1))

plt.show()

# %% 4.0 Box Plot to Illustrate Modal Water Depths

connect_order = [
    ("giw", "Unditched"),
    ("first order", "Ditch connected"),
    ("flow-through", "Flow-through connected"),
]

series = []
series_labels = []
colors = []
hatches = []
positions = []

method_config = [
    ("modal_depth_delineated", "modal depth", ""),
    ("delineated_spill_h_min", "geomorphic spill", "////"),
]

for i, (conn_key, conn_label) in enumerate(connect_order):
    d = plot_df.loc[plot_df["connectivity_key"] == conn_key]
    base = i * 3 + 1

    for j, (col, method_label, hatch) in enumerate(method_config):
        vals = d[col].dropna().to_numpy() * 100
        series.append(vals)
        series_labels.append(f"{conn_label} | {method_label}")
        colors.append(connectivity_config[conn_key]["color"])
        hatches.append(hatch)
        positions.append(base + j)

fig, ax = plt.subplots(figsize=(8, 6))

bp = ax.boxplot(
    series,
    positions=positions,
    widths=0.8,
    patch_artist=True,
    showfliers=False,
)

for box, c, hatch in zip(bp["boxes"], colors, hatches):
    box.set_facecolor(c)
    box.set_alpha(0.65)
    box.set_edgecolor("black")
    box.set_hatch(hatch)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

for pos, vals, c in zip(positions, series, colors):
    x_jitter = np.random.normal(loc=pos, scale=0.05, size=len(vals))
    ax.scatter(
        x_jitter,
        vals,
        color=c,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.8,
        s=45,
        zorder=3,
    )

ax.set_xticks([1.5, 4.5, 7.5])
ax.set_xticklabels([label for _, label in connect_order])
ax.set_ylabel("Depth (cm)", fontsize=12)
ax.grid(axis="y", alpha=0.25)

legend_elements = [
    Patch(facecolor="lightgray", alpha=0.65, edgecolor="black", hatch="", label="modal depth"),
    Patch(facecolor="lightgray", alpha=0.65, edgecolor="black", hatch="////", label="geomorphic spill"),
    Patch(facecolor="#1B7F79", alpha=0.65, edgecolor="black", label="Unditched"),
    Patch(facecolor="#6C5B7B", alpha=0.65, edgecolor="black", label="Ditch connected"),
    Patch(facecolor="#C46A1A", alpha=0.65, edgecolor="black", label="Flow-through connected"),
]
ax.legend(handles=legend_elements, loc="best", fontsize=11)

plt.tight_layout()
plt.show()


# %% 4.1 Print means and standard deviations associated with the boxplot series

for label, vals in zip(series_labels, series):
    mean = np.nanmean(vals)
    std = np.nanstd(vals, ddof=1)
    print(f"{label}: mean={mean:.3f}, sd={std:.3f}, n={len(vals)}")

# Ditched diff = 15 cm
# Unditched diff = 10 cm
# Flow-through diff = 19 cm

# %% 4.2 Run an ANOVA to see if modal water depths and estimated spills are different between classes

anova_vars = [
    ("modal_depth_delineated", "modal depth"),
    ("delineated_spill_h_min", "geomorphic spill"),
]

for col, name in anova_vars:
    groups = []

    for key, cfg in connectivity_config.items():
        vals = plot_df.loc[plot_df["connectivity_key"] == key, col].dropna().to_numpy()

        groups.append(vals)
        print(f"{name} | {cfg['label']}: n={len(vals)}, mean={np.mean(vals) * 100:.2f} cm")

    f_stat, p_val = f_oneway(*groups)
    print(f"{name} ANOVA: F={f_stat:.3f}, p={p_val:.4g}")

    print()

# %% 
