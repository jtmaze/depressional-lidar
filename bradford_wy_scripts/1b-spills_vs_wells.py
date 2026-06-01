# %% 1.0 Libraries and file paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "D:/depressional_lidar/data/bradford/"

est_spills_path = f"{data_dir}/out_data/bradford_estimated_basin_spills_no_smooth.csv"
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
    if vals.size == 0:
        return np.nan
    if vals.min() == vals.max():
        return float(vals[0])

    edges = np.arange(vals.min(), vals.max() + bin_width, bin_width)
    if edges.size < 2:
        return float(np.median(vals))

    counts, _ = np.histogram(vals, bins=edges)
    idx = np.argmax(counts)
    modal = (edges[idx] + edges[idx + 1]) / 2
    median_depth = np.median(vals)

    # Keeps the original safeguard against low-water oscillation modes.
    if modal < median_depth:
        above_median_vals = vals[vals >= median_depth]
        if above_median_vals.size == 0:
            return float(modal)
        if above_median_vals.min() == above_median_vals.max():
            return float(above_median_vals[0])

        edges = np.arange(
            above_median_vals.min(),
            above_median_vals.max() + bin_width,
            bin_width,
        )
        if edges.size >= 2:
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


# %% 3.0 2x2 panel plot (all wetlands + connectivity classes)

modal_with_connect = modal_df.merge(
    connect[["wetland_id", "connectivity"]],
    on="wetland_id",
    how="left",
)

connectivity_config = {
    "first order": {"color": "#6C5B7B", "label": "Ditch connected"},
    "giw": {"color": "#1B7F79", "label": "Unconnected"},
    "flow-through": {"color": "#C46A1A", "label": "Flow-through connected"},
}


plot_df = modal_with_connect.dropna(
    subset=["connectivity", "delineated_spill_h_min", "modal_depth_delineated"]
).copy()
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

fig.supxlabel("Delineated spill depth (m)", fontsize=18)
fig.supylabel("Modal water depth (m)", fontsize=18)


fig.subplots_adjust(wspace=0.08, hspace=0.10)
plt.tight_layout(rect=(0.04, 0.04, 1, 1))

plt.show()


# %%
