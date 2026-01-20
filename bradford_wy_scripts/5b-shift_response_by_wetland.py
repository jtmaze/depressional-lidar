# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt

lai_buffer_dist = 150
data_dir = "D:/depressional_lidar/data/bradford/out_data/"
shift_path = data_dir + f'/modeled_logging_stages/shift_results_LAI_{lai_buffer_dist}m.csv'

shift_data = pd.read_csv(shift_path)
plot_data = shift_data[
    (shift_data['model_type'] == 'ols') &
    (shift_data['data_set'] == 'full')
].copy()

# %% 2.0 Boxplot showing depth shifts by log_id

fig, ax = plt.subplots(figsize=(10, 5))

groups = [plot_data.loc[plot_data["log_id"] == id, "mean_depth_change"] 
          for id in plot_data["log_id"].unique()]

ax.boxplot(groups, labels=plot_data["log_id"].unique())
ax.set_xlabel("log_id")
ax.set_ylabel("Depth Change (m)")
ax.set_title("Modeled Depth Change (post-logging) by log_id")
plt.xticks(rotation=90)

ax.axhline(y=0, color='red')

plt.tight_layout()
plt.show()

# %% Boxplot showing depth_shifts by ref_id

fig, ax = plt.subplots(figsize=(10, 5))

groups = [plot_data.loc[plot_data['ref_id'] == id, "mean_depth_change"]
          for id in plot_data['ref_id'].unique()]

ax.boxplot(groups, labels=plot_data["ref_id"].unique())
ax.set_xlabel("ref_id")
ax.set_ylabel("Mean Depth Change (m)")
ax.set_title("Mean Depth Change by ref_id")
plt.xticks(rotation=90)

ax.axhline(y=0, color='red')

plt.tight_layout()
plt.show()


# %%
