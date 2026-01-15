# %% 1.0 File paths and libraries

import pandas as pd

stage_dir = "D:/depressional_lidar/data/bradford/in_data/stage_data/"
input_path = stage_dir + "bradford_stage_Winter2025_offsets_tracked.csv"
output_path = stage_dir + "bradford_daily_stage_Winter2025.csv"

# %% 2.0 Group data to daily and make the dataframe simpler. 

df = pd.read_csv(input_path).drop(
    columns=['head_m', 'offset_value', 'offset_version', 'notes']
)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

daily_df = df.groupby(['date', 'well_id']).agg({
    'well_depth_m': 'mean',
    'flag': 'max'
}).reset_index()


# %% 3.0 Write the output

daily_df.to_csv(output_path, index=False)


# %%
