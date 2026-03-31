# %% 1.0 Libraries and file paths

import pandas as pd
import geopandas as gpd

coords_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/bradford_soil_samples_locations.csv'
stream_gauge_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/sam_howley_streams.csv'
coords = pd.read_csv(coords_path)

soil_cors_out_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/bradford_soil_samples_locations.shp'
stream_out_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/stream_gauge_locations.shp'
# %% 2.0 Reformat the data 

gdf = gpd.GeoDataFrame(
    coords, 
    geometry=gpd.points_from_xy(coords.Long, coords.Lat),
    crs='EPSG:4326'  # WGS84
)
gdf = gdf.to_crs('EPSG:26917')

gdf.rename(
    columns={
        'Lat': 'lat_coord', 
        'Long': 'long_coord',
        'Site_ID': 'site_id'
    },
    inplace=True
)

gdf["type"] = gdf["person"].apply(
    lambda p: "soil_core" if p == "Faith" else f"{p}_soil_core"
)

gdf.drop(columns=['person'], inplace=True)

gdf["site_id"] = gdf["site_id"].fillna("Undefined")

gdf["sample_id"] = (
    gdf["site_id"]
    + "___"
    + gdf["type"]
    + "___"
    + (gdf.index + 1).astype(str)
)

# %% 3.0 Write the file to a geopandas dataframe

gdf.to_file(soil_cors_out_path) 

# %% 4.0 Read and reformat the stream gauge data to generate points

streams = pd.read_csv(stream_gauge_path)

streams["Site_ID"] = (
    streams["Site_ID"]
    .astype("string")
    .str.strip()
)

# One row per unique site/coordinate combo
stream_sites = (
    streams
    .dropna(subset=["Site_ID", "Lat", "Long"])
    .groupby(["Site_ID", "Lat", "Long"], as_index=False)
    .size()
    .rename(columns={"size": "n_records"})   # optional count of original rows
)

stream_gdf = gpd.GeoDataFrame(
    stream_sites,
    geometry=gpd.points_from_xy(stream_sites["Long"], stream_sites["Lat"]),
    crs="EPSG:4326"
)

stream_gdf.rename({'Site_ID': 'basin_id'}, inplace=True)

# %% 5.0 Write the stream data points
stream_gdf.to_file(stream_out_path)

# %% 6.0 Quick stream depth timeseries

import plotly.express as px

plot_df = streams.copy()
plot_df["Site_ID"] = plot_df["Site_ID"].astype("string").str.strip()
plot_df['Date'] = pd.to_datetime(plot_df['Date'], errors="coerce")
plot_df["depth"] = pd.to_numeric(plot_df["depth"], errors="coerce")

plot_df = (
    plot_df
    .dropna(subset=["Site_ID", 'Date', "depth"])
    .sort_values(["Site_ID", 'Date'])
)

fig = px.line(
    plot_df,
    x='Date',
    y="depth",
    color="Site_ID",
    title="Stream Depth Time Series by Site",
    labels={'Date': "Date", "depth": "Depth"}
)

fig.update_traces(mode="lines")
fig.update_layout(template="plotly_white", legend_title_text="Site_ID")
fig.show()

# %%
