import pandas as pd
from pandas import DataFrame
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pykrige as pkr

class WellsWaterLevel:
    """
    A class converting well obervations into points for pykrige opperations
    """
    def __init__(
            self, 
            df: DataFrame, 
            begin_obs: datetime, 
            end_obs: datetime):

        # 1) Filter timeframe of interest
        df_clean = df[(df['Date'] >= begin_obs) & 
                (df['Date'] <= end_obs)].copy()
        # 2) Drop flagged observations
        df_clean = df_clean[df_clean['Flag'] == 0]
        # 3) Drop NaNs
        df_clean = df_clean.dropna()

        # 4) Get summary information
        self.unique_wells = df_clean['SiteID'].unique()
        self.well_count = len(self.unique_wells)
        self.wl_mean = df_clean.groupby('SiteID').agg({
            'WaterLevel': 'mean'
        }).reset_index()

    def merge_well_coords(
        self,
        well_coords: DataFrame,
        lat_col: str,
        lon_col: str,
        epsg_code: str 
    ) -> GeoDataFrame:
        """
        Matches wl_mean summary dataframe to well coordinates making a GeoDataFrame
        """
        wl_summary = self.wl_mean
        merged = wl_summary.merge(
            well_coords[['SiteID', lat_col, lon_col]],
            on='SiteID',
            how='left'
        )
        # Create Point geometries from lat/lon
        merged['geom'] = gpd.points_from_xy(
            merged[lon_col],
            merged[lat_col],
            crs=epsg_code
        )
        wl_points = GeoDataFrame(merged, geometry='geom')

        # NOTE: Will need to modify this code when interpolating Bradford wells
        # Keeping it simple for Delmarva
        if wl_points.crs != 'EPSG:26917':
            wl_points.to_crs('EPSG:26917', inplace=True)

        return wl_points
        
class InterpolationResult:
    """
    Takes WaterLevel points as a GeoDataframe and implements Kriging
    """

    def __init__(
            self,
            wl_points: GeoDataFrame, # In projected coords
            boundary: GeoDataFrame, #
    ):
        
        self.x_samples = wl_points.geometry.x.to_numpy()
        self.y_samples = wl_points.geometry.y.to_numpy()
        self.z_samples = wl_points['WaterLevel'].to_numpy()

        # Make bounding box to run interpolation over
        # NOTE: grid resolution is an open question right now
        if boundary.crs != 'EPSG:26917':
            boundary.to_crs('EPSG:26917', inplace=True)
        bbox = boundary.total_bounds
        minx, miny, maxx, maxy = bbox

        self.x_grid = np.linspace(minx, maxx, 100)
        self.y_grid = np.linspace(miny, maxy, 100)

    def ordinary_kriging(
            self,
            variogram_model: str, 
            nlags: int,
            plot_variogram: bool
    ):
        """
        Interpolates the GW surface using ordinary kriging
        """
        ok = pkr.ok.OrdinaryKriging(
            self.x_samples,
            self.y_samples,
            self.z_samples,
            variogram_model=variogram_model,
            nlags=nlags,
            enable_plotting=plot_variogram
        )

        z_result, sigma_squared = ok.execute(
            "grid", 
            self.x_grid, 
            self.y_grid
        ) # NOTE: Could implement the 'mask' option later to only evaluate certian cells.

        self.z_result = z_result 

    def plot_interpolation_result(
            self
    ):
        
        fig, ax = plt.figure(figsize=(7, 7))
        cf = ax.contourf(self.x_grid, self.y_grid, self.z_result, cmap='viridis')
        plt.colorbar(cf, ax=ax, label='Interpolated GW')
        ax.scatter(self.x_samples, self.y_samples, s=50)
        plt.show()


