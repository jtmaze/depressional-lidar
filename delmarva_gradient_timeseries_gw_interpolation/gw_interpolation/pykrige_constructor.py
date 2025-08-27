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
            end_obs: datetime, 
            well_type: str = None
        ):

        # Filter by well type if specified
        if well_type is None:
            filtered_df = df
        elif well_type == 'UW':
            filtered_df = df[df['SiteID'].str.contains('UW')]
        elif well_type == 'UW_CH':
            filtered_df = df[df['SiteID'].str.contains('UW') | df['SiteID'].str.contains('CH')]
        else:
            raise ValueError(f"Invalid well_type: {well_type}. Supported types are None, 'UW', or 'UW_CH'")

        # Clean and process the filtered dataframe
        # 1) Filter timeframe of interest
        df_clean = filtered_df[(filtered_df['Date'] >= begin_obs) & 
                (filtered_df['Date'] <= end_obs)].copy()
        # 2) Drop flagged observations
        df_clean = df_clean[df_clean['Flag'] == 0]
        # 3) Drop NaNs
        df_clean = df_clean.dropna()

        # 4) Get summary information
        self.unique_wells = df_clean['SiteID'].unique()
        self.well_count = len(self.unique_wells)
        self.wl_mean = df_clean.groupby('SiteID').agg({
            'waterLevel': 'mean'
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
            well_coords[['SiteID', lat_col, lon_col, 'Elevation']],
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

        wl_points['relative_wl'] = wl_points['waterLevel'] + wl_points['Elevation']

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
        self.z_samples = wl_points['relative_wl'].to_numpy() / 100 
        print(self.z_samples)
        # Make bounding box to run interpolation over
        # NOTE: grid resolution is an open question right now
        if boundary.crs != 'EPSG:26917':
            boundary.to_crs('EPSG:26917', inplace=True)
            
        self.boundary = boundary
        bbox = boundary.total_bounds
        minx, miny, maxx, maxy = bbox

        self.x_grid = np.linspace(minx, maxx, 1000)
        self.y_grid = np.linspace(miny, maxy, 1000)

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
        ) # NOTE: Could implement the 'mask' option later to only evaluate certian cells, think sigma_squared.

        self.z_result = z_result
        self.sigma_squared = sigma_squared 

    def plot_interpolation_result_contour(
            self
    ):
        
        fig, ax = plt.subplots(figsize=(7, 7))
        cf = ax.contourf(self.x_grid, self.y_grid, self.z_result, cmap='viridis')
        plt.colorbar(cf, ax=ax, label='Interpolated GW (meters relative to lowest wetland bottom)')
        ax.scatter(self.x_samples, self.y_samples, color='red', s=50)
        self.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
        plt.show()

    def plot_interpolation_result(
            self
    ):
        
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(self.z_result, extent=[self.x_grid.min(), self.x_grid.max(), 
                                             self.y_grid.min(), self.y_grid.max()], 
                       origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label='Interpolated GW (meters relative to lowest wetland bottom)')
        ax.scatter(self.x_samples, self.y_samples, color='red', s=50)
        self.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
        plt.show()

    def plot_sigma_squared(self):
        
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(self.sigma_squared**(0.5), 
                       extent=[self.x_grid.min(), self.x_grid.max(), 
                              self.y_grid.min(), self.y_grid.max()], 
                       origin='lower', cmap='Reds', aspect='auto')
        plt.colorbar(im, ax=ax, label='Kriging Uncertainty (m)')
        ax.scatter(self.x_samples, self.y_samples, color='blue', s=50)
        self.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
        ax.set_title('Kriging Uncertainty (prediction variance)')
        plt.show()


    def plot_masked_result(
            self,
            sigma_squared_threshold: float, 
    ):
        sigma_squared_mask = (self.sigma_squared**(0.5) <= sigma_squared_threshold)
        masked_result = np.where(sigma_squared_mask, self.z_result, np.nan)

        fig, ax = plt.subplots(figsize=(7, 7))

        im = ax.imshow(masked_result, extent=[self.x_grid.min(), self.x_grid.max(), 
                                         self.y_grid.min(), self.y_grid.max()], 
                   origin='lower', cmap='viridis', aspect='auto')
        # Fix missing closing quote
        plt.colorbar(im, ax=ax, label=f'Interpolated GW (meters relative to lowest wetland bottom)')
        ax.scatter(self.x_samples, self.y_samples, color='red', s=50)
        self.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
        ax.set_title(f'Kriging Result (masked where uncertainty > {sigma_squared_threshold}m)')
        plt.show()


