import pandas as pd
from pandas import DataFrame
import geopandas as gpd
from geopandas import GeoDataFrame
from datetime import datetime

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
    ):
        
        self.x_samples = wl_points.geometry.x.to_numpy()
        self.y_samples = wl_points.geometry.y.to_numpy()
        self.z_samples = wl_points['WaterLevel'].to_numpy()



