# Incorporates the well stage timeseries with basin attributes to model basin dynamics

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import datetime
import os

from basin_attributes import WetlandBasin, ClippedDEM, WellPoint


@dataclass
class WellStageTimeseries:
    """Class to handle and process well stage (water level) timeseries data."""
    well_id: str
    timeseries_data: pd.DataFrame  # DataFrame with datetime index and water_level column
    
    
    @classmethod
    def from_csv(cls, file_path: str, well_id: str, date_column: str = 'date', 
                water_level_column: str = 'water_level', well_id_column: str = 'well_id'):
        """
        Create a WellStageTimeseries from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            well_id: ID of the well to extract data for
            date_column: Name of the column containing date information
            water_level_column: Name of the column containing water level data
            well_id_column: Name of the column containing well identifiers
        """
        df = pd.read_csv(file_path)
        print(df.columns)
        if well_id_column in df.columns:
            df = df[df[well_id_column] == well_id]
        
        df[date_column] = pd.to_datetime(df[date_column])
        

        df = df.set_index(df[date_column])
        
        if water_level_column != 'water_level':
            df = df.rename(columns={water_level_column: 'water_level'})
        
        # Keep only necessary column and aggregate to daily values
        df = df[['water_level']].resample('D').mean()
        
        return cls(well_id=well_id, timeseries_data=df)
    
    def plot(self):
        """Plot the water level time series."""
        fig, ax = plt.subplots(figsize=(12, 6))
        self.timeseries_data['water_level'].plot(ax=ax)
        ax.set_ylabel(f'Water Elevation, m)')
        ax.set_title(f'Well {self.well_id} Water Elevation Time Series')
        ax.grid(True)
        plt.show()


@dataclass
class BasinDynamics:
    """
    Class to model the dynamic behavior of a wetland basin based on water level data.
    Combines basin topography with well water level time series to model inundation.
    """
    basin: WetlandBasin
    well_stage: WellStageTimeseries
    well_to_dem_offset: float = 0.0  #NOTE: Adding this to illustrate sensitivity of wetland dynamics to well elevations
    
    @cached_property
    def well_point(self) -> WellPoint:
        """Get the well point from the basin."""
        return self.basin.establish_well_point(self.basin.well_point_info)
    
    def calculate_inundation_map(self, water_elevation: float) -> np.ndarray:
        """
        Calculate the inundation map for a given water elevation. 
        """
        # Get the DEM
        dem = self.basin.clipped_dem.dem
        
        # 1=inundated, 0=dry, np.nan=out of basin
        inundation_map = np.zeros_like(dem, dtype=np.uint8)
        inundation_map[~np.isnan(dem) & (dem <= water_elevation)] = 1
        inundation_map = np.where(np.isnan(dem), np.nan, inundation_map)  # Keep NaNs as NaNs
        
        return inundation_map
    
    def calculate_inundation_timeseries(self) -> Dict[pd.Timestamp, np.ndarray]:
        """
        Calculate inundation maps for the entire time series.
        """
        # Get well data
        well_data = self.well_stage.timeseries_data
        
        # Initialize results dictionary
        inundation_maps = {}
        
        # Calculate inundation map for each date
        for date, row in well_data.iterrows():
            # Convert water level in well to water surface elevation in DEM datum
            water_level = row['water_level']
            water_elevation = self.well_point.elevation_dem + water_level + self.well_to_dem_offset
            # Calculate inundation map
            inundation_map = self.calculate_inundation_map(water_elevation)
            
            # Store in dictionary
            inundation_maps[date] = inundation_map
            
        return inundation_maps
    
    def calculate_inundation_summary(self, inundation_maps: Dict[pd.Timestamp, np.ndarray] = None) -> np.ndarray:
        """
        Calculate a summary of inundation frequency across the time series.
        Takes a dictionary of inundation maps with pd.Timestamp, np.ndarray

        Returns an array with the fraction of time each cell was inundated.
        """
        # Calculate inundation maps if not provided
        if inundation_maps is None:
            inundation_maps = self.calculate_inundation_timeseries()
            
        # Stack inundation maps
        stack = np.stack(list(inundation_maps.values()))
        
        # Calculate frequency of inundation
        inundation_frequency = np.sum(stack, axis=0) / stack.shape[0]
        
        return inundation_frequency
    
    def calculate_inundated_area_timeseries(self, inundation_maps: Dict[pd.Timestamp, np.ndarray] = None) -> pd.Series:
        """
        Calculate the inundated area for each timestep.
        """
        # Calculate inundation maps if not provided
        if inundation_maps is None:
            inundation_maps = self.calculate_inundation_timeseries()
            
        # NOTE: This assumes the transform in clipped_dem gives us the cell size
        cell_size = abs(self.basin.clipped_dem.transform.a)  # Cell width in meters
        print(f'Cell size from DEM meta: {cell_size} m')
        cell_area = cell_size * cell_size  # Cell area in sq meters
        
        # Calculate area for each timestep
        areas = {}
        for date, inundation_map in inundation_maps.items():
            inundated_cells = np.sum(inundation_map)
            areas[date] = inundated_cells * cell_area
            
        return pd.Series(areas)
    
    def visualize_inundation(self, date: pd.Timestamp = None, water_elevation: float = None):
        """
        Visualize inundation for a specific date (pd.datetime) or water elevation (float).
        """
        if date is None and water_elevation is None:
            raise ValueError("Either date or water_elevation must be provided")
            
        # Calculate water elevation from date if needed
        if water_elevation is None:
            water_level = self.well_stage.timeseries_data.loc[date, 'water_level']
            water_elevation = self.well_point.elevation_dem - self.well_point.depth + water_level
            
        # Calculate inundation map
        inundation_map = self.calculate_inundation_map(water_elevation)
        
        # Get DEM for visualization
        dem = self.basin.clipped_dem.dem
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot DEM with hill shade effect
        plt.imshow(dem, cmap='gray', interpolation='nearest')
        plt.colorbar(label='Elevation (m)')
        
        # Add inundation overlay
        plt.imshow(inundation_map, cmap='Blues', alpha=0.4)
        
        # NOTE: Add the well point to the basin visualization.
        
        # Add title
        if date is not None:
            plt.title(f"Inundation Map for {date.date()} - Water Elevation: {water_elevation:.2f}m")
        else:
            plt.title(f"Inundation Map for Water Elevation: {water_elevation:.2f}m")
            
        plt.show()
    

    