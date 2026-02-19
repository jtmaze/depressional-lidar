# NOTE: This shim facilites imports by bringing the root directory higher
import sys
PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataclasses import dataclass
from functools import cached_property
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
from shapely.geometry import Point, Polygon


from wetland_utilities.basin_attributes import WetlandBasin, ClippedDEM, WellPoint


@dataclass
class WellStageTimeseries:
    """Class to handle and process well stage (water level) timeseries data."""
    well_id: str
    timeseries_data: pd.DataFrame  # DataFrame with datetime index and water_level column
    basin: WetlandBasin
    
    @classmethod
    def from_csv(cls, file_path: str, well_id: str, basin: WetlandBasin, date_column: str = 'date', 
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
        df = pd.read_csv(file_path, dtype={'flag': 'Int64'})
        
        if well_id_column in df.columns:
            df = df[df[well_id_column] == well_id]

        if water_level_column != 'water_level':
            df = df.rename(columns={water_level_column: 'water_level'})
        if date_column != 'date':
            df = df.rename(columns={date_column: 'date'})
        
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.set_index(df['date'])
        # Keep only necessary columns and aggregate to daily values
        df = df[['water_level', 'flag']].resample('D').agg({
            'water_level': 'mean',
            'flag': lambda x: x.mode().iloc[0] if not x.mode().empty else 0
        }).round({'flag': 0}).astype({'flag': int})

        return cls(well_id=well_id, timeseries_data=df, basin=basin)
    
    def plot(self):
        """Plot the water level time series."""
        
        # Basin low elevation is important for understanding well timeseries
        basin_low_elevation = self.basin.deepest_point.elevation
        well_point_elevation = self.basin.well_point.elevation_dem
        diff = basin_low_elevation - well_point_elevation
        
        fig, ax = plt.subplots(figsize=(12, 6))
        self.timeseries_data['water_level'].plot(ax=ax)
        ax.set_ylabel(f'Water Depth at Well (m)')
        ax.set_title(f'Well {self.well_id} Stage')
        ax.grid(True)
        plt.axhline(y=0, color='blue', linestyle='--', label='Well Depth')
        plt.axhline(y=diff, color='red', linestyle='--', label='Depth of Basin Low')
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
        return self.basin.well_point
    
    def calculate_inundation_map(self, water_elevation: float) -> np.ndarray:
        """
        Calculate the inundation map for a given water elevation. 
        """
        # Get the DEM
        dem = self.basin.clipped_dem.dem
        
        # 1=inundated, 0=dry, np.nan=out of basin
        inundation_map = np.zeros_like(dem, dtype=np.float16)

        inundation_map[~np.isnan(dem) & (dem <= water_elevation)] = 1.0
        inundation_map = np.where(np.isnan(dem), np.nan, inundation_map)  # Keep NaNs as NaNs
        
        return inundation_map

    def calculate_tai_map(self, water_elevation: float, max_depth: float, min_depth: float):

        dem = self.basin.clipped_dem.dem
        depth = water_elevation - dem

        tai_map = np.zeros_like(dem, dtype=np.float16)
        tai_map = np.where((depth < max_depth) & (depth > min_depth), 1, tai_map)
        tai_map = np.where(np.isnan(dem), np.nan, tai_map)  # Keep NaNs outside the basin boundary as NaNs

        return tai_map

    def visualize_single_inundation_map(self, date: pd.Timestamp = None, water_elevation: float = None):
        """
        Visualize inundation for a specific date (pd.datetime) or water elevation (float).
        """
        if date is None and water_elevation is None:
            raise ValueError("Either date or water_elevation must be provided")
            
        if water_elevation is None:
            water_level = self.well_stage.timeseries_data.loc[date, 'water_level']
            water_elevation = self.well_point.elevation_dem + water_level
        
        # Get inundation map, DEM, and well point for visualization
        inundation_map = self.calculate_inundation_map(water_elevation)
        dem = self.basin.clipped_dem.dem
        well_point = self.well_point
        well_point_x = well_point.location.x.values[0]
        well_point_y = well_point.location.y.values[0]
        row, col = rio.transform.rowcol(
            self.basin.clipped_dem.transform, 
            well_point_x, 
            well_point_y
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(dem, cmap='gray', interpolation='nearest')
        plt.colorbar(label='Elevation (m)')
        plt.imshow(inundation_map, cmap='Blues', alpha=0.4)
        ax.scatter(col, row, color='red', 
                   marker='x', s=100, label=f'Well Location @{well_point.elevation_dem:.2f}m')
        plt.legend()

        if date is not None:
            plt.title(f"Inundation Map for {date.date()} - Water Elevation: {water_elevation:.2f}m")
        else:
            plt.title(f"Inundation Map for Water Elevation: {water_elevation:.2f}m")
            
        plt.show()

    def visualize_single_tai_map(
            self, 
            date: pd.Timestamp = None, 
            water_elevation: float = None,
            max_depth: float = None,
            min_depth: float = None
    ):

        if date is None and water_elevation is None:
            raise ValueError("Either date or water_elevation must be provided")

        if water_elevation is None:
            water_level = self.well_stage.timeseries_data.loc[date, 'water_level']
            water_elevation = self.well_point.elevation_dem + water_level

        # Get TAI map, DEM, and well point for visualization
        tai_map = self.calculate_tai_map(water_elevation, max_depth, min_depth)
        dem = self.basin.clipped_dem.dem
        well_point = self.well_point
        well_point_x = well_point.location.x.values[0]
        well_point_y = well_point.location.y.values[0]

        row, col = rio.transform.rowcol(
            self.basin.clipped_dem.transform,
            well_point_x,
            well_point_y
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(dem, cmap='gray', interpolation='nearest')
        plt.colorbar(label='Elevation (m)')
        plt.imshow(tai_map, cmap='Oranges', alpha=0.2)
        ax.scatter(col, row, color='red',
                   marker='x', s=100, label=f'Well Location @{well_point.elevation_dem:.2f}m')
        plt.legend()

        if date is not None:
            plt.title(
                f"TAI Map for {date.date()} - Water Elevation: {water_elevation:.2f}m\n"
                f"Depth at {max_depth} to {min_depth}m"
            )
        else:
            plt.title(
                f"TAI Map for Water Elevation: {water_elevation:.2f}m\n"
                f"Depth at {max_depth} to {min_depth}m"
            )
        plt.show()

    def calculate_inundation_stacks(self) -> Dict[pd.Timestamp, np.ndarray]:
        """
        Calculate inundation maps for the entire time series.
        """
        # Get well data
        well_data = self.well_stage.timeseries_data
        
        # Initialize results dictionary
        inundation_stacks = {}
        
        # Calculate inundation map for each date
        for date, row in well_data.iterrows():
            # Convert water level in well to water surface elevation in DEM datum
            water_level = row['water_level']
            water_elevation = self.well_point.elevation_dem + water_level + self.well_to_dem_offset
            # Calculate inundation map
            inundation_map = self.calculate_inundation_map(water_elevation)
            
            # Store in dictionary
            inundation_stacks[date] = inundation_map
            
        return inundation_stacks

    def calculate_tai_stacks(self, max_depth: float, min_depth: float) -> Dict[pd.Timestamp, np.ndarray]:

        well_data = self.well_stage.timeseries_data

        tai_stacks = {}

        for date, row in well_data.iterrows():
            water_level = row['water_level']
            water_elevation = self.well_point.elevation_dem + water_level + self.well_to_dem_offset
            tai_map = self.calculate_tai_map(water_elevation, max_depth, min_depth)
            tai_stacks[date] = tai_map

        return tai_stacks

    def aggregate_inundation_stacks(self, inundation_stacks: Dict[pd.Timestamp, np.ndarray] = None) -> np.ndarray:
        """
        Calculate a summary of inundation frequency across the time series.
        Takes a dictionary of inundation maps with pd.Timestamp, np.ndarray

        Returns an array with the fraction of time each cell was inundated.
        """
        # Calculate inundation maps if not provided
        if inundation_stacks is None:
            inundation_stacks = self.calculate_inundation_stacks()

        # Stack inundation maps
        stack = np.stack(list(inundation_stacks.values()))

        # Calculate frequency of inundation
        inundation_frequency = np.sum(stack, axis=0, dtype=np.float16) / stack.shape[0]
        
        return inundation_frequency
    
    def aggregate_tai_stacks(
            self, 
            max_depth: float, 
            min_depth: float, 
            tai_stacks: Dict[pd.Timestamp, np.ndarray] = None
        ) -> np.ndarray:

        if tai_stacks is None:
            tai_stacks = self.calculate_tai_stacks(max_depth, min_depth)

        # Stack TAI maps 
        stack = np.stack(list(tai_stacks.values()))
        tai_frequency = np.nansum(stack, axis=0, dtype=np.float16) / stack.shape[0]

        return tai_frequency
    
    def calculate_inundated_area_timeseries(
            self, 
            inundation_stacks: Dict[pd.Timestamp, np.ndarray] = None,
        ) -> pd.Series:

        """
        Calculate the inundated area for each timestep.
        """
        # Calculate inundation maps if not provided
        if inundation_stacks is None:
            inundation_stacks = self.calculate_inundation_stacks()

        # NOTE: This assumes the transform in clipped_dem gives us the cell size
        cell_size = abs(self.basin.clipped_dem.transform.a)  # Cell width in meters
        print(f'Cell size from DEM meta: {cell_size} m')
        cell_area = cell_size * cell_size  # Cell area in sq meters
        
        # Calculate area for each timestep
        areas = {}
        for date, inundation_map in inundation_stacks.items():
            inundation_map = inundation_map.astype(np.float32)
            inundation_map = np.where(np.isinf(inundation_map), 0, inundation_map)

            inundated_cells = np.nansum(inundation_map)
            areas[date] = inundated_cells * cell_area

        return pd.Series(areas)
    
    def calculate_tai_timeseries(
            self, 
            max_depth: float, 
            min_depth: float, 
            tai_stacks: Dict[pd.Timestamp, np.ndarray] = None
        ) -> pd.Series:

        if tai_stacks is None:
            tai_stacks = self.calculate_tai_stacks(max_depth=max_depth, min_depth=min_depth)

        cell_size = abs(self.basin.clipped_dem.transform.a)
        print(f'Cell size from DEM meta: {cell_size} m')
        cell_area = cell_size * cell_size

        tai_areas = {}
        for date, tai_map in tai_stacks.items():
            tai_map = tai_map.astype(np.float64)
            tai_cells = np.nansum(tai_map)
            tai_areas[date] = tai_cells * cell_area

        return pd.Series(tai_areas)
    
    def plot_inundated_area_timeseries(self, area_timeseries: pd.Series = None):
        """
        Plot the inundated area timeseries.
        """
        if area_timeseries is None:
            area_timeseries = self.calculate_inundated_area_timeseries()

        plt.figure(figsize=(12, 6))
        plt.plot(area_timeseries.index, area_timeseries.values, marker='o')
        plt.title(f"{self.well_stage.well_id} Inundated Area Timeseries")
        plt.xlabel("Date")
        plt.ylabel("Inundated Area (m²)")
        plt.grid()
        plt.show()

    def plot_tai_area_timeseries(self, tai_timeseries: pd.Series = None, max_depth: float = None, min_depth: float = None):
        """
        Plot the TAI area timeseries.
        """
        if tai_timeseries is None:
            tai_timeseries = self.calculate_tai_timeseries(max_depth, min_depth)

        plt.figure(figsize=(12, 6))
        plt.plot(tai_timeseries.index, tai_timeseries.values, color='Orange', marker='o')
        plt.title(
            f"{self.well_stage.well_id} TAI Area Timeseries\n"
            f"Depth {min_depth} to {max_depth} m"
        )
        plt.xlabel("Date")
        plt.ylabel("TAI Area (m²)")
        plt.grid()
        plt.show()

    def plot_inundated_area_histogram(self, area_timeseries: pd.Series = None):
        """
        Plot a histogram of inundated areas.
        """
        if area_timeseries is None:
            area_timeseries = self.calculate_inundated_area_timeseries()
        # Debug: Check for infinite values
        infinite_mask = np.isinf(area_timeseries.values)
        if infinite_mask.any():
            print(f"Found {infinite_mask.sum()} infinite values")
            print(f"Dates with infinity: {area_timeseries[infinite_mask].index.tolist()}")

        plt.figure(figsize=(10, 6))
        plt.hist(area_timeseries.values, bins=40, color='blue', alpha=0.7, edgecolor='black')
        plt.title(f"{self.well_stage.well_id} Inundated Area Histogram")
        plt.xlabel("Inundated Area (m²)")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()

    def plot_tai_area_histogram(
            self, 
            tai_timeseries: pd.Series = None, 
            max_depth: float = None, 
            min_depth: float = None, 
            as_pct: bool = False
        ):
        """
        Plot a histogram of TAI areas.
        """
        if tai_timeseries is None:
            tai_timeseries = self.calculate_tai_timeseries(max_depth, min_depth)

        plt.figure(figsize=(10, 6))
        if as_pct:
            total_days = len(tai_timeseries)
            weights = np.ones_like(tai_timeseries.values) / total_days * 100
            area, _ = self.basin.calculate_hypsometry(method="total_cdf")
            max_area = max(area)
            plt.hist((tai_timeseries.values / max_area) * 100, bins=40, color='orange', alpha=0.7, edgecolor='black', weights=weights)
            plt.xlabel("TAI Area (Percent of Total Area)")
            plt.ylabel("Percent Observations")
        else:
            plt.hist(tai_timeseries.values, bins=40, color='orange', alpha=0.7, edgecolor='black')
            plt.xlabel("TAI Area (m²)")
            plt.ylabel("Days")

        plt.title(f"{self.well_stage.well_id} TAI Area Histogram\nDepth {min_depth} to {max_depth} m")
        plt.grid()
        plt.show()

    def map_tai_stacks(
            self, 
            tai_frequency: np.array = None, 
            max_depth: float = None, 
            min_depth: float = None,
            show_basin_footprint: bool = False,
            cbar_min: float = None,
            cbar_max: float = None
        ):
        """
        Map the TAI stacks for a given depth range.
        """
        if tai_frequency is None:
            if max_depth is None or min_depth is None:
                raise ValueError("Either tai_frequency array or both max_depth and min_depth must be provided")
            tai_frequency = self.aggregate_tai_stacks(max_depth, min_depth)

        dem = self.basin.clipped_dem.dem
        tai_percent = tai_frequency * 100
        tai_percent = np.where(np.isnan(dem), np.nan, tai_percent)  # Keep NaNs outside the basin boundary as NaNs
        well_point = self.well_point
        well_point_x = well_point.location.x.values[0]
        well_point_y = well_point.location.y.values[0]

        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Show TAI frequency (values from 0-1 representing frequency of being in TAI zone)
        im = show(tai_percent, ax=ax, cmap='RdYlBu_r', alpha=0.8, 
                transform=self.basin.clipped_dem.transform, 
                vmin=0, vmax=np.nanmax(tai_percent))

        # Add colorbar
        cbar = plt.colorbar(im.images[0], ax=ax, shrink=0.8)
        cbar.set_label('TAI Frequency (0-100%) of Days', rotation=270, labelpad=20)

        if show_basin_footprint:
            footprint = self.basin.footprint
            footprint.boundary.plot(ax=ax, color='green', linewidth=2, alpha=0.8, label='Basin Footprint')

        # Plot well point
        ax.scatter(well_point_x, well_point_y, color='green', 
                marker='x', s=100, linewidths=3,
                label=f'Well Location @{well_point.elevation_dem:.2f}m')
        
        # Add legend and labels
        ax.legend(loc='upper right')
        ax.set_xlabel('(m)')
        ax.set_ylabel('(m)')
        
        ax.set_title(f'TAI Frequency Map\n Defined by depth range: {min_depth:.2f}m to {max_depth:.2f}m')

        plt.tight_layout()
        plt.show()

    def map_inundation_stacks(
            self, 
            inundation_frequency: np.array = None, 
            show_basin_footprint: bool = False,
            cbar_min: float = 0,
            cbar_max: float = None
    ):
        """
        Map the inundation stacks.
        # TODO: Add an argument to make a constant color scale bar
        """
        if inundation_frequency is None:
            inundation_frequency = self.aggregate_inundation_stacks()

        dem = self.basin.clipped_dem.dem
        inundation_percent = inundation_frequency * 100
        inundation_percent = np.where(np.isnan(dem), np.nan, inundation_percent)  # Keep NaNs outside the basin boundary as NaNs
    
        if cbar_max is None:
            vmax = np.nanmax(inundation_percent)
        else:
            vmax = cbar_max

        # Add the well point
        well_point = self.well_point
        well_point_x = well_point.location.x.values[0]
        well_point_y = well_point.location.y.values[0]

        
        fig, ax = plt.subplots(figsize=(12, 8))

        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#8B4513', '#FFFFFF', '#0000FF']  # Brown, White, Blue
        custom_cmap = LinearSegmentedColormap.from_list('brown_white_blue', colors, N=256)

        if show_basin_footprint:
            footprint = self.basin.footprint
            footprint.boundary.plot(ax=ax, color='green', linewidth=2, alpha=0.8, label='Basin Footprint')
        
        # Show inundation frequency (values from 0-1 representing frequency of being inundated)
        im = show(inundation_percent, ax=ax, cmap=custom_cmap, alpha=1,
                transform=self.basin.clipped_dem.transform,
                vmin=cbar_min, vmax=vmax)

        # Add colorbar
        cbar = plt.colorbar(im.images[0], ax=ax, shrink=0.8)
        cbar.set_label('Inundation Frequency % of Days', rotation=270, labelpad=20, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Plot well point
        ax.scatter(well_point_x, well_point_y, color='limegreen', 
                marker='x', s=400, linewidths=7,
                label=f'Well Location')
        
        # Add legend and labels
        #ax.legend(loc='upper right', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('')

        plt.show()



        