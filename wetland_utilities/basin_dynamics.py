# NOTE: This shim facilites imports by bringing the root directory higher
import sys
PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import geopandas as gpd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
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
                water_level_column: str = 'well_depth_m', well_id_column: str = 'wetland_id',
                crop_dates: tuple = None):
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

        if crop_dates is not None:
            df = df.loc[crop_dates[0]:crop_dates[1]]

        
        # Keep only necessary columns and aggregate to daily values
        df = df[['water_level', 'flag']].resample('D').agg({
            'water_level': 'mean',
            'flag': lambda x: x.mode().iloc[0] if not x.mode().empty else 0
        }).round({'flag': 0}).astype({'flag': int})

        df = df.dropna(subset='water_level')

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
        ax.set_title(f'Well {self.well_id} Well Depth')
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

    # ── Private generic engine ──────────────────────────────────────────────

    def _water_elevation_at(self, water_level: float) -> float:
        """Convert well water level to DEM-referenced water surface elevation."""
        return self.well_point.elevation_dem + water_level + self.well_to_dem_offset

    def _calculate_map_stacks(
            self,
            map_fn: Callable[[float], np.ndarray],
    ) -> Dict[pd.Timestamp, np.ndarray]:
        """Apply map_fn(water_elevation) at each timestep in the well timeseries."""
        stacks = {}
        for date, row in self.well_stage.timeseries_data.iterrows():
            water_elv = self._water_elevation_at(row['water_level'])
            stacks[date] = map_fn(water_elv)
        return stacks

    def _aggregate_stacks(
            self,
            stacks: Dict[pd.Timestamp, np.ndarray],
            method: str = "frequency",
    ) -> np.ndarray:
        """
        Collapse a stack dict into a summary array.
        method='frequency': fraction of timesteps with value > 0 (for binary maps)
        method='mean':      nanmean across timesteps (for continuous maps)
        """
        stack = np.stack(list(stacks.values())).astype(np.float32, copy=False)
        if method == "frequency":
            valid = np.isfinite(stack)
            wet = (stack > 0) & valid
            wet_count = np.sum(wet, axis=0, dtype=np.float32)
            valid_count = np.sum(valid, axis=0, dtype=np.float32)
            frequency = np.full(wet_count.shape, np.nan, dtype=np.float32)
            np.divide(wet_count, valid_count, out=frequency, where=valid_count > 0)
            return frequency
        elif method == "mean":
            return np.nanmean(stack, axis=0).astype(np.float32)
        raise ValueError(f"Unknown aggregation method: {method}")

    def _calculate_area_timeseries(
            self,
            stacks: Dict[pd.Timestamp, np.ndarray],
    ) -> pd.Series:
        """Sum of (cell values * cell_area) at each timestep."""
        cell_size = abs(self.basin.clipped_dem.transform.a)
        cell_area = cell_size * cell_size
        areas = {}
        for date, arr in stacks.items():
            arr = arr.astype(np.float32)
            arr = np.where(np.isinf(arr), 0, arr)
            areas[date] = np.nansum(arr) * cell_area
        return pd.Series(areas)

    # ── Private visualization helpers ───────────────────────────────────────

    def _get_well_xy(self) -> tuple:
        """Return (x, y) coordinates of the well in CRS space."""
        wp = self.well_point
        return wp.location.x.values[0], wp.location.y.values[0]

    def _choose_scale_bar_length(self, axis_width: float) -> float:
        """Pick a readable scale-bar length that spans about 20% of the map width."""
        if not np.isfinite(axis_width) or axis_width <= 0:
            return 0.0

        target = axis_width * 0.2
        magnitude = 10 ** np.floor(np.log10(target))
        normalized = target / magnitude

        if normalized < 1.5:
            nice_length = 1
        elif normalized < 3.5:
            nice_length = 2
        elif normalized < 7.5:
            nice_length = 5
        else:
            nice_length = 10

        return float(nice_length * magnitude)

    def _add_scale_bar(self, ax, scale_bar_length: float = None, location: str = 'lower left') -> None:
        """Add a simple distance scale bar in projected map units."""
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        axis_width = abs(x_max - x_min)
        axis_height = abs(y_max - y_min)

        if scale_bar_length is None:
            scale_bar_length = self._choose_scale_bar_length(axis_width)
        if scale_bar_length <= 0:
            return

        label = f"{scale_bar_length:g} m"
        fontprops = FontProperties(size=11, weight='bold')
        scale_bar = AnchoredSizeBar(
            ax.transData,
            scale_bar_length,
            label,
            location,
            pad=0.4,
            color='black',
            frameon=True,
            size_vertical=max(axis_height * 0.01, 1.0),
            fontproperties=fontprops,
        )
        scale_bar.patch.set_facecolor('white')
        scale_bar.patch.set_alpha(0.8)
        scale_bar.patch.set_edgecolor('none')
        ax.add_artist(scale_bar)

    def _plot_area_timeseries(self, area_ts, title, ylabel, color='blue'):
        plt.figure(figsize=(12, 6))
        plt.plot(area_ts.index, area_ts.values, marker='o', color=color)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def _plot_area_histogram(self, area_ts, title, xlabel, ylabel="Frequency", color='blue'):
        plt.figure(figsize=(10, 6))
        plt.hist(area_ts.values, bins=40, color=color, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def _map_frequency(
            self, frequency, title, cbar_label, cmap='RdYlBu_r',
            show_basin_footprint=False, vmin=None, vmax=None, as_percent=True,
    ):
        """Generic frequency/mean map visualization."""
        dem = self.basin.clipped_dem.dem
        data = frequency * 100 if as_percent else frequency
        data = np.where(np.isnan(dem), np.nan, data)
        finite_vals = data[np.isfinite(data)]
        if finite_vals.size == 0:
            raise ValueError("No finite map values available for plotting")
        if vmin is None:
            vmin = float(np.nanmin(data))
        if vmax is None:
            vmax = float(np.nanmax(data))
        if np.isclose(vmax, vmin):
            vmax = vmin + 1.0
        well_x, well_y = self._get_well_xy()

        fig, ax = plt.subplots(figsize=(12, 8))
        im = show(data, ax=ax, cmap=cmap, alpha=0.8,
                  transform=self.basin.clipped_dem.transform, vmin=vmin, vmax=vmax,
                  adjust=False)
        cbar = plt.colorbar(im.images[0], ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, rotation=270, labelpad=20)

        if show_basin_footprint and self.basin.footprint is not None:
            self.basin.footprint.boundary.plot(
                ax=ax, color='green', linewidth=2, alpha=0.8, label='Basin Footprint'
            )

        ax.scatter(well_x, well_y, color='green', marker='x', s=100, linewidths=3,
                   label=f'Well Location @{self.well_point.elevation_dem:.2f}m')
        ax.legend(loc='upper right')
        ax.set_xlabel('(m)')
        ax.set_ylabel('(m)')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    # ── Map kernels ─────────────────────────────────────────────────────────

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

    def calculate_depth_map(self, water_elevation: float) -> np.ndarray:
        """Continuous depth-to-water-table. Positive = inundated, negative = unsaturated."""
        dem = self.basin.clipped_dem.dem
        depth = water_elevation - dem
        return np.where(np.isnan(dem), np.nan, depth)

    # ── Visualization: single-timestep maps ─────────────────────────────────

    def visualize_single_inundation_map(self, date: pd.Timestamp = None, water_elevation: float = None):
        """
        Visualize inundation for a specific date (pd.datetime) or water elevation (float).
        """
        if date is None and water_elevation is None:
            raise ValueError("Either date or water_elevation must be provided")
            
        if water_elevation is None:
            water_level = self.well_stage.timeseries_data.loc[date, 'water_level']
            water_elevation = self._water_elevation_at(water_level)
        
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
            water_elevation = self._water_elevation_at(water_level)

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

    # ── Stacks / aggregation / timeseries ───────────────────────────────────

    def calculate_inundation_stacks(self) -> Dict[pd.Timestamp, np.ndarray]:
        return self._calculate_map_stacks(self.calculate_inundation_map)

    def calculate_tai_stacks(self, max_depth: float, min_depth: float) -> Dict[pd.Timestamp, np.ndarray]:
        return self._calculate_map_stacks(
            lambda we: self.calculate_tai_map(we, max_depth, min_depth)
        )

    def calculate_depth_stacks(self) -> Dict[pd.Timestamp, np.ndarray]:
        return self._calculate_map_stacks(self.calculate_depth_map)

    def aggregate_inundation_stacks(self, inundation_stacks: Dict[pd.Timestamp, np.ndarray] = None) -> np.ndarray:
        stacks = inundation_stacks or self.calculate_inundation_stacks()
        return self._aggregate_stacks(stacks, method="frequency")
    
    def aggregate_tai_stacks(
            self, 
            max_depth: float, 
            min_depth: float, 
            tai_stacks: Dict[pd.Timestamp, np.ndarray] = None
        ) -> np.ndarray:
        stacks = tai_stacks or self.calculate_tai_stacks(max_depth, min_depth)
        return self._aggregate_stacks(stacks, method="frequency")

    def aggregate_depth_stacks(self, stacks: Dict[pd.Timestamp, np.ndarray] = None) -> np.ndarray:
        stacks = stacks or self.calculate_depth_stacks()
        return self._aggregate_stacks(stacks, method="mean")
    
    def calculate_inundated_area_timeseries(
            self, 
            inundation_stacks: Dict[pd.Timestamp, np.ndarray] = None,
        ) -> pd.Series:
        stacks = inundation_stacks or self.calculate_inundation_stacks()
        return self._calculate_area_timeseries(stacks)
    
    def calculate_tai_timeseries(
            self, 
            max_depth: float, 
            min_depth: float, 
            tai_stacks: Dict[pd.Timestamp, np.ndarray] = None
        ) -> pd.Series:
        stacks = tai_stacks or self.calculate_tai_stacks(max_depth, min_depth)
        return self._calculate_area_timeseries(stacks)

    # ── Visualization: timeseries ───────────────────────────────────────────

    def plot_inundated_area_timeseries(self, area_timeseries: pd.Series = None):
        if area_timeseries is None:
            area_timeseries = self.calculate_inundated_area_timeseries()
        self._plot_area_timeseries(
            area_timeseries,
            title=f"{self.well_stage.well_id} Inundated Area Timeseries",
            ylabel="Inundated Area (m²)",
        )

    def plot_tai_area_timeseries(self, tai_timeseries: pd.Series = None, max_depth: float = None, min_depth: float = None):
        if tai_timeseries is None:
            tai_timeseries = self.calculate_tai_timeseries(max_depth, min_depth)
        self._plot_area_timeseries(
            tai_timeseries,
            title=f"{self.well_stage.well_id} TAI Area Timeseries\nDepth {min_depth} to {max_depth} m",
            ylabel="TAI Area (m²)",
            color='Orange',
        )

    # ── Visualization: histograms ──────────────────────────────────────────

    def plot_inundated_area_histogram(self, area_timeseries: pd.Series = None):
        if area_timeseries is None:
            area_timeseries = self.calculate_inundated_area_timeseries()
        self._plot_area_histogram(
            area_timeseries,
            title=f"{self.well_stage.well_id} Inundated Area Histogram",
            xlabel="Inundated Area (m²)",
        )

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

    # ── Visualization: frequency / mean maps ───────────────────────────────

    def map_tai_stacks(
            self, 
            tai_frequency: np.array = None, 
            max_depth: float = None, 
            min_depth: float = None,
            show_basin_footprint: bool = False,
            cbar_min: float = None,
            cbar_max: float = None
        ):
        if tai_frequency is None:
            if max_depth is None or min_depth is None:
                raise ValueError("Either tai_frequency array or both max_depth and min_depth must be provided")
            tai_frequency = self.aggregate_tai_stacks(max_depth, min_depth)
        self._map_frequency(
            tai_frequency,
            title=f'TAI Frequency Map\n Defined by depth range: {min_depth:.2f}m to {max_depth:.2f}m',
            cbar_label='TAI Frequency (0-100%) of Days',
            cmap='RdYlBu_r',
            show_basin_footprint=show_basin_footprint,
        )

    def map_inundation_stacks(
            self, 
            inundation_frequency: np.array = None, 
            show_basin_footprint: bool = False,
            cbar_min: float = 0,
            cbar_max: float = None,
            plot_well: bool = True,
            show_scale_bar: bool = True,
            scale_bar_length: float = None,
            title: str = None,
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
        finite_vals = inundation_percent[np.isfinite(inundation_percent)]
        if finite_vals.size == 0:
            raise ValueError("No finite inundation-frequency values available for plotting")
    
        if cbar_max is None:
            vmax = float(np.nanmax(inundation_percent))
        else:
            vmax = cbar_max
        vmin = cbar_min
        if np.isclose(vmax, vmin):
            vmax = vmin + 1.0

        # Add the well point
        well_point = self.well_point
        well_point_x = well_point.location.x.values[0]
        well_point_y = well_point.location.y.values[0]

        
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['#8B4513', '#FFFFFF', '#0000FF']  # Brown, White, Blue
        custom_cmap = LinearSegmentedColormap.from_list('brown_white_blue', colors, N=256)

        if show_basin_footprint and self.basin.footprint is not None:
            footprint = self.basin.footprint
            footprint.boundary.plot(ax=ax, color='red', linewidth=2, alpha=0.8, label='Basin Footprint')
        elif show_basin_footprint:
            footprint = well_point.buffer(self.basin.transect_buffer)
            footprint.boundary.plot(ax=ax, color='red', linewidth=2, alpha=0.8, label='Basin Footprint')
        
        # Show inundation frequency (values from 0-1 representing frequency of being inundated)
        im = show(inundation_percent, ax=ax, cmap=custom_cmap, alpha=1,
                transform=self.basin.clipped_dem.transform,
            vmin=vmin, vmax=vmax,
            adjust=False)

        # Add colorbar
        cbar = plt.colorbar(im.images[0], ax=ax, shrink=0.8)
        cbar.set_label('Inundation Frequency % of Days', rotation=270, labelpad=20, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Plot well point
        if plot_well:
            ax.scatter(well_point_x, well_point_y, color='limegreen', 
                    marker='x', s=400, linewidths=7,
                    label=f'Well Location')

        if show_scale_bar:
            self._add_scale_bar(ax, scale_bar_length=scale_bar_length)

        #plot_shape.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
        
        # Add legend and labels
        #ax.legend(loc='upper right', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if title is None:
            title = (
                f"{self.well_stage.well_id} Inundation Frequency (% Days)\n"
                f"Well-to-DEM offset: {self.well_to_dem_offset:+.2f} m"
            )
        ax.set_title(title)

        if np.isclose(float(np.nanmax(inundation_percent)), 0.0):
            ax.text(
                0.02,
                0.98,
                "All mapped frequencies are 0% for this scenario.",
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=11,
                color='black',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
            )

        plt.show()

    def map_depth_stacks(self, depth_frequency: np.array = None, show_basin_footprint: bool = False):
        if depth_frequency is None:
            depth_frequency = self.aggregate_depth_stacks()
        self._map_frequency(
            depth_frequency,
            title='Mean Water Depth Map',
            cbar_label='Mean Depth (m)',
            cmap='RdYlBu',
            show_basin_footprint=show_basin_footprint,
            as_percent=False,
        )
