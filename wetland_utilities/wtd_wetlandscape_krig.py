
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pykrige as pkr

@dataclass
class WellArray:
    """
    Processes well points and timeseries into kriging-ready water surface elevation points.

    Upon initiation the following steps happen:
        1) The well_ts is filtered to match the timeframe. If begin == end, filter by that single date. 
        2) Group well_ts by wetland_id and take the mean water level (well_depth_m).
        3) Join grouped well_ts to well_pts on "wetland_id", then compute wse_m = well_z - mean_depth.

    The result is stored in `wtd_points`, a GeoDataFrame ready for the kriging workflow.
    """
    well_pts: gpd.GeoDataFrame  # must have: wetland_id, well_z, geometry
    well_ts: pd.DataFrame       # must have: wetland_id, date, well_depth_m
    begin: str                  # date string, e.g. "2024-01-01"
    end: str                    # date string, e.g. "2024-12-31"

    def __post_init__(self):
        filtered = self._filter_timeseries()
        mean_depth = self._group_mean_depth(filtered)
        self.wtd_points = self._join_and_compute_wse(mean_depth)

    def _filter_timeseries(self) -> pd.DataFrame:
        ts = self.well_ts.copy()
        ts[ts['flag'] != 2]
        ts['date'] = pd.to_datetime(ts['date'])
        begin = pd.to_datetime(self.begin)
        end = pd.to_datetime(self.end)

        # NOTE: Ignoring flagging issues on well data for now
        if begin == end:
            return ts[ts['date'] == begin]
        return ts[(ts['date'] >= begin) & (ts['date'] <= end)]

    @staticmethod
    def _group_mean_depth(filtered_ts: pd.DataFrame) -> pd.DataFrame:
        return (
            filtered_ts
            .groupby('wetland_id', as_index=False)
            .agg(mean_depth=('well_depth_m', 'mean'))
        )

    def _join_and_compute_wse(self, mean_depth: pd.DataFrame) -> gpd.GeoDataFrame:
        pts = self.well_pts.merge(mean_depth, on='wetland_id', how='inner')
        pts['wse_m'] = pts['z_dem'] - pts['mean_depth']
        return pts

@dataclass 
class WTDSurface:
    """
    Runs ordinary kriging on a WellArray and exposes the interpolated surface,
    uncertainty grid, and visualization methods.
    """
    well_array: WellArray
    krig_params: dict  # {variogram_model: str, nlags: int}
    coarse_grid_dims: tuple  # (nx, ny) grid resolution
    boundary: gpd.GeoDataFrame

    def __post_init__(self):
        self._samples = self._extract_xyz()
        self._x_grid, self._y_grid = self._build_grid()

    def _extract_xyz(self) -> dict:
        pts = self.well_array.wtd_points
        return {
            'x': pts.geometry.x.to_numpy(),
            'y': pts.geometry.y.to_numpy(),
            'z': pts['wse_m'].to_numpy(),
        }

    def _build_grid(self) -> tuple[np.ndarray, np.ndarray]:
        # Handle both GeoDataFrame/GeoSeries and raw Shapely geometries
        if hasattr(self.boundary, 'total_bounds'):
            minx, miny, maxx, maxy = self.boundary.total_bounds
        else:
            minx, miny, maxx, maxy = self.boundary.bounds
            
        nx, ny = self.coarse_grid_dims
        return np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny)

    @cached_property
    def okr_result(self) -> dict:
        """Run ordinary kriging and return z_result and sigma_squared arrays."""
        ok = pkr.ok.OrdinaryKriging(
            self._samples['x'],
            self._samples['y'],
            self._samples['z'],
            variogram_model=self.krig_params.get('variogram_model', 'linear'),
            nlags=self.krig_params.get('nlags', 6),
        )
        z_result, sigma_squared = ok.execute('grid', self._x_grid, self._y_grid)
        return {'z': np.asarray(z_result), 'sigma_squared': np.asarray(sigma_squared)}

    # ---- Visualization helpers ------------------------------------------------

    def _base_plot(self, data: np.ndarray, cmap: str, label: str, title: str = None):
        """Shared plotting scaffold for imshow-based maps."""
        fig, ax = plt.subplots(figsize=(7, 7))
        extent = [self._x_grid.min(), self._x_grid.max(),
                  self._y_grid.min(), self._y_grid.max()]
        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label=label)
        ax.scatter(self._samples['x'], self._samples['y'], color='red', s=50)

        # Handle both GeoDataFrame/GeoSeries and raw Shapely geometries
        if hasattr(self.boundary, 'plot'):
            self.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
        else:
            gpd.GeoSeries([self.boundary]).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)

        if title:
            ax.set_title(title)
        plt.show()

    def plot_interpolation_result(self):
        self._base_plot(
            self.okr_result['z'],
            cmap='viridis',
            label='Interpolated WSE (m)',
        )

    def plot_sigma_squared(self):
        sigma = self.okr_result['sigma_squared'] ** 0.5
        self._base_plot(
            sigma,
            cmap='Reds',
            label='Kriging Uncertainty (m)',
            title='Kriging Uncertainty (prediction std dev)',
        )

    def plot_masked_result(self, sigma_threshold: float):
        """Show interpolation masked where uncertainty exceeds sigma_threshold."""
        sigma = self.okr_result['sigma_squared'] ** 0.5
        masked = np.where(sigma <= sigma_threshold, self.okr_result['z'], np.nan)
        self._base_plot(
            masked,
            cmap='viridis',
            label='Interpolated WSE (m)',
            title=f'Kriging Result (masked where uncertainty > {sigma_threshold} m)',
        )

    # def plot_contour(self):
    #     fig, ax = plt.subplots(figsize=(7, 7))
    #     cf = ax.contourf(self._x_grid, self._y_grid, self.okr_result['z'], cmap='viridis')
    #     plt.colorbar(cf, ax=ax, label='Interpolated WSE (m)')
    #     ax.scatter(self._samples['x'], self._samples['y'], color='red', s=50)
    #     self.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
    #     plt.show()






