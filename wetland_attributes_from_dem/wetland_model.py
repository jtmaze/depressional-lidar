
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from functools import cached_property

from basin_attributes import WetlandBasin
from basin_dynamics import WellStageTimeseries

@dataclass
class ForcingData:
    data: pd.DataFrame
    data_source: str = "ERA-5"

    @classmethod
    def from_csv(cls, file_path: str, data_source: str):
        """
        Constructs forcing data and sets up column names to handle different sources
        """
        raw = pd.read_csv(file_path)
        if data_source == "ERA-5":
            df = raw[['date', 'pet_m', 'precip_m']]
        else:
            print("Warning no column name handling for other data sources")

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        return cls(data_source=data_source, data=df)

@dataclass
class WetlandModel:
    basin: WetlandBasin
    well_stage_timeseries: WellStageTimeseries
    forcing_data: ForcingData
    # Model parameters with defaults
    delta: float = 0.38
    beta: float = 4.1
    alpha: float = 1.11
    n: float = 2.07
    n1: float = 0.15
    a: float = 2000
    c: float = 4

    @cached_property
    def raw_hypsometric_curve(self) -> pd.DataFrame:
        cum_area_m2, elevation= self.basin.calculate_hypsometry(method="pct_trim")
        return pd.DataFrame({'cum_area_m2': cum_area_m2, 'elevation': elevation})
    
    @cached_property
    def scaled_hypsometric_curve(self) -> pd.DataFrame:
        return self.rescale_hypsometric_curve()
    
    @cached_property
    def adjust_stage_timeseries(self) -> pd.DataFrame:
        """Convert well water levels to depth relative to basin low point."""
        ts = self.well_stage_timeseries.timeseries_data.copy()
        basin_low = self.basin.deepest_point.elevation
        well_height = self.basin.well_point.elevation_dem - basin_low
        ts['basin_depth'] = ts['water_level'] + well_height

        return ts
    
    @cached_property
    def Q_timeseries(self) -> pd.DataFrame:
        """
        Generate discharge timeseries from basin depth
        """
        ts = self.adjust_stage_timeseries.copy()
        ts['Q'] = ts['basin_depth'].apply(lambda h: self.Qh_A(h))

        return ts[['Q']]
    
    @cached_property
    def Sy_lookup_table(self) -> pd.DataFrame:
        """
        Precompute r_ET values over the expected depth range for fast lookups.
        """
        # Get depth range from adjusted stage timeseries
        h_min = self.adjust_stage_timeseries['basin_depth'].min() - 0.5
        h_max = self.adjust_stage_timeseries['basin_depth'].max() + 0.5
        
        # Create domain
        h_values = np.linspace(h_min, h_max, 200)
        Sy_values = [self.Sy(h) for h in h_values]
        
        return pd.DataFrame({'h': h_values, 'Sy': Sy_values})
    
    @cached_property
    def r_ET_lookup_table(self) -> pd.DataFrame:
        """
        Precompute r_ET values over the expected depth range for fast lookups.
        """
        # Get depth range from adjusted stage timeseries
        h_min = self.adjust_stage_timeseries['basin_depth'].min() - 0.5  # Add padding
        h_max = self.adjust_stage_timeseries['basin_depth'].max() + 0.5
        
        # Create domain
        h_values = np.linspace(h_min, h_max, 200)
        r_ET_values = [self.r_ET(h) for h in h_values]
        
        return pd.DataFrame({'h': h_values, 'r_ET': r_ET_values})


    @cached_property
    def Sy_timeseries(self) -> pd.DataFrame:
        """
        Generate Sy timeseires based on basin depth timeseries
        """
        ts = self.adjust_stage_timeseries.copy()
        lookup = self.Sy_lookup_table
        ts['Sy'] = np.interp(ts['basin_depth'], lookup['h'], lookup['Sy'])

        return ts[['Sy']]
    
    @cached_property
    def rET_timeseries(self) -> pd.DataFrame:
        """
        Generate relative ET timeseries from basin depth
        """
        ts = self.adjust_stage_timeseries.copy()
        lookup = self.r_ET_lookup_table
        ts['r_ET'] = np.interp(ts['basin_depth'], lookup['h'], lookup['r_ET'])

        return ts[['r_ET']]
    
    @staticmethod
    def _Ar_h_func(h_val, df):
        idx = np.argmin(np.abs(df['depth'] - h_val))
        return df['area_rescaled'].iloc[idx]
    
    @staticmethod
    def _h_Ar_func(Ar_val, df):
        idx = np.argmin(np.abs(df['area_rescaled'] - Ar_val))
        return df['depth'].iloc[idx]

    def rescale_hypsometric_curve(self):
        """
        """
        df = self.raw_hypsometric_curve
        df['depth'] = df['elevation'].transform(
            lambda x: (x - x.min())
        )
        df['area_rescaled'] = df['cum_area_m2'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        out_df = df[['depth', 'area_rescaled']].copy()

        return out_df

    def r_ET(self, h):

        df = self.scaled_hypsometric_curve.copy()

        Ar_hdelta_term = WetlandModel._Ar_h_func(h + self.delta, df)

        # Integrand
        def integrand(Ar):
            return np.exp(-self.beta * (WetlandModel._h_Ar_func(Ar, df) - h - self.delta))
        
        # Compute the integral
        int_result, _ = integrate.quad(integrand, Ar_hdelta_term, 1.0)

        result = Ar_hdelta_term + int_result

        return result
    
    def Sy(self, h):

        df = self.scaled_hypsometric_curve.copy()

        Ar_h = WetlandModel._Ar_h_func(h, df)

        term1 = Ar_h + (self.n1 * (1 - Ar_h))

        def integrand(Ar):
            h_Ar = WetlandModel._h_Ar_func(Ar, df)
            term = self.alpha * (h_Ar - h)
            #NOTE: Added this to help when integrand wasn't computing well.
            if term < 0: 
                return 0.001
            return (1 + term**self.n) ** (-(self.n + 1) / self.n)
        
        int_result, _ = integrate.quad(integrand, Ar_h, 1.0)

        result = term1 + self.n1 * int_result

        return result
        
    def plot_rET_and_Sy(self):
        """
        Plot r_ET and Sy on the same figure for comparison.
        """
        rET_data = self.r_ET_lookup_table
        Sy_data = self.Sy_lookup_table
        
        plt.figure(figsize=(10, 6))
        
        # Plot r_ET
        plt.plot(rET_data['h'], rET_data['r_ET'], 
                label='r_ET', linewidth=2)
        
        # Plot Sy
        plt.plot(Sy_data['h'], Sy_data['Sy'], 
                label='Sy', linewidth=2)
        
        plt.xlabel('h (depth)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('r_ET and Sy vs h', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.show()

    def Qh_A(self, h):
        """Calculate normalized outflow as a function of depth."""
        max_A = self.raw_hypsometric_curve['cum_area_m2'].max()
        # TODO: Modify Spill logic for Bradfod
        spill = 0.1
        Qh = self.a * ((h - spill) ** self.c)
        Qh_A = Qh / max_A
        
        return Qh_A

    def plot_Qh_A(self, n_points: int = 100):
        """
        Plot normalized outflow (Qh/A) as a function of depth.
        """
        h_min = self.scaled_hypsometric_curve['depth'].min()
        h_max = self.scaled_hypsometric_curve['depth'].max()
        
        # Create domain - only plot where h > spill (otherwise negative)
        domain = np.linspace(h_min, h_max, n_points)
        vals = [self.Qh_A(h) for h in domain]
        
        plt.figure(figsize=(10, 6))
        plt.plot(domain, vals, linewidth=2)
        plt.xlabel('h (depth)', fontsize=12)
        plt.ylabel('Qh/A (normalized outflow [m/s])', fontsize=12)
        plt.title('Normalized Discharge vs Depth', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_Q_timeseries(self):
        """
        Plot discharge over time.
        """
        df = self.Q_timeseries
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Q'], linewidth=1.5)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Q (normalized discharge [m/s])', fontsize=12)
        plt.title('Discharge Timeseries', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_Sy_timeseries(self):
        """
        Plot specific yield over time.
        """
        data = self.Sy_timeseries
        
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Sy'], linewidth=1.5, color='green')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sy (specific yield)', fontsize=12)
        plt.title('Specific Yield Timeseries', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_rET_timeseries(self):
        """
        Plot relative ET over time.
        """
        data = self.rET_timeseries
        
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['r_ET'], linewidth=1.5, color='orange')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('r_ET (relative evapotranspiration)', fontsize=12)
        plt.title('Relative ET Timeseries', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    

    

