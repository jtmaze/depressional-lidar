
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from functools import cached_property

from basin_attributes import WetlandBasin
from basin_dynamics import WellStageTimeseries

@dataclass
class WetlandModel:
    basin: WetlandBasin
    well_stage_timeseries: WellStageTimeseries

    @cached_property
    def raw_hypsometric_curve(self) -> pd.DataFrame:
        cum_area_m2, elevation= self.basin.calculate_hypsometry(method="pct_trim")
        return pd.DataFrame({'cum_area_m2': cum_area_m2, 'elevation': elevation})
    
    @cached_property
    def scaled_hypsometric_curve(self) -> pd.DataFrame:
        return self.rescale_hypsometric_curve()
    
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

    def r_ET(self, h, delta: float = 0.38, beta: float = 4.1):

        df = self.scaled_hypsometric_curve.copy()

        Ar_hdelta_term = WetlandModel._Ar_h_func(h + delta, df)

        # Integrand
        def integrand(Ar):
            return np.exp(-beta * (WetlandModel._h_Ar_func(Ar, df) - h - delta))
        
        # Compute the integral
        int_result, _ = integrate.quad(integrand, Ar_hdelta_term, 1.0)

        result = Ar_hdelta_term + int_result

        return result
    
    def rET_for_plot(self):
        from scipy.optimize import brentq

        def _find_domain_rET(tgt, low, high):
            return brentq(lambda h: self.r_ET(h) - tgt, low, high)
        
        try: 
            h_min = _find_domain_rET(0.05, -2, -0.5)
        except Exception:
            h_min = -1
        try:
            h_max = _find_domain_rET(1, 0.4, 1.5)
        except Exception:
            h_max = 0.5

        domain = np.linspace(h_min, h_max, 100)
        vals = [self.r_ET(h) for h in domain]

        return dict(zip(domain, vals))

    def Sy(self, h, alpha: float = 1.11, n: float = 2.07, n1=0.15):

        df = self.scaled_hypsometric_curve.copy()

        Ar_h = WetlandModel._Ar_h_func(h, df)

        term1 = Ar_h + (n1 * (1 - Ar_h))

        def integrand(Ar):
            h_Ar = WetlandModel._h_Ar_func(Ar, df)
            term = alpha * (h_Ar - h)
            if term < 0: 
                return 0.05
            return (1 + term**n) ** (-(n + 1) / n)
        
        int_result, _ = integrate.quad(integrand, Ar_h, 1.0)

        result = term1 + n1 * int_result

        return result
    
    def Sy_for_plot(self):
        from scipy.optimize import brentq

        def _find_domain_Sy(tgt, low, high):
            return brentq(lambda h: self.Sy(h) - tgt, low, high)
        
        try: 
            h_min = _find_domain_Sy(0.15, -2, -0.5)
        except Exception:
            h_min = -1
        try:
            h_max = _find_domain_Sy(1, 0.2, 1.5)
        except Exception:
            h_max = 0.5

        domain = np.linspace(h_min, h_max, 100)
        vals = [self.Sy(h) for h in domain]

        return dict(zip(domain, vals))
    
    def plot_rET_and_Sy(self):
        """
        Plot r_ET and Sy on the same figure for comparison.
        """
        rET_data = self.rET_for_plot()
        Sy_data = self.Sy_for_plot()
        
        plt.figure(figsize=(10, 6))
        
        # Plot r_ET
        plt.plot(list(rET_data.keys()), list(rET_data.values()), 
                label='r_ET', linewidth=2)
        
        # Plot Sy
        plt.plot(list(Sy_data.keys()), list(Sy_data.values()), 
                label='Sy', linewidth=2)
        
        plt.xlabel('h (depth)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('r_ET and Sy vs h', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.show()

    def Q(self, a: float = 2_000, c: float = 4):



    
    
    

    

