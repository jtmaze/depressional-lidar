import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Original dataset (first table) - numeric conversions applied
    original = {
        0: 6.75,  # averaged ~6.5 - 7.0
        50: 3.2,
        100: 2.4,
        200: 1.6,
        500: 0.65,
        700: 0.45,
        1000: 0.28,
        1200: 0.22,
        1500: 0.16,
        1700: 0.13,
        2000: 0.10,
    }

    # 20-day estimation (Demand: 105,633 gpd)
    drawdown_20_days = {
        0: 1.1,
        50: 0.57,
        100: 0.47,
        200: 0.39,
        500: 0.27,
        700: 0.22,
        1000: 0.18,
        1200: 0.16,
        1500: 0.13,
        1700: 0.12,
        2000: 0.10,
    }

    # 30-day estimation (Demand: 116,196 gpd)
    drawdown_30_days = {
        0: 1.64,
        50: 0.89,
        100: 0.74,
        200: 0.61,
        500: 0.43,
        700: 0.37,
        1000: 0.30,
        1200: 0.27,
        1500: 0.28,
        1700: 0.20,
        2000: 0.18,
    }

    # Time-averaged equivalent (from latest image)
    time_averaged = {
        800: 1.9,
        1000: 1.6,
        1500: 1.0,
        2000: 0.6,
        2700: 0.2,
    }

    # More realistic long-term values: use midpoints for ranges
    long_term = {
        800: 0.7,   # midpoint of 0.6-0.8
        1000: 0.55, # midpoint of 0.5-0.6
        1500: 0.35,
        2000: 0.2,
        2700: 0.05, # use small value for <0.1
    }

    # Combine all scenarios into a DataFrame
    scenarios = {
        'original': original,
        'drawdown_20_days': drawdown_20_days,
        'drawdown_30_days': drawdown_30_days,
        'time_averaged': time_averaged,
        'long_term': long_term,
    }

    # Build union of distances
    all_distances = sorted({d for s in scenarios.values() for d in s.keys()})

    df = pd.DataFrame(index=all_distances)
    for name, data in scenarios.items():
        df[name] = [data.get(d, np.nan) for d in all_distances]

    df.index.name = 'distance_ft'

    # Save combined CSV
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'drawdown_combined.csv')
    df.to_csv(csv_path)
    print(f'Wrote combined data to {csv_path}')

    # Plotting and curve fits
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(9,6))

    colors = ['C0','C1','C2','C3','C4']

    def fit_powerlaw(x, y):
        # Fit y = a * x^b using log transform. Skip zeros and nonpositive y.
        mask = (x > 0) & (y > 0)
        xk = x[mask]
        yk = y[mask]
        if len(xk) < 2:
            return None, None
        lx = np.log(xk)
        ly = np.log(yk)
        b, loga = np.polyfit(lx, ly, 1)
        a = np.exp(loga)
        # note: returned parameters correspond to y = exp(loga) * x**b
        return a, b

    x_vals_plot = np.linspace(1, max(all_distances), 500)

    for (name, col), c in zip(df.items(), colors):
        x = np.array(df.index.values)
        y = df[name].values
        ax.scatter(x, y, label=name, color=c, s=40)

        a, b = fit_powerlaw(x, y)
        if a is not None:
            yfit = a * (x_vals_plot**b)
            ax.plot(x_vals_plot, yfit, color=c, linestyle='--', alpha=0.8)
            print(f'Fitted {name}: y = {a:.4g} * x^{b:.4g}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance (ft)')
    ax.set_ylabel('Drawdown (ft)')
    ax.set_title('Drawdown scenarios and power-law fits')
    ax.legend()

    plot_path = os.path.join(out_dir, 'drawdown_fits.png')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    print(f'Saved plot to {plot_path}')

if __name__ == '__main__':
    main()
