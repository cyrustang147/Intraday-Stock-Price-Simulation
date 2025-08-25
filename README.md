# Intraday-Stock-Price-Simulation

A compact Python simulator that generates intraday mid-prices and trade events by combining a coarse-grid GARCH(1,1) model with a multiplicative parabolic intraday seasonal volatility profile. The simulator produces time-stamped trades (price, volume) and convenient grid outputs for analysis and plotting.

---

## Key features

* Deterministic **parabolic intraday seasonality** for volatility (higher at open/close).
* Coarse-grid **GARCH(1,1)** simulation mapped to per-step variance and then modulated by seasonality.
* Inhomogeneous Poisson trade arrival process whose intensity depends on instantaneous volatility.
* Log-normal trade volumes scaled to local volatility and optional microstructure noise on prices.
* Returns trade-level outputs (`time`, `price`, `volume`) plus useful grid-level arrays (`midprice_grid`, `grid_sigma`, `times_grid`).

## Requirements

* Python 3.8+
* numpy
* pandas
* matplotlib

Install with pip if needed:

```bash
pip install numpy pandas matplotlib
```

## Files / Functions (what's in the code)

* `intraday_parabolic_seasonality(n_steps, open_amplitude, close_amplitude)`

  * Builds a parabolic seasonal factor `s_t` on `[0,1]` such that `mean(s_t**2) == 1` and `s(0)==open_amplitude`, `s(1)==close_amplitude` with a minimum of `1` somewhere in the interval.

* `simulate_coarse_garch_with_seasonality(rng, daily_vol, open_amplitude, close_amplitude, trading_seconds, dt_coarse, alpha_g, beta_g, mu=0.0, burnin=500)`

  * Simulates coarse-grid GARCH(1,1) log-returns and applies the seasonal factor multiplicatively to variance. Returns `coarse_returns`, `coarse_sigma`, and `s`.

* `simulate_trades(...)`

  * Convenience wrapper that:

    1. Generates a mid-price path by exponentiating cumulative coarse log-returns.
    2. Builds a volatility-dependent intensity `lambda_t` for an inhomogeneous Poisson process (trades per second).
    3. Simulates trades inside each grid interval, samples volumes (log-normal) scaled to local volatility, and applies microstructure noise to prices.
  * Returns a dictionary with arrays: `trade_times`, `prices`, `volumes`, `grid_sigma`, `times_grid`, `midprice_grid`.

## Quick usage

Inside a Python script or notebook:

```python
from your_module import simulate_trades

sim = simulate_trades(dt=1.0, seed=42)

# Access trade-level data
trade_times = sim['trade_times']
prices = sim['prices']
volumes = sim['volumes']

# Access grid-level data
midprice_grid = sim['midprice_grid']
grid_sigma = sim['grid_sigma']

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame({'time': trade_times, 'price': prices, 'volume': volumes})
```

To reproduce the plotting used in the example: open a Jupyter notebook, run the simulation, then use `matplotlib` to visualize the first 60s of activity or the full-day mid-price with minutely volumes.

## Parameters you may want to tune

* `daily_vol`: target daily volatility (std). Typical values: `0.01 - 0.05` depending on asset.
* `open_amplitude`, `close_amplitude`: control how much more volatile open/close are relative to the midday trough. Must be `> 1.0`.
* `alpha_g`, `beta_g`: GARCH(1,1) parameters (`alpha + beta < 1` required).
* `lambda0`, `k_intensity`: baseline trade rate and sensitivity to volatility.
* `v0`, `vol_gamma`, `volume_noise_sigma`: volume scale, dependency on volatility, and lognormal noise level.
* `micro_sigma`, `tick_size`: microstructure noise magnitude and tick discretisation.

## Output and interpretation

* `trade_times`, `prices`, `volumes` form the trade tape. Prices are rounded to `tick_size`.
* `midprice_grid` is the (multiplicative) mid-price computed on the coarse grid; trades are interpolated between grid points.
* `grid_sigma` shows the per-step standard deviation after applying the GARCH and seasonal multiplier.

## Common extensions / ideas

* Add signed trade impact where large volumes move the mid-price (temporary or permanent impact).
* Make volumes follow a more realistic conditional distribution (e.g. mixture models or empirical sampling).
* Add limit order book simulation for quoted spreads and submission/cancellation dynamics.
* Save results to CSV/Parquet for downstream analysis.

## Example plot description (as in the provided script)

* **Showcase 1**: first 60 seconds — blue line: midprice; red dots: tick-rounded trades.
* **Showcase 2**: full-day midprice path (line) plus minutely aggregated traded volumes (bars).

## License

MIT License — adapt and reuse freely.

## Contact / author

You — please consider adding your name, email, or GitHub handle here if you want others to cite or contact you about the simulator.
