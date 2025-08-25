# Intraday-Stock-Price-Simulation

A compact Python simulator combines a GARCH(1,1) model with a parabolic intraday volatility trend and an inhomogeneous Poisson simulator. The simulator generates intraday trade events (time, price, volume) and convenient grid outputs for analysis and plotting.

---

## Key features

* Deterministic **parabolic intraday seasonality** for volatility (higher at market open/close).
* Coarse grid **GARCH(1,1)** to simulate volatility and log-returns, and then modulated by seasonality.
* Inhomogeneous Poisson arrival process whose intensity depends on instantaneous volatility to simulate trade arrivals.
* Trade volumes scaled to local volatility
* Add microstructure noise to mid-prices to generate trade prices
* Returns trade-level outputs (`time`, `price`, `volume`) plus useful grid-level arrays (`midprice_grid`, `grid_sigma`, `times_grid`).

(See Maths.md for background maths ideas)

## Requirements

* Python 3.8+
* numpy
* pandas
* matplotlib

Install with pip if needed:

```bash
pip install numpy pandas matplotlib
```

## Functions (what's in the code)

* `intraday_parabolic_seasonality(n_steps, open_amplitude, close_amplitude)`
  * Builds a parabolic seasonal factor `s_t` on `[0,1]` such that `mean(s_t**2) == 1` and `s(0)/min{s_t}==open_amplitude`, `s(1)/min{s_t}==close_amplitude`

* `simulate_coarse_garch_with_seasonality(rng, daily_vol, open_amplitude, close_amplitude, trading_seconds, dt_coarse, alpha_g, beta_g, mu=0.0, burnin=500)`
  * Simulates coarse-grid GARCH(1,1) volatilities and log-returns and applies the seasonal factor `s` multiplicatively to variance.
  * Returns `coarse_returns`, `coarse_sigma`, and `s`.

* `simulate_trades(...)`
  * Generates a mid-price path by exponentiating cumulative coarse log-returns.
  * Builds a volatility-dependent intensity `lambda_t` for an inhomogeneous Poisson process (trades per second).
  * Simulates trades inside each grid interval, samples volumes (log-normal) scaled to local volatility, and applies microstructure noise to prices.
  * Returns a dictionary with arrays: `trade_times`, `prices`, `volumes`, `grid_sigma`, `times_grid`, `midprice_grid`.

(See code comments for detailed explanations.)
 
## Output and interpretation

* `trade_times`, `prices`, `volumes` form the trade tape. Prices are rounded to `tick_size`.
* `midprice_grid` is the mid-price computed on the coarse grid; trades are interpolated between grid points.
* `grid_sigma` shows the per-step standard deviation after applying the GARCH and seasonal multiplier.

## Quick usage

Inside a Python script or notebook:

```python
sim = simulate_trades(dt=1.0, seed=67)

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

## Example plots (as in the provided script)

* **Showcase 1**: first 60 seconds — blue line: midprice; red dots: tick-rounded trades, observe the relationship between trades (arrival time, price) and midprice path.
* **Showcase 2**: full-day midprice path (line) plus minutely aggregated traded volumes (bars), a typical intraday stock chart.

To reproduce the example plots: open a Jupyter notebook, run the simulation, then use `matplotlib` to visualise

## Parameters you may want to tune

* `daily_vol`: target daily volatility (std). Typical values: `0.01 - 0.05` depending on asset.

* `open_amplitude`, `close_amplitude`: control how much more volatile open/close are relative to the midday trough. Must be > 1.0.

* `alpha_g`, `beta_g`: GARCH(1,1) parameters (alpha + beta < 1 required).

* `lambda0`, `k_intensity`: baseline trade rate and sensitivity to volatility.

* `v0`, `vol_gamma`, `volume_noise_sigma`: volume scale, dependency on volatility, and lognormal noise level.

* `micro_sigma`, `tick_size`: microstructure noise magnitude and tick discretisation.

## Common extensions / ideas

* Add signed trade impact where large volumes move the mid-price (temporary or permanent impact).
* Make volumes follow a more realistic conditional distribution (e.g. mixture models or empirical sampling).
* Add limit order book simulation for quoted spreads and submission/cancellation dynamics.
* Save results to CSV for downstream analysis.

## License

MIT License — adapt and reuse freely.

## Contact / author

Cyrus Tang — Github: cyrustang147
