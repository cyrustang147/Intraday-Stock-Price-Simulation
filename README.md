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

## Requirements

* Python 3.8+
* numpy
* pandas
* matplotlib

Install with pip if needed:

```bash
pip install numpy pandas matplotlib
```

## Mathematical ideas

### GARCH(1,1) model

The GARCH(1,1) is a standard model for simulating returns and volatility dynamics. In its simplest form (no seasonality, zero mean):

$$\epsilon_t = \sigma_t z_t$$

where $z_t$ are i.i.d. standard Normal random variables, independent of $\sigma_t$. The conditional variance evolves as:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

with parameters $\omega, \alpha, \beta > 0$ and the stationarity condition $\alpha + \beta < 1$.

Taking expectations and noting that $\mathbb{E}[\epsilon_t^2] = \mathbb{E}[\sigma_t^2]$, we obtain:

$$\omega = \overline{\sigma^2}(1 - \alpha - \beta)$$

where $\overline{\sigma^2}$ is the long-run variance per step (calibrated from the daily volatility).

**Why GARCH(1,1)?**
- **Simplicity**: minimal number of parameters, easy to estimate and simulate.
- **Empirical adequacy**: captures two key facts of financial returns — volatility clustering and heavy tails. More lags rarely improve results substantially.


### Seasonality

Volatility is typically higher at the market open and close. A simple way to model this is with a parabola. Let:
- $o$ = ratio of open volatility to midday volatility,
- $p$ = ratio of close volatility to midday volatility.

We construct a parabola $f(u) = a u^2 + b u + c$ on $u \in [0,1]$ such that:
- $f(0) = c = o$
- $f(1) = a + b + c = p$
- $\min f(u^\star) = 1$ for some $u^\star \in [0,1]$

Solution:
- $b = -2((o - 1) + \sqrt{(o - 1)(p - 1)})$
- $a = p - o - b$
- $c = o$

Finally, $f$ is normalized so that $\Vert f \Vert_{L^2} = 1$. This ensures the volatility process retains the intended overall scale.


### Poisson trade arrivals and volumes

Given the seasonal volatility $\sigma_t$, we model trade arrivals with an **inhomogeneous Poisson process** with intensity $\lambda_t$. The expected number of trades in an interval of length $dt$ is:

$$\mathbb{E}[N_t] = \lambda_t dt$$

According to the Mixture-of-Distributions Hypothesis (MDH), information flow drives both volatility and volume. Thus:
- Higher volatility $\implies$ more trades (higher $\lambda_t$).
- Higher volatility $\implies$ larger average trade sizes.

In our simulator, both $\lambda_t$ and expected trade volume scale with volatility, making their product proportional to variance.

Trades are distributed uniformly within each grid interval. Trade prices are interpolated between grid points, with small microstructure noise added.


## Functions (what's in the code)

* `intraday_parabolic_seasonality(n_steps, open_amplitude, close_amplitude)`

* `simulate_coarse_garch_with_seasonality(rng, daily_vol, open_amplitude, close_amplitude, trading_seconds, dt_coarse, alpha_g, beta_g, mu=0.0, burnin=500)`

* `simulate_trades(...)`

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
