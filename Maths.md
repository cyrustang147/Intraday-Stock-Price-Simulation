# Mathematical ideas behind the codes

## GARCH(1,1) model

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


## Seasonality

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


## Poisson trade arrivals and volumes

Given the seasonal volatility $\sigma_t$, we model trade arrivals with an **inhomogeneous Poisson process** with intensity $\lambda_t$. The expected number of trades in an interval of length $dt$ is:

$$\mathbb{E}[N_t] = \lambda_t dt$$

According to the Mixture-of-Distributions Hypothesis (MDH), information flow drives both volatility and volume. Thus:
- Higher volatility $\implies$ more trades (higher $\lambda_t$).
- Higher volatility $\implies$ larger average trade sizes.

In our simulator, both $\lambda_t$ and expected trade volume scale with volatility, making their product proportional to variance.

Trades are distributed uniformly within each grid interval. Trade prices are interpolated between grid points, with small microstructure noise added.

