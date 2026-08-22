# ARIMA

**Topic:** time series forecasting. **Prerequisites:** college-level algebra and calculus.

ARIMA is a family of forecasting models for equally-spaced time series. It requires only the series itself, fits in milliseconds, and remains the baseline that any forecaster must beat. Understanding its derivation clarifies when it works and when it fails.

**Strategy:** transform the series via logs and/or differencing until stationary; model the remainder as a linear machine driven by white noise; forecast by taking expectations; verify residuals look like white noise.

---

## Preliminaries

**Expectation.** A random variable $X$ has expectation $E[X] = \sum_i x_i\,p_i$ (discrete) or $E[X] = \int x\,f(x)\,dx$ (continuous with density $f$). Expectation is linear:

$$E[aX + bY] = a\,E[X] + b\,E[Y]$$

with no additional assumptions.

**Variance.** With $\mu_X = E[X]$,

$$\mathrm{Var}(X) = E\big[(X-\mu_X)^2\big]$$

measures spread; its square root is the standard deviation. From linearity: $\mathrm{Var}(X+c) = \mathrm{Var}(X)$ and $\mathrm{Var}(aX) = a^2\mathrm{Var}(X)$.

**Covariance and correlation.**

$$\mathrm{Cov}(X,Y) = E\big[(X-\mu_X)(Y-\mu_Y)\big] \qquad \mathrm{Corr}(X,Y) = \frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)\,\mathrm{Var}(Y)}}$$

Covariance measures whether deviations share a sign; $\mathrm{Var}(X) = \mathrm{Cov}(X,X)$. Correlation rescales to $[-1,1]$ and is unit-free. Covariance is linear: $\mathrm{Cov}(aX,bY) = ab\,\mathrm{Cov}(X,Y)$ and $\mathrm{Cov}(X+Z,Y) = \mathrm{Cov}(X,Y) + \mathrm{Cov}(Z,Y)$. **Uncorrelated** means $\mathrm{Cov} = 0$.

**The workhorse identity.** Expanding $\mathrm{Var}(X+Y) = E[(X-\mu_X+Y-\mu_Y)^2]$:

$$\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y)$$

For uncorrelated variables, variances add: $\mathrm{Var}(\sum_i a_iX_i) = \sum_i a_i^2\mathrm{Var}(X_i)$.

**The normal distribution.** $X \sim N(\mu, \sigma^2)$ has a bell-curve density centred at $\mu$ with variance $\sigma^2$. Key facts:

1. *95% rule:* ~95% of probability lies within $\mu \pm 2\sigma$ (precisely 1.96σ). This is why 2 appears in significance bands and forecast intervals.
2. *Closure:* linear combinations of jointly normal variables are normal. So weighted sums of normal shocks are normal, legitimising "$\pm 2$ standard errors" as 95% intervals.
3. *Independence:* for jointly normal variables, uncorrelated implies independent. Non-jointly normal variables can be uncorrelated yet dependent. Likelihood construction requires independence, not just uncorrelatedness.

**p-values.** Propose a null hypothesis, compute a statistic, and ask: *if the null were true, what is P(statistic at least this extreme)?* That probability is the p-value. Small p (< 0.05) means the data would be unlikely under the null — reject it. Large p means no evidence against the null, which is *not* proof it is true.

**Sample statistics.** We observe samples, not the full process. From data $y_1, \dots, y_T$, compute estimates (hats):

$$\bar y = \frac{1}{T}\sum_{t=1}^{T} y_t \qquad\qquad \hat\rho_k = \frac{\sum_{t=k+1}^{T}(y_t-\bar y)(y_{t-k}-\bar y)}{\sum_{t=1}^{T}(y_t-\bar y)^2}$$

The sample autocorrelation $\hat\rho_k$ is the empirical $\rho_k$.

---

## Part I — Theory

### 1. Problem statement

Forecasting: given observations $y_1, \dots, y_T$ at equal time intervals, produce for each horizon $h \ge 1$ a point forecast $\hat y_{T+h}$ with an uncertainty interval. The process has two parts: transform the raw series to stationary form, then fit a linear model to the stationary remainder.

**Types.**

| Symbol | Mathematical type | Remarks |
|---|---|---|
| $y_t$ | real-valued sequence indexed by $t = 1, \dots, T$ | equal spacing assumed |
| $\varepsilon_t$ | white-noise sequence | only random primitive |
| $B$ | operator on sequences, $By_t = y_{t-1}$ | algebraic variable |
| $\phi(B),\ \theta(B)$ | polynomials in $B$ of degrees $p$, $q$ | carry model structure |
| $(p, d, q)$ | non-negative integers | fit to data |
| $\hat y_{T+h}$ | real number per horizon $h$ | meaningless without uncertainty interval |

Note: the transformation is part of the model — systems differing in $d$ or logs are distinct forecasters with non-comparable likelihoods. Long-horizon forecasts carry little information beyond the mean or trend.

---

### 2. White noise and stationarity

#### 2.1 White noise

White noise $\varepsilon_t$ satisfies, for all $t$ and $s \ne t$:

$$E[\varepsilon_t] = 0 \qquad \mathrm{Var}(\varepsilon_t) = \sigma^2 \qquad \mathrm{Cov}(\varepsilon_t, \varepsilon_s) = 0$$

Mean zero, constant variance, no autocorrelation — *linear unpredictability*: no linear function of past shocks predicts the next. This is weaker than full independence (uncorrelated shocks can be dependent), but sufficient for ARIMA, a linear machine. White noise is ARIMA's only random primitive; all models below are deterministic linear machines fed by white noise. Shocks need not be normal; normality is added only for likelihoods and intervals.

Pattern: two different $\varepsilon$'s in an expectation give zero; a shock with itself gives $\sigma^2$.

#### 2.2 Stationarity

A series $y_t$ is **weakly stationary** if its first two moments are time-invariant:

1. $E[y_t] = \mu$ — the same for every $t$;
2. $\mathrm{Var}(y_t) = \gamma_0$ — the same for every $t$;
3. $\mathrm{Cov}(y_t, y_{t-k}) = \gamma_k$ — depending only on the gap $k$, not on $t$.

The $\gamma_k$ are the **autocovariances**; normalising gives the **autocorrelation function (ACF)**:

$$\rho_k = \frac{\gamma_k}{\gamma_0} \in [-1, 1]$$

"Weakly" constrains only means and covariances. For normal series, weak and strict stationarity coincide. We use "stationary" to mean weakly stationary.

#### 2.3 Why stationarity is the entry ticket

You observe only one history. Averaging over time only yields meaningful results if there is a single time-invariant quantity to estimate.

Under stationarity, this holds: fixed mean $\mu$, constant variance $\gamma_0$, and autocorrelations $\rho_k$ depending only on lag $k$. Then $\bar y$ estimates $\mu$ and $\hat\rho_k$ estimates $\rho_k$.

Without stationarity, each sample may come from a different distribution. In a trending series, the mean changes over time, so $\bar y$ averages different quantities. The ACF shows huge, slowly fading positive bars at all lags — a signature that differencing is needed.

#### 2.4 Two workouts

**Is white noise stationary?** Mean 0, variance $\sigma^2$, $\gamma_k = 0$ for $k \ge 1$. Yes — the classic stationary series, with ACF $\rho_0 = 1$ and $\rho_k = 0$ beyond. A flat ACF means "nothing left to model."

**Is a random walk stationary?** $y_t = \varepsilon_1 + \cdots + \varepsilon_t$. Variance: $\mathrm{Var}(y_t) = t\sigma^2$, which grows with $t$ — condition 2 fails, so not stationary.

#### 2.5 The log transform, justified

Many series vary proportionally to their level. Model: $y_t = L_t(1+u_t)$, take logs:

$$\log y_t = \log L_t + \log(1+u_t) \approx \log L_t + u_t$$

using $\log(1+x) \approx x$. On the log scale, proportional swings become additive — constant variance restored. Rule: **if swings scale with level, log first.** Forecasts in logs must be exponentiated back.

---

### 3. The lag operator and differencing

#### 3.1 The lag operator

The lag operator $B$ shifts a series backward: $By_t = y_{t-1}$. Repeated: $B^k y_t = y_{t-k}$, with $B^0 = 1$ and $Bc = c$. $B$ is linear: $B(ax_t + by_t) = aBx_t + bBy_t$. Polynomials in $B$ behave like ordinary polynomials.

#### 3.2 Differencing is a discrete derivative

$$\nabla y_t = y_t - y_{t-1} = (1-B)y_t$$

This is a discrete derivative. On a linear trend $(at+b)$: $(1-B)(at+b) = a$ — degree reduction, mirroring differentiation. On $t^2$: $(1-B)t^2 = 2t-1$ — degree 2 → 1. So $(1-B)$ maps degree $n$ to $n-1$, and $d$ applications $(1-B)^d$ annihilate polynomial trends of degree $< d$.

The inverse is cumulative summing: if $z_t = (1-B)y_t$ then $y_t = y_0 + z_1 + \cdots + z_t$ — discrete integration, the **I** in ARIMA.

#### 3.3 Trend plus noise, drift, and the cost of over-differencing

Difference a trending series $y_t = at + b + \varepsilon_t$:

$$\nabla y_t = a + \varepsilon_t - \varepsilon_{t-1}$$

Two consequences: the slope $a$ survives as **drift** (nonzero mean of differenced series), and the noise $\varepsilon_t - \varepsilon_{t-1}$ is no longer white.

For $z_t = \varepsilon_t - \varepsilon_{t-1}$: $\mathrm{Var}(z_t) = 2\sigma^2$ (variances add, constants squared). Lag-1 covariance:

$$\gamma_1 = E[(\varepsilon_t - \varepsilon_{t-1})(\varepsilon_{t-1} - \varepsilon_{t-2})] = -\sigma^2$$

So $\rho_1 = -1/2$ exactly. The difference doubled the variance and introduced lag-1 correlation of $-0.5$.

The same happens, in proportion, whenever a stationary series is differenced. Let $y_t$ be stationary with autocovariances $\gamma_k$ and set $z_t = y_t - y_{t-1}$. By the workhorse identity, $\mathrm{Var}(z_t) = 2\gamma_0 - 2\gamma_1$, and expanding the lag-1 product term by term, $\mathrm{Cov}(z_t, z_{t-1}) = 2\gamma_1 - \gamma_0 - \gamma_2$. Take the simplest fading pattern as a hypothesis about the series: autocovariances that decay geometrically, $\gamma_k = \phi^k\gamma_0$ for some $|\phi| < 1$. Then

$$\mathrm{Var}(z_t) = 2(1-\phi)\,\gamma_0 \qquad \rho_1^{(z)} = \frac{-(1-\phi)^2\gamma_0}{2(1-\phi)\gamma_0} = -\frac{1-\phi}{2}$$

White noise is the case $\phi = 0$: variance doubled, $\rho_1 = -1/2$. The closer the series already was to a random walk ($\phi \to 1$), the less differencing costs — at $\phi = 0.7$ the variance *falls* to $0.6\,\gamma_0$ and $\rho_1 = -0.15$ — which is why the damage is easy to miss when the series is persistent. In this geometric case differencing always pushes the lag-1 autocorrelation negative; the farther it lands below zero, the more stationary the series was before you differenced it. A lone negative $\rho_1$ spike near $-0.5$ after differencing is the signature of differencing white noise: *reduce* $d$.

#### 3.4 The number of differences

In practice $d \in \{0, 1, 2\}$; more is rarely needed. Prefer smaller $d$ — over-differencing inflates the noise and manufactures negative autocorrelation that a spurious MA term is then needed to absorb.

---

### 4. Autoregressive models

#### 4.1 The model and its solution

After logs and/or differencing, you have a stationary series. The simplest hypothesis: the present is a linear function of the recent past plus a shock,

$$y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t$$

— **AR($p$)**. Collect terms using lag-operator algebra:

$$\phi(B)y_t = c + \varepsilon_t \qquad \phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$$

Set $c = 0$ from here — it only shifts the mean.

**Patterns captured by AR:** persistence and infinite memory — today depends on every past value, which in turn carry the imprint of every shock that ever happened. The ACF exhibits geometric decay (possibly sign-alternating or quasi-cyclic from complex roots), identifying AR as the model for fading, never-cutting correlations.

For AR(1), divide and expand $\frac{1}{1-x} = 1 + x + x^2 + \cdots$ with $x = \phi B$:

$$y_t = \frac{1}{1-\phi B}\varepsilon_t = \sum_{k=0}^{\infty}\phi^k \varepsilon_{t-k} = \varepsilon_t + \phi\varepsilon_{t-1} + \phi^2\varepsilon_{t-2} + \cdots$$

Treating this as a formal power series, the inverse is verified by multiplying by $(1-\phi B)$. Convergence requires $|\phi| < 1$ — the stationarity condition.

**Interpretation:** today is every shock that ever happened, discounted by $\phi$ per period. A shock from 10 periods back weighs $\phi^{10}$ — infinite memory with geometric fade.

**MA($\infty$) weights.** When written as $y_t = \sum_{j=0}^{\infty}\psi_j\varepsilon_{t-j}$, the $\psi_j$ are **impulse responses**: how much one shock matters $j$ periods later. Always $\psi_0 = 1$. For AR(1), $\psi_j = \phi^j$; for ARMA, $\psi_j$ are coefficients of $\theta(B)/\phi(B)$.

#### 4.2 The variance, every step shown

Apply the workhorse identity: different shocks are uncorrelated, variances add, constants squared:

$$\mathrm{Var}(y_t) = \sum_{k=0}^{\infty}\phi^{2k}\sigma^2 = \frac{\sigma^2}{1-\phi^2}$$

A geometric series, convergent when $|\phi| < 1$, constant in $t$ as stationarity requires. Note $\mathrm{Var}(y_t) > \sigma^2$: persistence amplifies noise ($\phi = 0.95$ gives ~10×).

#### 4.3 Yule–Walker: deriving the ACF

The standard trick: multiply the model by a past value and take expectations. For $k \ge 1$:

$$E[y_t\,y_{t-k}] = \phi\,E[y_{t-1}\,y_{t-k}] + E[\varepsilon_t\,y_{t-k}]$$

The last term is 0: $y_{t-k}$ is a weighted sum of shocks dated $t-k$ and earlier, and $\varepsilon_t$ is uncorrelated with every one of them. With zero means, the surviving expectations are autocovariances:

$$\gamma_k = \phi\,\gamma_{k-1} \ \ (k \ge 1) \qquad\Longrightarrow\qquad \gamma_k = \phi^k\gamma_0 \qquad\Longrightarrow\qquad \boxed{\rho_k = \phi^k}$$

Geometric decay. AR memory fades, never cuts off. For AR($p$) this yields the **Yule–Walker equations**: $p$ linear equations relating $\rho$'s and $\phi$'s. Uses: forward for the theoretical ACF; backward, with sample correlations in place of $\rho$'s, for closed-form $\phi$ estimates (least squares and Yule–Walker agree up to end effects).

#### 4.4 General $p$: stationarity by factoring

Over its roots $z_1, \dots, z_p$, $\phi(B)$ factors as:

$$\phi(B) = \prod_{i=1}^{p}(1 - \lambda_i B) \qquad \lambda_i = 1/z_i$$

Since polynomials in $B$ multiply like numbers, invert factor by factor:

$$y_t = \frac{1}{(1-\lambda_1 B)\cdots(1-\lambda_p B)}\varepsilon_t$$

Each factor is a geometric series requiring $|\lambda_i| < 1$, i.e. $|z_i| > 1$. So:

$$\text{AR}(p) \text{ stationary } \iff \text{ every root of } \phi(z) = 0 \text{ lies outside the unit circle}$$

This generalises AR(1)'s condition $|\phi| < 1$. Small coefficient ↔ distant root; the inversion explains why the condition reads backwards.

#### 4.5 How general ACFs decay: the recursion method

Yule–Walker for AR($p$) gives, at lags beyond $p$, the recursion $\rho_k = \phi_1\rho_{k-1} + \cdots + \phi_p\rho_{k-p}$. Solving this linear recursion: try $\rho_k = \lambda^k$; substitution shows $\lambda$ must satisfy the **characteristic equation**, whose solutions are $\lambda_i = 1/z_i$ from the factoring argument. The general solution is $\rho_k = \sum_i c_i\lambda_i^k$. Stationarity means every $|\lambda_i| < 1$, so **every AR's ACF is a mixture of geometric decays** — the fading-ACF fingerprint for identification.

#### 4.6 Boundary and special cases

- **$\phi = 1$ (unit root):** $\psi_j = 1$, shocks never fade, $y_t = \varepsilon_1 + \cdots + \varepsilon_t$ — random walk, variance $t\sigma^2$, non-stationary. Cure: difference with $(1-B)$, which removes the factor $(1 - B)$.
- **$-1 < \phi < 0$:** stationary, but $\rho_k = \phi^k$ alternates sign — the series overcorrects every period.
- **Complex roots → quasi-cycles.** AR(2) roots can be a complex-conjugate pair. In polar form $\lambda = re^{\pm i\omega}$, the conjugate terms combine into a damped cosine:

$$\rho_k \sim r^k\cos(\omega k + \text{phase}) \qquad r < 1$$

Period $2\pi/\omega$, decaying at rate $r$. Complex roots produce quasi-periodic wobbles without explicit sine waves.

**AR's fingerprint:** ACF is a mixture of geometric decays — fading, possibly sign-flipping or cycling, never cliffing.

---

### 5. Moving-average models

#### 5.1 The model and its weights

AR makes today depend on past *values*; MA makes it depend on past *shocks* directly:

$$y_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \cdots + \theta_q\varepsilon_{t-q} = \mu + \theta(B)\varepsilon_t \qquad \theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$$

**MA($q$)**. A shock takes exactly $q$ periods to finish rippling through the system. Set $\mu = 0$.

**Patterns captured by MA:** transience and finite memory — today depends directly on a finite window of recent shocks. The ACF cuts off to zero after lag $q$, creating the characteristic cliff. MA excels at modeling sharp, short-lived deviations where the effect of a shock disappears completely after a fixed number of periods.

An MA($q$) is already in MA($\infty$) form — weights simply stop:

$$\psi_0 = 1 \qquad \psi_j = \theta_j \  (1 \le j \le q) \qquad \psi_j = 0 \  (j > q)$$

$\psi_0 = 1$ means today's shock always enters undiscounted.

Stationarity is automatic: finite sum of uncorrelated shocks, workhorse identity applies directly —

$$\mathrm{Var}(y_t) = \sigma^2\big(1 + \theta_1^2 + \cdots + \theta_q^2\big) = \sigma^2\sum_j\psi_j^2$$

Finite and constant for any $\theta$'s. Mean 0, covariances depend only on lag: all weak stationarity conditions hold unconditionally. No root condition is needed for stationarity.

#### 5.2 The ACF, computed

For MA(1), $y_t = \varepsilon_t + \theta\varepsilon_{t-1}$. $\gamma_1$ is the expectation of the product. Expanding by linearity (different shocks die, matched shocks give $\sigma^2$):

$$\gamma_1 = E[(\varepsilon_t + \theta\varepsilon_{t-1})(\varepsilon_{t-1} + \theta\varepsilon_{t-2})] = \theta\sigma^2$$

Divide by $\gamma_0 = (1+\theta^2)\sigma^2$:

$$\boxed{\rho_1 = \frac{\theta}{1+\theta^2}}$$

($\theta = 0.7$ gives ~0.47.) At lag 2 and beyond, $y_t$ and $y_{t-2}$ share no shocks, so $\rho_k = 0$ *exactly* for $k \ge 2$ — a cliff, not a fade. Generally: MA($q$)'s ACF is nonzero through lag $q$, zero after. The anti-AR fingerprint.

**Bound.** Maximise $f(\theta) = \theta/(1+\theta^2)$ by calculus: $f'(\theta) = (1-\theta^2)/(1+\theta^2)^2 = 0$ at $\theta = \pm 1$, so $f(\pm 1) = \pm 1/2$. Thus $|\rho_1| \le 1/2$ always for MA(1). An ACF showing $\rho_1 = 0.8$ with a clean cutoff is *impossible* for MA(1) — the pattern rules it out. The boundary value $\rho_1 = -1/2$ is reached only at $\theta = -1$, which is exactly the lag-1 autocorrelation of differenced white noise: over-differencing manufactures an MA(1) sitting on the edge of its parameter space.

#### 5.3 Invertibility

Invertibility ensures shocks can be stably recovered and parameters uniquely identified. Without it, multiple parameter values produce identical autocorrelations, making estimation impossible. For example, substitution $\theta \to 1/\theta$ in $\rho_1 = \theta/(1+\theta^2)$ gives the same $\rho_1$, but different variances. Models $(\theta, \sigma^2)$ and $(1/\theta, \theta^2\sigma^2)$ produce identical autocovariances: the data cannot distinguish them — *unidentified*.

The fix: require $\theta(z) = 1+\theta z$'s root $z = -1/\theta$ to lie outside the unit circle, i.e. $|\theta| < 1$, selecting one member of each pair. That is **invertibility**, the mirror of AR's stationarity condition, guarding identifiability.

Invert the operator with geometric series ($x = -\theta B$):

$$\varepsilon_t = \frac{1}{1+\theta B}y_t = \sum_{k=0}^{\infty}(-\theta)^k y_{t-k}$$

Converges when $|\theta| < 1$: shocks are stably recoverable, and any error in the starting shock is multiplied by $(-\theta)^t$ and dies out. Rearranged, $y_t = \varepsilon_t - \sum_{k \ge 1}(-\theta)^k y_{t-k}$: MA(1) = AR($\infty$), the mirror of AR(1) = MA($\infty$). Its dependence on past *values* never cuts off, just as AR's dependence on past *shocks* never cuts off.

#### 5.4 Why mix AR and MA

Each type is infinitely expensive in the other's currency: a pure AR model needs infinitely many lags to capture direct shock influence, and a pure MA model needs infinitely many shocks to capture past values influence. Take ARMA(1,1): $(1-\phi B)y_t = (1+\theta B)\varepsilon_t$. Write as pure AR: $\pi(B)y_t = \varepsilon_t$ with $\pi(B) = (1-\phi B)/(1+\theta B)$. Expand:

$$\pi(B) = (1-\phi B)(1 - \theta B + \theta^2 B^2 - \cdots)$$

Coefficient of $B^j$: $(-\theta)^j - \phi(-\theta)^{j-1} = -(-\theta)^{j-1}(\theta+\phi)$. The AR form:

$$y_t = \sum_{j=1}^{\infty}\tilde\pi_j y_{t-j} + \varepsilon_t \qquad \tilde\pi_j = (\phi+\theta)(-\theta)^{j-1}$$

Nonzero at every lag (unless $\phi = -\theta$, degenerate white noise). Finite AR($p$) truncates this tail and mis-fits. Matching it may take 10 lags where ARMA(1,1) uses two parameters. **Parameter thrift is the case for mixing: AR and MA together model rich patterns efficiently.**

---

### 6. The full model

$$\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t$$

**ARIMA($p, d, q$)**. $d$ differences to reach stationarity, then $p$ AR and $q$ MA terms on the differenced series, with $c$ carrying drift. The stationarity condition applies to $\phi(B)$ and the invertibility condition to $\theta(B)$; the unit root is removed by differencing $(1-B)^d$, not estimated as a parameter.

---

## Part II — Practice

### 7. Choosing p, d, q

#### 7.1 Order of operations

**Settle $d$ first** — log if swings scale with level, difference while the plot drifts, use ADF to formalise. Fingerprints are only meaningful on stationary series. Warning: an ACF with huge, slowly fading positive bars at all lags signals non-stationarity — difference and re-plot. Then compute the sample ACF $\hat\rho_k$ and PACF on the differenced series.

#### 7.2 Settling $d$: the ADF test

**Purpose.** The ADF test formalises choosing $d$. For series $y_t$, it tests:

- $H_0$: $y_t$ has a unit root (non-stationary, random walk)
- $H_1$: $y_t$ is stationary

Applied to $y_t$, then $\nabla y_t$, then $\nabla^2 y_t$, $d$ is the number of differences before $H_0$ is first rejected. Limitations: the test addresses only unit root non-stationarity (not variance growth or mean shifts). Failure to reject $H_0$ means the data cannot distinguish a unit root from stationarity with the available sample.

**Intuition.** Write $y_t = \alpha + \phi y_{t-1} + \varepsilon_t$ and subtract $y_{t-1}$:

$$\nabla y_t = \alpha + \beta y_{t-1} + \varepsilon_t \qquad \beta = \phi - 1$$

The unit root $\phi = 1$ is now $\beta = 0$: $\nabla y_t$ does not depend on $y_{t-1}$, and the series is a random walk. Stationarity ($|\phi| < 1$) is $\beta < 0$. Taking expectations of the original equation: $\mu = \alpha + \phi\mu$, so $\alpha = (1-\phi)\mu = -\beta\mu$, giving:

$$\nabla y_t = \beta(y_{t-1} - \mu) + \varepsilon_t$$

Expected change is proportional to distance from mean. At $\beta = 0$ the pull vanishes and the series wanders. Hypotheses: $H_0:\ \beta = 0$, $H_1:\ \beta < 0$. The test is **one-sided** (negative $\beta$ only). The intercept $\alpha$ is required; omitting it forces reversion toward zero, misdescribing series with $\mu \ne 0$.

The "augmented" part of the name handles series that depend on more than one past value. If $y_t$ depends on $y_{t-1}, \dots, y_{t-p}$ (an AR($p$) model), the regression above omits lags $2, \dots, p$; their effect ends up in the error term, which is then autocorrelated rather than white, and the estimate of $\beta$ and its standard error are unreliable. The cure is to rewrite the model so that the unit root is again a single coefficient. For $p = 2$, $y_t = \alpha + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \varepsilon_t$; write $\phi_2 y_{t-2} = \phi_2 y_{t-1} - \phi_2 \nabla y_{t-1}$ and subtract $y_{t-1}$:

$$\nabla y_t = \alpha + \underbrace{(\phi_1 + \phi_2 - 1)}_{\beta}\,y_{t-1} \;\underbrace{-\;\phi_2}_{\delta_1}\,\nabla y_{t-1} + \varepsilon_t$$

For general $p$, the same manipulation gives:

$$\nabla y_t = \alpha + \beta y_{t-1} + \sum_{j=1}^{p-1}\delta_j\nabla y_{t-j} + \varepsilon_t \qquad \beta = -\phi(1) = -(1 - \phi_1 - \cdots - \phi_p)$$

The identity $\beta = -\phi(1)$ is key. The AR factoring argument shows a unit root exists exactly when $\phi(1) = 0$, i.e., $\beta = 0$. Whatever $p$, the question "unit root or not?" reduces to "is $\beta = 0$?". Lagged differences absorb remaining AR structure, so $\varepsilon_t$ is white and $\beta$ can be tested in isolation. The $\delta_j$ are nuisance parameters.

**Statistic and distribution.** The regression is estimated by OLS. The test statistic:

$$\tau = \frac{\hat\beta}{\mathrm{SE}(\hat\beta)}$$

The null distribution is non-standard. In standard regression, $t$-ratios are ~$N(0,1)$ in large samples, relying on regressors having stable variance. Under $H_0$ with $\alpha = 0$, $y_{t-1}$ is a random walk with variance $t\sigma^2$, growing without bound, so the approximation fails: $\tau$'s null is shifted left and non-normal.

Dickey and Fuller (1979) obtained the distribution by simulation; MacKinnon (1996) supplies response-surface approximations for p-values. Critical values are more negative: for intercept-only regression and large $T$, 5% critical value is ~$-2.86$ vs $-1.645$ for normal. Comparing $\tau$ with normal quantiles would reject $H_0$ too often (under-differencing). Critical values depend on deterministic terms (none, intercept, or intercept+trend), so regression terms and table must match.

**Procedure.** For series $x_t$ (initially $y_t$), with $T$ observations:

1. *Stabilise variance first.* If swings scale with level, log before testing. The regression assumes constant-variance shocks.
2. *Choose deterministic terms.* Always include intercept. Add linear trend $\gamma t$ only if testing for stationarity around a linear trend (this lesson uses intercept-only, as trends are differenced).
3. *Choose lag order $k$.* $k$ must ensure uncorrelated residuals (invalid p-values otherwise) and be no larger than necessary. Standard: $k_{\max} = \lfloor 12\,(T/100)^{1/4} \rfloor$ (Schwert's rule), then select $k \le k_{\max}$ by AIC over the fitted regressions. `statsmodels.tsa.stattools.adfuller(x, regression='c', autolag='AIC')` implements this, returning $\tau$, p-value, $k$ used, and critical values. Said and Dickey (1984) show validity for general ARMA provided $k$ grows with $T$.
4. *Decide.* Reject $H_0$ at 5% level if $\tau$ < critical value (p-value < 0.05).
5. *Iterate.* If $H_0$ rejected for $x_t = y_t$, set $d = 0$. Else difference, set $x_t = \nabla y_t$, repeat from step 3; rejection gives $d = 1$. One further round gives $d = 2$; stop. Each round discards one observation and inherits over-differencing costs (doubled variance, spurious $-0.5$ correlation), so retain the smallest $d$ with rejection.

**Limitations.** ADF has low power near the null: with $\phi = 0.95$ and $T = 100$, the test often fails to reject $H_0$ even though the series is stationary (slowly reverting series ≈ random walk over short windows). Therefore, failure to reject is read together with the plot, the ACF (slowly fading positive bars signal unit root), and the $-0.5$ lag-1 diagnostic from Part I (over-differencing). Also, a mean shift can cause failure to reject, mimicking a unit root; the plot exposes this. The test is one input for choosing $d$, with ACF/PACF diagnostics as others.

#### 7.3 The PACF

The **partial autocorrelation** at lag $k$ is the coefficient on $y_{t-k}$ in a regression of $y_t$ on $y_{t-1}, \dots, y_{t-k}$ jointly. It measures the **direct** contribution of lag $k$, after intermediate lags have absorbed routed effects.

Example: AR(1) PACF at lag 2. Regress $y_t$ on $y_{t-1}, y_{t-2}$. The model itself answers: $y_t = \phi y_{t-1} + 0\cdot y_{t-2} + \varepsilon_t$. Best coefficients: $(\phi, 0)$. So:

$$\mathrm{PACF}(1) = \phi \qquad \mathrm{PACF}(k) = 0 \ \text{ for } k \ge 2$$

The ACF at lag 2 is $\phi^2 \ne 0$ (Yule–Walker). ACF sees knock-on correlation; PACF asks for direct effects and finds none. Generalising: for AR($p$), regression on $p$ lags recovers the model exactly, so **PACF cliffs at $p$**. Conversely, MA(1) = AR($\infty$) with coefficient $-(-\theta)^k$ on $y_{t-k}$, nonzero at every lag, so **MA's PACF fades**.

The two tools are mirror images, and together they identify pure models:

| | ACF | PACF |
|---|---|---|
| AR($p$) | fades — mixture of geometric decays | **cliffs after $p$** |
| MA($q$) | **cliffs after $q$** | fades |
| ARMA($p,q$) | fades | fades |

The intuition for the mirror: ACF measures dependence on past *values*, which for AR runs through an infinite chain of intermediate values; PACF strips the chain out and sees only the $p$ direct links. For MA the roles swap — the dependence on past *shocks* stops at $q$, but each shock is itself an infinite combination of past values, so the direct links never stop.

#### 7.4 Reading sample plots: Bartlett's bands

You observe $\hat\rho_k$, not $\rho_k$ — a noisy average. Calibration (a result taken on faith): **if the series is white noise, then $\hat\rho_k \sim N(0, 1/T)$ approximately.** Standard deviation $1/\sqrt{T}$; by the 95% rule, true zero correlation lands inside $\pm 2/\sqrt{T}$ ~95% of the time. These are the dashed bands on every ACF/PACF plot. **"Cliffs" means "falls inside the bands"** — never "equals zero." Mind multiplicity: at 5% false-alarm rate, one spurious excursion among 20 lags is *expected*; a lone outlier at lag 7 is noise until theory gives a reason to expect it.

#### 7.5 When both fade: grid search plus AIC

Real ARMA processes make both plots fade; real data is messier. The pragmatic fallback: fit every $(p, q)$ with $p, q \le 3$, and score by:

$$\mathrm{AIC} = -2\ell + 2k$$

$\ell$ = maximised log-likelihood (measure of fit, higher is better); $k$ = parameter count. Penalty: adding a parameter can never lower $\ell$ (optimiser can set it to zero), so in-sample fit improves mechanically, even when fitting noise. $2k$ corrects "fit to past" to "expected fit to fresh data" (the factor 2 is taken on faith; the *need* for a penalty is proved here). Lowest AIC wins. **BIC** swaps $2k$ for $k\log T$ — heavier penalty, smaller models; on disagreement, BIC is conservative.

Two rules:
- **Compare AICs only at same $d$.** Differencing changes the dataset; likelihoods measure fit to different data, non-comparable.
- **Near-tie → smaller model.** Penalty is an estimate; one-point gap is noise, fewer parameters means less coefficient variance.

#### 7.6 Ljung–Box: the ACF plot as one number

Aggregating the first $m$ sample autocorrelations:

$$Q = T(T+2)\sum_{k=1}^{m}\frac{\hat\rho_k^2}{T-k}$$

A weighted sum of squared ACF bars. Under white-noise null, each $\hat\rho_k^2 \approx 1/T$ (Bartlett), so $Q$ stays modest; large $Q$ (small p-value) signals correlation somewhere in the first $m$ lags without defending any single bar. The weighting $T(T+2)/(T-k)$ is a finite-sample refinement, taken on faith. Main use: validation, applied to residuals.

#### 7.7 The earned diagnostic

Differenced series with $\hat\rho_1 \approx -0.5$ and little else: this is *exactly* the over-differencing signature from Part I — differencing white noise gives $\rho_1 = -1/2$, and differencing any stationary series pushes $\rho_1$ to $-(1-\phi)/2$. Fix: reduce $d$, not add an MA term. Fitting an MA(1) would require $\theta = -1$ (the MA(1) bound $|\rho_1| \le 1/2$ at its extreme), placing a root at $z = 1$ — on the unit circle. Non-invertible: unidentified and unestimable. A milder negative $\hat\rho_1$ after differencing deserves the same suspicion when the undifferenced ACF was already fading rather than flat.

**Procedure:** $d$ by plot + ADF, watching for $-0.5$ tell → read ACF/PACF cliffs against $\pm 2/\sqrt{T}$ for candidate $(p, q)$ → grid + AIC to adjudicate → smaller on near-ties → carry winner to fitting.

---

### 8. Fitting

**Principle:** write shocks as functions of parameters, minimise $\sum_t \varepsilon_t^2$ by setting gradients to zero — then derive the same result from probability.

#### 8.1 AR(1) by least squares, fully derived

With mean removed ($c = 0$), $\varepsilon_t = y_t - \phi y_{t-1}$. Define:

$$S(\phi) = \sum_{t=2}^{T}(y_t - \phi y_{t-1})^2$$

Differentiate: $\frac{dS}{d\phi} = -2\sum(y_t - \phi y_{t-1})y_{t-1}$. Set to zero:

$$\sum y_t y_{t-1} - \phi\sum y_{t-1}^2 = 0 \qquad\Longrightarrow\qquad \boxed{\hat\phi = \frac{\sum_{t=2}^{T} y_t y_{t-1}}{\sum_{t=2}^{T} y_{t-1}^2}}$$

Second-derivative: $S'' = 2\sum y_{t-1}^2 > 0$ — a minimum. Closed form, three array operations.

#### 8.2 Recognising the estimator

Set the sample ACF at lag 1 beside the box:

$$\hat\rho_1 = \frac{\sum_{t=2}^{T}(y_t-\bar y)(y_{t-1}-\bar y)}{\sum_{t=1}^{T}(y_t-\bar y)^2}$$

Same architecture: adjacent products over squares. Differences are end-effects: $\hat\rho_1$ subtracts $\bar y$ (washed out by prior mean removal), denominator runs over all $T$ terms vs $T-1$ — a discrepancy vanishing as $T$ grows. So **the least-squares estimator is the lag-1 sample autocorrelation**, up to vanishing edges: fitting AR(1) and reading the first ACF bar are the same. This is Yule–Walker $\rho_1 = \phi$ run backwards with sample estimates. For AR($p$): $\partial S/\partial\phi_i = 0$ yields the Yule–Walker system with sample correlations — $p$ linear equations, one matrix solve, closed form. **Pure AR estimation is algebra.** (With $c$, $\partial S/\partial c = 0$ adds one more linear equation — fitting a line through $(y_{t-1}, y_t)$ points.)

#### 8.3 MA terms: conditional sum of squares

For MA(1), $\varepsilon_t = y_t - \theta\varepsilon_{t-1}$, defined recursively. **Conditional sum of squares (CSS)**: assume $\varepsilon_0 = 0$, roll forward: $\varepsilon_1 = y_1$, $\varepsilon_2 = y_2 - \theta\varepsilon_1$, $\varepsilon_3 = y_3 - \theta\varepsilon_2$, etc. Unwind: $\varepsilon_t$ is a polynomial in $\theta$ of degree $t-1$. So $S(\theta) = \sum\varepsilon_t^2$ is high-degree, $dS/d\theta = 0$ is nonlinear, no closed form. Minimise numerically.

*The $\varepsilon_0 = 0$ assumption:* recursion vs truth shows startup error carries weight $(-\theta)^t$, decaying because **invertibility holds**. *Exact alternative:* Kalman filter averages over $\varepsilon_0$; it and CSS agree to several decimals on decent samples.

#### 8.4 The likelihood, built honestly

Now add $\varepsilon_t \sim N(0, \sigma^2)$, density $f(\varepsilon) = (2\pi\sigma^2)^{-1/2}e^{-\varepsilon^2/2\sigma^2}$. Three steps:

1. **Factorisation.** Joint density of all $T$ shocks is the product of individual densities — legitimate because uncorrelated normal variables are independent.
2. **Change of variables.** We observe $y$'s, not $\varepsilon$'s; recursions map data to shocks one-to-one given parameters, transformation carries density without distortion, so likelihood is the product of $f(\varepsilon_t(\phi, \theta))$.
3. **Logs.**

$$\ell(\phi, \theta, \sigma^2) = \sum_t \log f(\varepsilon_t) = -\frac{T}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_t \varepsilon_t(\phi, \theta)^2$$

$(\phi, \theta)$ appear only inside $S = \sum\varepsilon_t^2$, with a minus sign. **Maximising likelihood over dynamics parameters is minimising $S$** — same optimisation, probabilistic interpretation. (Caveat: exactly true for conditional setup; exact likelihood adds a small startup correction — Kalman story — which is why software defaults and pure CSS differ in the third decimal.)

#### 8.5 What the clothing buys

First, $\hat\sigma^2$ by gradient:

$$\frac{\partial\ell}{\partial\sigma^2} = -\frac{T}{2\sigma^2} + \frac{S}{2\sigma^4} = 0 \qquad\Longrightarrow\qquad \hat\sigma^2 = \frac{S}{T}$$

Average squared residual, used for forecast intervals. Second, **standard errors from curvature**: $\hat\phi$'s precision is read from how sharply $\ell$ peaks. Sharp peak → data strongly singles out one value → small SE; flat peak → many values near-equally likely → large SE. Formally $\mathrm{SE} \approx 1/\sqrt{-\ell''}$ at the maximum. AR(1) sanity check: curvature $\propto \sum y_{t-1}^2$, so more data and predictor spread both tighten the estimate — ordinary regression behaviour. $\ell$'s maximised value feeds AIC in model selection.

#### 8.6 Reading software output

Each coefficient ± SE with a p-value (P(estimate this far from zero | true coefficient = 0)). A coefficient within noise of zero is a spare part — drop it, refit, let AIC adjudicate. Plus $\hat\sigma^2$ and $\ell$.

---

### 9. Forecasting

#### 9.1 Conditional expectation and optimality

$E[X \mid \text{data}]$ is the expectation recomputed with observed information. It inherits linearity. Key rules: known values come out as themselves ($E[y_T \mid \text{data}] = y_T$); fresh shocks are unchanged ($E[\varepsilon_{T+j} \mid \text{data}] = 0$).

The forecast minimising $E[(y_{T+h} - a)^2 \mid \text{data}]$ is $a = E[y_{T+h} \mid \text{data}]$ — same as the mean minimising $E[(X-a)^2]$. **Take conditional expectations of the model equation.** Practical rules:

1. Future value $y_{T+j}$ → its conditional expectation (its own forecast $\hat y_{T+j}$).
2. Future shock $\varepsilon_{T+j}$ → **0** (evaluation rule).
3. Past shock $\varepsilon_{T-j}$ → known: reconstructed residual $\hat\varepsilon_{T-j}$ from CSS.

#### 9.2 Worked recursion: ARMA(1,1)

$y_t = \phi y_{t-1} + \varepsilon_t + \theta\varepsilon_{t-1}$. Condition on data at time $T$:

$$\hat y_{T+1} = \phi y_T + \theta \hat\varepsilon_T \qquad \text{(all known)}$$

$$\hat y_{T+2} = \phi \hat y_{T+1} + \underbrace{0}_{\varepsilon_{T+2}} + \theta\cdot\underbrace{0}_{\varepsilon_{T+1}} = \phi \hat y_{T+1}$$

Thereafter $\hat y_{T+h} = \phi \hat y_{T+h-1}$: MA memory exhausts after $q$ steps, then pure geometric decay toward the mean. **Every stationary model's long-horizon forecast is the mean.** Since $\phi^h = e^{h\ln\phi}$ and $\ln\phi \approx -(1-\phi)$ near $\phi = 1$, the signal decays like $e^{-h(1-\phi)}$: **$1/(1-\phi)$ is the time constant** — the horizon at which the forecast's deviation from the mean has shrunk to $1/e \approx 37\%$ of its starting size. (The half-life, where it has halved, is $\ln 2/(1-\phi) \approx 0.69/(1-\phi)$.) At $\phi = 0.9$ the time constant is 10 periods; by $h = 30$ the deviation is $0.9^{30} \approx 4\%$ of its start and you are announcing the mean.

#### 9.3 If d ≥ 1: integrate back

You modelled the *differenced* series; recover levels by cumulative summing: $\hat y_{T+h} = y_T + \hat z_{T+1} + \cdots + \hat z_{T+h}$. If the differenced forecasts settle at the differenced series' mean — $c$ for a pure MA, $c/(1-\phi_1-\cdots-\phi_p)$ in general — the level climbs by that amount per period indefinitely: a linear trend extrapolated. Random-walk case: all $\hat z = 0$, so forecast is **flat at $y_T$ forever** — unit root's meaning, and the naive benchmark for backtesting.

#### 9.4 The forecast error, exactly

Write the model in MA($\infty$) form at the future date:

$$y_{T+h} = \sum_{j=0}^{\infty}\psi_j\varepsilon_{T+h-j}$$

Split at $j = h$. Terms with $j \ge h$ involve shocks dated $\le T$ — known; conditional expectations kill terms with $j < h$ and keep these, so the known tail **is** the forecast. Subtract:

$$e_{T+h} = y_{T+h} - \hat y_{T+h} = \sum_{j=0}^{h-1}\psi_j\varepsilon_{T+h-j}$$

The forecast error is *exactly* the $h$ not-yet-happened shocks, each entering with its impulse weight.

#### 9.5 The interval

Uncorrelated shocks: variances add, constants square out (workhorse identity):

$$\boxed{\mathrm{Var}(e_{T+h}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2}$$

with $\hat\sigma^2 = S/T$. Sanity checks: $h = 1$ gives $\sigma^2$ (irreducible one-step noise, $\psi_0 = 1$). AR(1), $\psi_j = \phi^j$: variance approaches $\sigma^2/(1-\phi^2)$ — **fan saturates at series' variance**. Random walk, $\psi_j = 1$: variance $h\sigma^2$, unbounded. Normal shocks make $e_{T+h}$ normal, so 95% interval:

$$\hat y_{T+h} \pm 2\sqrt{\mathrm{Var}(e_{T+h})}$$

**Report the interval.** A point forecast without its fan hides this quantity.

When $d = 1$, errors compound: level error is cumulative sum of differenced-series errors, so you cannot integrate interval endpoints — integrate errors and re-derive variance. Equivalent: undifferenced model has $\psi$-weights from $\theta(B)/[\phi(B)(1-B)]$ — $(1-B)$ in denominator adds non-decaying random-walk component, which is *why* weights do not die and $d \ge 1$ fans widen without bound. Software handles this; the concept is the boxed formula with the right weights.

---

### 10. Validation

#### 10.1 Residual autopsy

If the model captured the structure, residuals $\hat\varepsilon_t$ should be white noise — "nothing left to model." Test with identification tools: ACF inside $\pm 2/\sqrt{T}$ (Bartlett's bands), Ljung–Box on first $m$ residual correlations returning comfortable p-value. Small p → correlation survives → revisit $(p, q)$. Mind p-value asymmetry: large p-value means *not caught*, never *proved correct*.

#### 10.2 Backtest

In-sample diagnostics certify the past; *forecasting* requires out-of-sample evidence. Rolling-origin backtest: pick an origin $n < T$; fit on $y_1, \dots, y_n$; forecast $n+1$ to $n+h$ and record the errors against the held-out $y_{n+1}, \dots, y_{n+h}$; slide the origin forward; repeat; average. Benchmark: naive forecast $\hat y_{n+h} = y_n$ — *optimal under a random walk*. Beating it means the series has exploitable structure beyond a unit root. Exchange rates are the standard example of a series on which the naive forecast cannot be beaten: the machinery detects structure but cannot conjure it.

---

### 11. Implementation notes

The exercises use plain array libraries (NumPy). The boxed AR(1) least-squares estimator is three array operations; simulation, sample ACF, and residuals are equally direct. Only CSS minimisation for MA terms needs a numerical optimiser — write the objective (recursion is a short loop) and hand to `scipy.optimize.minimize`; the concept is in the objective, not the optimiser. Keep a full ARIMA implementation (`statsmodels`) as the answer key only. Deep-learning frameworks offer no advantage: no learned representation, framework overhead would drown a ~30-line exercise.

### 12. ARIMA estimation in a nutshell

```text
ARIMA_ESTIMATION(y, max_p = 3, max_q = 3, max_d = 2, h = 0, TOL = 1):
    // Estimate an ARIMA(p,d,q) model for series y and forecast h steps ahead.
    // TOL is the AIC gap treated as a near-tie, resolved in favour of the smaller model.
    // T(x) denotes the length of series x; S(model) the sum of squared residuals.
    m = 20                               // number of residual lags aggregated by Ljung-Box

    // STEP 1: Preprocess - stabilise variance
    logged = swings_scale_with_level(y)
    y_work = logged ? log(y) : y

    // STEP 2: Determine integration order d - remove unit roots
    z = y_work
    d = 0
    while d < max_d and ADF_test(z, regression = 'c', autolag = 'AIC') fails to reject:
        z = difference(z)
        d = d + 1

    // STEPS 3-5: Identify, fit, validate (loop until residuals are white noise)
    rejected = { }
    repeat:
        // STEP 3: Select orders (p,q)
        bands = ±2 / sqrt(T(z))          // Bartlett's bands
        acf = sample_ACF(z)
        pacf = sample_PACF(z)
        if d > 0 and acf.lag(1) ≈ -0.5 and acf beyond lag 1 inside bands:
            return failure("over-differenced: reduce d")

        // Fingerprints name the orders to expect: PACF cliff at p → AR(p),
        // ACF cliff at q → MA(q), both fading → mixed. The grid adjudicates by AIC.
        candidates = { (i,j) : 0 <= i <= max_p, 0 <= j <= max_q } - rejected

        if candidates is empty:
            return failure("no candidate order survived validation")

        // Select best model by AIC
        best_aic = infinity
        (p, q) = none
        for (i, j) in candidates, ordered by increasing i + j:
            aic = AIC(fit_ARMA(z, i, j))
            if aic < best_aic - TOL:
                best_aic = aic
                (p, q) = (i, j)

        // STEP 4: Fit ARIMA(p,d,q) to y_work; the model applies the d differences itself
        model = ARIMA(y_work, order = (p,d,q), constant = (d <= 1))   // constant = drift; omitted at d = 2
        residuals = model.residuals()
        sigma_squared = S(model) / T(residuals)

        // STEP 5: Validate - check residuals have no structure
        // Ljung-Box degrees of freedom are reduced by the p + q fitted ARMA coefficients
        acf_ok = ACF(residuals) inside ±2 / sqrt(T(residuals))
        ljung_box_ok = Ljung_Box(residuals, m, dof = m - p - q).p > 0.05

        if acf_ok and ljung_box_ok:
            status = "VALID"
            break
        rejected += (p, q)

    // STEP 6: Backtest - compare against naive forecast
    skill = rolling_origin_backtest(y_work, order = (p,d,q))
    naive = rolling_origin_backtest_naive(y_work)
    if skill does not beat naive:
        status = "VALID BUT NOT USEFUL"

    // STEP 7: Forecast
    forecasts = none
    intervals = none
    if h > 0:
        forecasts = model.forecast(horizon = h)
        intervals = forecasts ± 2 * sqrt(forecast_variance(model, sigma_squared, h))
        if logged:
            intervals = exp(intervals)
            forecasts = exp(forecasts)

    return model, (p, d, q), status, forecasts, intervals
```

Note: Step 2 requires judgment — ADF has low power near the null and may misread mean shifts as unit roots. Always examine the plot and ACF alongside the test. A lag-1 ACF near $-0.5$ after differencing means $d$ is too large, not that ARIMA is inappropriate. If residuals remain correlated for every $(p, q)$ candidate, the series has structure — seasonality, a level shift, changing variance — that no ARIMA of these orders captures.

---

## Part III — Exercises

Work through the exercises in order; each uses results from the previous ones.

1. **Simulate and recover.** Simulate $y_t = 0.7\,y_{t-1} + \varepsilon_t$ for $T = 500$ with normal shocks (a loop). Estimate $\hat\phi$ with the boxed least-squares formula; confirm it lands near $0.7$ — and near $\hat\rho_1$, as the estimator's identification with the lag-1 sample autocorrelation predicts. Repeat for $T = 50$ and observe the spread across several runs.

2. **Residual check.** Compute residuals $y_t - \hat\phi\,y_{t-1}$ from Exercise 1 and verify their sample ACF sits inside $\pm 2/\sqrt{500} \approx \pm 0.09$ (Bartlett's bands). Count the excursions across the first 20 lags and compare with the expected false-alarm rate.

3. **Over-differencing.** Difference the (already stationary) series of Exercise 1 and compute the sample ACF and variance of the result. Confirm the lag-1 value near $-(1-\phi)/2 = -0.15$ and the variance near $2(1-\phi) = 0.6$ times the original, as derived for differencing a stationary series. Then difference a white-noise series of the same length and confirm $-0.5$ and a doubling. Explain why "add an MA(1)" is the wrong repair in the second case, and why the first case is the harder one to notice.

4. **MA by CSS.** Simulate an MA(1) with $\theta = 0.6$. Implement the CSS recursion to compute $S(\theta)$ on a grid of $\theta$ values and plot it; confirm the minimum sits near $0.6$ and that the curve is not a parabola. Then minimise with a numerical optimiser and compare against a library ARIMA(0,0,1) fit.

5. **The fan.** For the fitted AR(1) of Exercise 1, compute $\mathrm{Var}(e_{T+h})$ from the forecast-interval formula for $h = 1, \dots, 30$ and verify the saturation at $\hat\sigma^2/(1-\hat\phi^2)$. Simulate 1000 continuations of the series and check that the empirical 95% coverage of the intervals is near nominal.

6. **Backtest.** On a real series of your choosing (after the log decision and the identification of $p, d, q$), run the rolling-origin backtest of the validation chapter against the naive forecast. Report the horizon at which the fitted model stops beating the baseline, and compare it with the time constant $1/(1-\hat\phi)$ from the forecast recursion.

---

## References

Nothing below is required: the derivations and the exercises above stand on their own. These are the sources for every result cited by name, for anyone who wants the original treatment.

**Cited in the text**

- Yule, G. U. (1927). *On a method of investigating periodicities in disturbed series, with special reference to Wolfer's sunspot numbers.* Philosophical Transactions of the Royal Society A 226, 267–298. The origin of autoregression and of the Yule–Walker equations. [doi:10.1098/rsta.1927.0007](https://doi.org/10.1098/rsta.1927.0007)
- Walker, G. (1931). *On periodicity in series of related terms.* Proceedings of the Royal Society A 131, 518–532. The other half of the Yule–Walker name. [doi:10.1098/rspa.1931.0069](https://doi.org/10.1098/rspa.1931.0069)
- Dickey, D. A. and Fuller, W. A. (1979). *Distribution of the estimators for autoregressive time series with a unit root.* Journal of the American Statistical Association 74(366), 427–431. The unit-root test used to settle $d$, and its non-standard critical values. [doi:10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Said, S. E. and Dickey, D. A. (1984). *Testing for unit roots in autoregressive-moving average models of unknown order.* Biometrika 71(3), 599–607. The augmented form of the ADF test and the validity of a data-chosen lag order. [doi:10.1093/biomet/71.3.599](https://doi.org/10.1093/biomet/71.3.599)
- MacKinnon, J. G. (1996). *Numerical distribution functions for unit root and cointegration tests.* Journal of Applied Econometrics 11(6), 601–618. The p-values and finite-sample critical values reported by software for the ADF test. [doi:10.1002/(SICI)1099-1255(199611)11:6\<601::AID-JAE417\>3.0.CO;2-T](https://doi.org/10.1002/(SICI)1099-1255(199611)11:6%3C601::AID-JAE417%3E3.0.CO;2-T)
- Bartlett, M. S. (1946). *On the theoretical specification and sampling properties of autocorrelated time-series.* Supplement to the Journal of the Royal Statistical Society 8(1), 27–41. The $\hat\rho_k \sim N(0, 1/T)$ approximation behind the significance bands of the ACF/PACF plots. [doi:10.2307/2983611](https://doi.org/10.2307/2983611)
- Akaike, H. (1974). *A new look at the statistical model identification.* IEEE Transactions on Automatic Control 19(6), 716–723. The criterion used to select $p$ and $q$. [doi:10.1109/TAC.1974.1100705](https://doi.org/10.1109/TAC.1974.1100705)
- Schwarz, G. (1978). *Estimating the dimension of a model.* Annals of Statistics 6(2), 461–464. BIC, the AIC's sibling in model selection. [doi:10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136)
- Ljung, G. M. and Box, G. E. P. (1978). *On a measure of lack of fit in time series models.* Biometrika 65(2), 297–303. The Ljung–Box statistic, including the finite-sample weighting taken on faith. [doi:10.1093/biomet/65.2.297](https://doi.org/10.1093/biomet/65.2.297)
- Kalman, R. E. (1960). *A new approach to linear filtering and prediction problems.* Journal of Basic Engineering 82(1), 35–45. The filter behind exact-likelihood fitting, which averages over the unknown starting shocks instead of setting them to zero. [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552)

**Background, not cited**

- Box, G. E. P. and Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day. The book that assembled Part II's steps into the identify–estimate–diagnose–forecast cycle; the whole methodology is often called Box–Jenkins. Current edition: Box, Jenkins, Reinsel and Ljung, [5th ed., Wiley, 2015](https://www.wiley.com/en-us/Time+Series+Analysis:+Forecasting+and+Control,+5th+Edition-p-9781118675021).
- Hamilton, J. D. (1994). *Time Series Analysis.* Princeton University Press. The standard graduate reference; chapters 1–5 cover this lesson's material with full rigour, including the proofs taken on faith here. [publisher page](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis)
- Brockwell, P. J. and Davis, R. A. (2016). *Introduction to Time Series and Forecasting.* 3rd ed., Springer. A gentler book-length treatment at the level of this lesson. [doi:10.1007/978-3-319-29854-2](https://doi.org/10.1007/978-3-319-29854-2)
- Hyndman, R. J. and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice.* 3rd ed., OTexts. The applied companion — identification, backtesting and software workflow; the full text is [free online](https://otexts.com/fpp3/).

---

## Licence

The prose of this lesson is licensed under [Creative Commons Attribution 4.0 International](LICENSE)
(CC BY 4.0): copy, adapt, translate and redistribute it, including commercially, provided you give
credit and indicate what you changed. The code samples it contains are licensed under Apache-2.0,
as is all code in [this repository](../../../LICENSE).
