# Volatility Models and Stochastic Processes

This document covers QuantLib's volatility modeling, stochastic processes, and pricing models for derivatives.

## Volatility Term Structures

Volatility term structures model how implied volatility varies across strikes and expiration dates.

### BlackVolTermStructure (Base Class)

Base class for all Black volatility surfaces.

```python
# Common interface for all volatility surfaces
vol_surface = ...  # Any concrete implementation

# Black volatility lookup
vol = vol_surface.blackVol(expiry, strike)          # Time and strike
vol = vol_surface.blackVol(expiry_date, strike)     # Date and strike

# Black variance
variance = vol_surface.blackVariance(expiry, strike)
variance = vol_surface.blackVariance(expiry_date, strike)

# Variance by time and moneyness
forward_variance = vol_surface.blackForwardVariance(t1, t2, strike)

# Surface properties
vol_surface.referenceDate()     # Surface anchor date
vol_surface.maxDate()          # Maximum date
vol_surface.minStrike()        # Minimum strike (if applicable)
vol_surface.maxStrike()        # Maximum strike (if applicable)
```

### BlackConstantVol

Flat volatility surface (constant across strikes and time).

```python
# Constructors
BlackConstantVol(
    reference_date,     # or settlement_days
    calendar,
    volatility,         # Constant volatility level
    day_counter
)

# Examples
flat_vol = ql.BlackConstantVol(
    referenceDate=ql.Date(15, 6, 2023),
    calendar=ql.TARGET(),
    volatility=0.25,    # 25% volatility
    dayCounter=ql.Actual365Fixed()
)

# From quote handle (dynamic volatility)
vol_quote = ql.SimpleQuote(0.25)
dynamic_flat_vol = ql.BlackConstantVol(
    settlementDays=2,
    calendar=ql.TARGET(),
    volatility=ql.QuoteHandle(vol_quote),
    dayCounter=ql.Actual365Fixed()
)

# Update volatility
vol_quote.setValue(0.30)  # Surface automatically updates
```

### BlackVarianceSurface

Volatility surface from discrete market data points.

```python
# Market data setup
expiry_dates = [
    ql.Date(15, 9, 2023),   # 3M
    ql.Date(15, 12, 2023),  # 6M
    ql.Date(15, 6, 2024),   # 1Y
    ql.Date(15, 6, 2025)    # 2Y
]

strikes = [80, 90, 100, 110, 120]  # Strike levels

# Market volatilities (rows = expiries, columns = strikes)
vol_matrix = ql.Matrix(len(expiry_dates), len(strikes))
market_vols = [
    [0.35, 0.30, 0.25, 0.30, 0.35],  # 3M expiry
    [0.33, 0.28, 0.24, 0.28, 0.33],  # 6M expiry  
    [0.32, 0.27, 0.23, 0.27, 0.32],  # 1Y expiry
    [0.31, 0.26, 0.22, 0.26, 0.31]   # 2Y expiry
]

for i, vol_row in enumerate(market_vols):
    for j, vol in enumerate(vol_row):
        vol_matrix[i][j] = vol

# Create surface
vol_surface = ql.BlackVarianceSurface(
    referenceDate=ql.Date(15, 6, 2023),
    calendar=ql.TARGET(),
    dates=expiry_dates,
    strikes=strikes,
    blackVolMatrix=vol_matrix,
    dayCounter=ql.Actual365Fixed()
)

# Interpolation methods
vol_surface.setInterpolation("bilinear")  # Default
# Other options: "bicubic"

# Enable extrapolation
vol_surface.enableExtrapolation()
```

### BlackVolCube

3D volatility cube for options with additional dimensions (e.g., swaptions).

```python
# For swaptions: volatility depends on option expiry, swap tenor, and strike
atmVol_surface = ...        # ATM volatility surface
vol_spreads = ...           # Volatility spreads
strike_spreads = ...        # Strike spreads

vol_cube = ql.SwaptionVolCube(
    atmVolStructure=ql.Handle(atmVol_surface),
    optionTenors=[ql.Period("1Y"), ql.Period("2Y")],
    swapTenors=[ql.Period("5Y"), ql.Period("10Y")],
    strikeSpreads=strike_spreads,
    volSpreads=vol_spreads,
    swapIndexBase=swap_index,
    shortSwapIndexBase=short_swap_index,
    vegaWeightedSmileFit=False
)
```

## Optionlet Volatilities

For cap/floor and other interest rate options.

### ConstantOptionletVolatility

Flat optionlet volatility.

```python
constant_caplet_vol = ql.ConstantOptionletVolatility(
    settlementDays=2,
    calendar=ql.TARGET(),
    businessDayConvention=ql.ModifiedFollowing,
    volatility=0.20,        # 20% vol for all caplets
    dayCounter=ql.Actual365Fixed()
)

# From quote handle
vol_quote = ql.QuoteHandle(ql.SimpleQuote(0.20))
dynamic_caplet_vol = ql.ConstantOptionletVolatility(
    settlementDays=2,
    calendar=ql.TARGET(),
    businessDayConvention=ql.ModifiedFollowing,
    volatility=vol_quote,
    dayCounter=ql.Actual365Fixed()
)
```

### OptionletStripper

Extract optionlet volatilities from cap/floor market prices.

```python
# Market cap/floor prices
cap_floor_prices = [...]  # Market prices
cap_floor_vols = [...]    # Market volatilities  

optionlet_stripper = ql.OptionletStripper1(
    termVolSurface=cap_floor_vol_surface,
    iborIndex=euribor_6m,
    switchStrike=None,      # ATM strike switch
    accuracy=1e-6,
    maxIter=100,
    volatilityType=ql.Normal,  # or ql.ShiftedLognormal
    displacement=0.0
)

# Extracted optionlet surface
optionlet_surface = optionlet_stripper.optionletFixingTenors()
```

## Swaption Volatilities

Volatility structures for swaptions (options on interest rate swaps).

### ConstantSwaptionVolatility

Flat swaption volatility.

```python
constant_swaption_vol = ql.ConstantSwaptionVolatility(
    settlementDays=2,
    calendar=ql.TARGET(),
    businessDayConvention=ql.ModifiedFollowing,
    volatility=0.15,        # 15% swaption volatility
    dayCounter=ql.Actual365Fixed()
)
```

### SwaptionVolatilityMatrix

Swaption volatility from discrete market points.

```python
# Market structure
option_tenors = [ql.Period("1Y"), ql.Period("2Y"), ql.Period("5Y")]  # Option expiries
swap_tenors = [ql.Period("2Y"), ql.Period("5Y"), ql.Period("10Y")]   # Underlying swap tenors

# Volatility matrix (rows = option tenors, columns = swap tenors)
swaption_vol_matrix = ql.Matrix(len(option_tenors), len(swap_tenors))
market_swaption_vols = [
    [0.18, 0.16, 0.15],     # 1Y into 2Y, 5Y, 10Y swaps
    [0.17, 0.15, 0.14],     # 2Y into 2Y, 5Y, 10Y swaps  
    [0.16, 0.14, 0.13]      # 5Y into 2Y, 5Y, 10Y swaps
]

for i, vol_row in enumerate(market_swaption_vols):
    for j, vol in enumerate(vol_row):
        swaption_vol_matrix[i][j] = vol

# Create swaption volatility surface
swaption_vol_surface = ql.SwaptionVolatilityMatrix(
    calendar=ql.TARGET(),
    businessDayConvention=ql.ModifiedFollowing,
    optionTenors=option_tenors,
    swapTenors=swap_tenors,
    vols=swaption_vol_matrix,
    dayCounter=ql.Actual365Fixed()
)
```

## Stochastic Processes

Model the random evolution of underlying assets and rates.

### BlackScholesProcess

Geometric Brownian motion for equity/FX modeling.

```python
# Market data handles
spot_handle = ql.QuoteHandle(ql.SimpleQuote(100.0))        # Current spot price
risk_free_handle = ql.YieldTermStructureHandle(            # Risk-free curve
    ql.FlatForward(0, ql.TARGET(), 0.05, ql.Actual365Fixed())
)
dividend_handle = ql.YieldTermStructureHandle(             # Dividend curve
    ql.FlatForward(0, ql.TARGET(), 0.02, ql.Actual365Fixed())
)
vol_handle = ql.BlackVolTermStructureHandle(               # Volatility surface
    ql.BlackConstantVol(0, ql.TARGET(), 0.25, ql.Actual365Fixed())
)

# Black-Scholes process: dS = (r-q)S dt + σS dW
bs_process = ql.BlackScholesMertonProcess(
    x0=spot_handle,         # Initial spot
    dividendTS=dividend_handle,
    riskFreeTS=risk_free_handle,
    blackVolTS=vol_handle
)

# Process methods
current_spot = bs_process.x0()                    # Current value
drift = bs_process.drift(time=1.0, x=100.0)      # Drift at time t, level x
diffusion = bs_process.diffusion(time=1.0, x=100.0)  # Diffusion coefficient
variance = bs_process.variance(time=1.0, x=100.0, dt=0.1)  # Variance over dt
```

### GeometricBrownianMotionProcess

Simplified GBM without dividend yield.

```python
gbm_process = ql.GeometricBrownianMotionProcess(
    initialValue=100.0,
    mu=0.08,            # Drift
    sigma=0.25          # Volatility
)
```

### Heston Stochastic Volatility Process

Model with stochastic volatility: dS = rS dt + √v S dW₁, dv = κ(θ-v)dt + σᵥ√v dW₂

```python
# Heston model parameters
heston_process = ql.HestonProcess(
    riskFreeTS=risk_free_handle,
    dividendTS=dividend_handle,
    s0=spot_handle,
    v0=0.06,            # Initial variance (25%² = 0.0625)
    kappa=2.0,          # Mean reversion speed
    theta=0.06,         # Long-term variance
    sigma=0.3,          # Vol of vol
    rho=-0.5            # Correlation between asset and vol
)

# Heston model (for calibration)
heston_model = ql.HestonModel(heston_process)
```

### Bates Jump-Diffusion Process

Heston with jumps in the underlying.

```python
bates_process = ql.BatesProcess(
    riskFreeTS=risk_free_handle,
    dividendTS=dividend_handle,
    s0=spot_handle,
    v0=0.06,            # Initial variance
    kappa=2.0,          # Vol mean reversion
    theta=0.06,         # Long-term variance  
    sigma=0.3,          # Vol of vol
    rho=-0.5,           # Correlation
    lambda_=0.1,        # Jump intensity (10% per year)
    nu=-0.1,            # Jump size mean
    delta=0.16          # Jump size standard deviation
)
```

## Interest Rate Models

Models for interest rate dynamics and fixed income derivatives.

### Short Rate Models

Model the instantaneous short rate process.

#### Vasicek Model

Mean-reverting Gaussian short rate: dr = κ(θ-r)dt + σ dW

```python
vasicek_model = ql.Vasicek(
    a=0.1,              # Mean reversion speed
    b=0.05,             # Long-term mean
    sigma=0.015,        # Volatility
    r0=0.04             # Initial short rate
)

# Model methods
vasicek_model.params()          # Model parameters
vasicek_model.a()              # Mean reversion
vasicek_model.b()              # Long-term mean  
vasicek_model.sigma()          # Volatility

# Analytical properties
discount_bond_price = vasicek_model.discountBond(
    now=0.0,
    maturity=1.0,
    rate=0.04
)
```

#### Hull-White Model

Extended Vasicek with time-dependent parameters, fitted to initial yield curve.

```python
# Hull-White model fitted to term structure
hw_model = ql.HullWhite(
    termStructure=risk_free_handle,
    a=0.1,              # Mean reversion (can be time-dependent)
    sigma=0.015         # Volatility (can be time-dependent)
)

# Two-factor Hull-White
hw2_model = ql.HullWhiteForwardProcess(
    riskFreeTS=risk_free_handle,
    a=0.1,
    sigma=0.015
)

# Calibration to market instruments
calibration_helpers = [...]  # Swaption helpers
calibration_method = ql.LevenbergMarquardt()
end_criteria = ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)

hw_model.calibrate(calibration_helpers, calibration_method, end_criteria)
```

#### Cox-Ingersoll-Ross (CIR) Model

Square-root process: dr = κ(θ-r)dt + σ√r dW

```python
cir_model = ql.CoxIngersollRoss(
    theta=0.05,         # Long-term mean
    k=0.1,              # Mean reversion speed
    sigma=0.015,        # Volatility
    r0=0.04             # Initial rate
)

# CIR ensures non-negative rates if 2κθ ≥ σ² (Feller condition)
feller_condition = 2 * cir_model.k() * cir_model.theta() >= cir_model.sigma()**2
```

#### Black-Karasinski Model

Log-normal short rate model.

```python
bk_model = ql.BlackKarasinski(
    termStructure=risk_free_handle,
    a=0.1,              # Mean reversion
    sigma=0.12          # Volatility
)
```

### Multi-Factor Models

#### Two-Factor Additive Models

G2++ model: two correlated factors.

```python
g2_model = ql.G2(
    termStructure=risk_free_handle,
    a=0.1,              # First factor mean reversion
    sigma=0.015,        # First factor volatility  
    b=0.2,              # Second factor mean reversion
    eta=0.02,           # Second factor volatility
    rho=-0.3            # Correlation between factors
)
```

### Market Models

LIBOR and swap market models.

#### LIBOR Market Model (LMM)

Models evolution of forward LIBOR rates.

```python
# Set up LIBOR market model
libor_periods = [...]  # Forward rate periods
correlations = ql.Matrix(n, n)  # Correlation matrix
volatilities = [...]   # Volatility functions

lmm_process = ql.LiborForwardModelProcess(
    size=len(libor_periods),
    index=euribor_6m
)

# Displaced-diffusion LMM
dd_lmm = ql.LiborForwardModel(
    process=lmm_process,
    volaModel=volatility_model,
    corrModel=correlation_model
)
```

## Model Calibration

Fit model parameters to market prices.

### Calibration Infrastructure

```python
# Calibration helper example (swaptions for Hull-White)
swaption_vols = [0.15, 0.14, 0.13]  # Market volatilities
swaption_helpers = []

for i, vol in enumerate(swaption_vols):
    tenor = ql.Period(f"{i+1}Y")
    helper = ql.SwaptionHelper(
        maturity=tenor,
        length=ql.Period("5Y"),
        volatility=ql.QuoteHandle(ql.SimpleQuote(vol)),
        index=euribor_6m,
        fixedLegTenor=ql.Period("1Y"),
        fixedLegDayCounter=ql.Thirty360(),
        floatingLegDayCounter=ql.Actual360(),
        termStructure=risk_free_handle
    )
    swaption_helpers.append(helper)

# Calibrate Hull-White model
hw_model = ql.HullWhite(risk_free_handle)
calibration_method = ql.LevenbergMarquardt()
end_criteria = ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)

# Perform calibration
hw_model.calibrate(swaption_helpers, calibration_method, end_criteria)

# Check calibration results
calibrated_a = hw_model.params()[0]      # Mean reversion
calibrated_sigma = hw_model.params()[1]  # Volatility

print(f"Calibrated a: {calibrated_a:.6f}")
print(f"Calibrated sigma: {calibrated_sigma:.6f}")

# Calibration quality
for i, helper in enumerate(swaption_helpers):
    market_value = helper.marketValue()
    model_value = helper.modelValue()
    error = abs(market_value - model_value)
    print(f"Helper {i}: Market {market_value:.6f}, Model {model_value:.6f}, Error {error:.6f}")
```

### Heston Model Calibration

```python
# Market option data for Heston calibration
strikes = [80, 90, 100, 110, 120]
expiries = [0.25, 0.5, 1.0]
market_prices = [...]  # Market option prices

# Create calibration helpers
heston_helpers = []
for expiry in expiries:
    for strike in strikes:
        # Market price lookup (simplified)
        market_price = ...  # Get market price for this strike/expiry
        
        helper = ql.HestonModelHelper(
            maturity=ql.Period(f"{int(expiry*12)}M"),
            calendar=ql.TARGET(),
            s0=100.0,
            strikePrice=strike,
            marketValue=ql.QuoteHandle(ql.SimpleQuote(market_price)),
            riskFreeRate=risk_free_handle,
            dividendRate=dividend_handle,
            impliedVolType=ql.BlackVolTermStructure.impliedVolatility
        )
        heston_helpers.append(helper)

# Initial Heston model
initial_heston_process = ql.HestonProcess(
    risk_free_handle, dividend_handle, spot_handle,
    v0=0.06, kappa=1.0, theta=0.06, sigma=0.3, rho=-0.5
)
heston_model = ql.HestonModel(initial_heston_process)

# Calibration  
optimizer = ql.LevenbergMarquardt()
end_criteria = ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)

heston_model.calibrate(heston_helpers, optimizer, end_criteria)

# Extract calibrated parameters
calibrated_params = heston_model.params()
print(f"Calibrated v0: {calibrated_params[0]:.6f}")
print(f"Calibrated kappa: {calibrated_params[1]:.6f}")
print(f"Calibrated theta: {calibrated_params[2]:.6f}")
print(f"Calibrated sigma: {calibrated_params[3]:.6f}")
print(f"Calibrated rho: {calibrated_params[4]:.6f}")
```

## Monte Carlo Simulation

Generate sample paths for stochastic processes.

### Path Generation

```python
# Set up Monte Carlo simulation
time_steps = 252    # Daily steps for 1 year
length = 1.0        # 1 year simulation
n_paths = 10000

# Random number generator
rng = ql.MersenneTwisterUniformRsg(1, 12345)
gaussian_rng = ql.BoxMullerGaussianRsg(rng)

# Path generator for Black-Scholes
path_generator = ql.PathGenerator(
    process=bs_process,
    length=length,
    timeSteps=time_steps,
    rsg=gaussian_rng,
    brownianBridge=False
)

# Generate paths
paths = []
for i in range(n_paths):
    sample = path_generator.next()
    path = sample.value()
    
    # Extract path values
    path_values = []
    for j in range(path.size()):
        path_values.append(path[j])
    
    paths.append(path_values)

# Analyze paths
final_values = [path[-1] for path in paths]
mean_final_value = sum(final_values) / len(final_values)
print(f"Mean final value: {mean_final_value:.2f}")
```

### Multi-Asset Simulation

```python
# Correlated asset simulation
n_assets = 2
correlation_matrix = ql.Matrix(n_assets, n_assets)
correlation_matrix[0][0] = correlation_matrix[1][1] = 1.0
correlation_matrix[0][1] = correlation_matrix[1][0] = 0.6  # 60% correlation

# Multi-dimensional random number generator  
multi_rng = ql.MersenneTwisterUniformRsg(n_assets, 12345)
multi_gaussian = ql.BoxMullerGaussianRsg(multi_rng)

# Create correlated processes (simplified example)
# In practice, use StochasticProcessArray for proper correlation
```

## Usage Examples

### Volatility Surface Construction and Analysis

```python
import QuantLib as ql
import numpy as np

# Market volatility data
strikes = [70, 80, 90, 100, 110, 120, 130]
expiry_days = [30, 60, 90, 180, 365]
expiry_dates = [ql.Date.todaysDate() + ql.Period(d, ql.Days) for d in expiry_days]

# Market volatility smile (volatility increases away from ATM)
import numpy as np

def vol_smile(strike, expiry_years, atm_vol=0.25):
    moneyness = np.log(strike / 100.0)  # ATM = 100
    time_factor = 0.1 * np.sqrt(expiry_years)  # Vol term structure
    smile = atm_vol + 0.15 * moneyness**2 + time_factor  # Convex smile
    return max(smile, 0.05)  # Floor at 5%

# Build volatility matrix
vol_matrix = ql.Matrix(len(expiry_dates), len(strikes))
for i, expiry_date in enumerate(expiry_dates):
    expiry_years = ql.Actual365Fixed().yearFraction(ql.Date.todaysDate(), expiry_date)
    for j, strike in enumerate(strikes):
        vol = vol_smile(strike, expiry_years)
        vol_matrix[i][j] = vol

# Create volatility surface
vol_surface = ql.BlackVarianceSurface(
    referenceDate=ql.Date.todaysDate(),
    calendar=ql.TARGET(),
    dates=expiry_dates,
    strikes=strikes,
    blackVolMatrix=vol_matrix,
    dayCounter=ql.Actual365Fixed()
)
vol_surface.enableExtrapolation()

# Analyze volatility surface
print("Volatility Surface Analysis:")
print("Strike\\Days", "\t".join(f"{d:3d}" for d in expiry_days))

for i, strike in enumerate(strikes):
    row = f"{strike:6.1f}"
    for j, expiry_date in enumerate(expiry_dates):
        vol = vol_surface.blackVol(expiry_date, strike)
        row += f"\t{vol:.3f}"
    print(row)

# ATM volatility term structure
print("\nATM Volatility Term Structure:")
atm_strike = 100.0
for expiry_date in expiry_dates:
    vol = vol_surface.blackVol(expiry_date, atm_strike)
    days = (expiry_date - ql.Date.todaysDate())
    print(f"{days} days: {vol:.3f}")
```

### Hull-White Interest Rate Simulation

```python
# Hull-White model simulation for bond portfolio
ql.Settings.instance().evaluationDate = ql.Date(15, 6, 2023)

# Market yield curve
market_rates = [0.02, 0.025, 0.03, 0.035, 0.04]
market_times = [0.25, 0.5, 1.0, 2.0, 5.0]
yield_curve = ql.ZeroCurve(
    [ql.Date.todaysDate() + ql.Period(int(t*365), ql.Days) for t in market_times],
    market_rates,
    ql.Actual365Fixed(),
    ql.TARGET()
)

# Hull-White model
hw_model = ql.HullWhite(ql.YieldTermStructureHandle(yield_curve), a=0.1, sigma=0.015)

# Create bond to value under different scenarios
bond_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("6M"),
    calendar=ql.TARGET()
)

bond = ql.FixedRateBond(
    settlementDays=2,
    faceAmount=100.0,
    schedule=bond_schedule,
    coupons=[0.04],  # 4% coupon
    accrualDayCounter=ql.ActualActual()
)

# Simulate interest rate scenarios
n_scenarios = 1000
time_horizon = 1.0  # 1 year
time_steps = 12     # Monthly steps

# Set up Monte Carlo
rng = ql.MersenneTwisterUniformRsg(1, 12345)
gaussian_rng = ql.BoxMullerGaussianRsg(rng)

# Hull-White process (simplified - actual implementation more complex)
initial_rate = 0.03
scenarios = []

for scenario in range(n_scenarios):
    # Simple Euler simulation (not production quality)
    rate_path = [initial_rate]
    dt = time_horizon / time_steps
    current_rate = initial_rate
    
    for step in range(time_steps):
        # Hull-White dynamics: dr = (theta(t) - a*r)dt + sigma*dW
        sample = gaussian_rng.nextSequence()
        dw = sample.value[0] * np.sqrt(dt)
        
        # Simplified drift (should use calibrated theta function)
        drift = hw_model.a() * (0.04 - current_rate) * dt  # Mean revert to 4%
        diffusion = hw_model.sigma() * dw
        
        current_rate += drift + diffusion
        rate_path.append(max(current_rate, 0.0))  # Ensure non-negative
    
    scenarios.append(rate_path)

# Analyze scenarios
final_rates = [scenario[-1] for scenario in scenarios]
mean_final_rate = np.mean(final_rates)
std_final_rate = np.std(final_rates)
rate_95_var = np.percentile(final_rates, 5)  # 5th percentile

print(f"Interest Rate Simulation Results ({n_scenarios} scenarios):")
print(f"Mean final rate: {mean_final_rate:.4f}")
print(f"Standard deviation: {std_final_rate:.4f}")
print(f"95% VaR (5th percentile): {rate_95_var:.4f}")

# Bond valuation under scenarios (simplified)
bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(yield_curve))
bond.setPricingEngine(bond_engine)
current_bond_price = bond.cleanPrice()

print(f"\nCurrent bond price: {current_bond_price:.3f}")
```

### Option Pricing with Stochastic Volatility

```python
# Heston model option pricing vs Black-Scholes
spot = 100.0
strike = 105.0
expiry = ql.Date(15, 12, 2023)
ql.Settings.instance().evaluationDate = ql.Date(15, 6, 2023)

# Market data
risk_free_rate = 0.05
dividend_yield = 0.02

# Black-Scholes setup
bs_vol = 0.25
bs_process = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot)),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), dividend_yield, ql.Actual365Fixed())),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), risk_free_rate, ql.Actual365Fixed())),
    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.TARGET(), bs_vol, ql.Actual365Fixed()))
)

# Heston setup (stochastic volatility)
heston_process = ql.HestonProcess(
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), risk_free_rate, ql.Actual365Fixed())),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), dividend_yield, ql.Actual365Fixed())),
    ql.QuoteHandle(ql.SimpleQuote(spot)),
    v0=bs_vol**2,       # Initial variance matches BS vol
    kappa=2.0,          # Mean reversion speed
    theta=bs_vol**2,    # Long-term variance
    sigma=0.3,          # Vol of vol (30%)
    rho=-0.5            # Negative correlation (leverage effect)
)

# Create options
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
exercise = ql.EuropeanExercise(expiry)
option = ql.VanillaOption(payoff, exercise)

# Price with Black-Scholes
bs_engine = ql.AnalyticEuropeanEngine(bs_process)
option.setPricingEngine(bs_engine)
bs_price = option.NPV()
bs_delta = option.delta()
bs_gamma = option.gamma()
bs_vega = option.vega()

# Price with Heston
heston_model = ql.HestonModel(heston_process)
heston_engine = ql.AnalyticHestonEngine(heston_model)
option.setPricingEngine(heston_engine)
heston_price = option.NPV()
heston_delta = option.delta()
heston_gamma = option.gamma()

print("Option Pricing Comparison:")
print(f"Black-Scholes price: {bs_price:.4f}")
print(f"Heston price: {heston_price:.4f}")
print(f"Price difference: {heston_price - bs_price:.4f}")
print()
print("Greeks Comparison:")
print(f"BS Delta: {bs_delta:.4f}, Heston Delta: {heston_delta:.4f}")
print(f"BS Gamma: {bs_gamma:.4f}, Heston Gamma: {heston_gamma:.4f}")
print(f"BS Vega: {bs_vega:.4f}")

# Analyze volatility smile effect
strikes_range = range(80, 121, 5)
print("\nImplied Volatility Smile:")
print("Strike\tBS Vol\tHeston IV")

for k in strikes_range:
    # Black-Scholes (constant vol)
    bs_vol_constant = bs_vol
    
    # Heston implied volatility
    heston_payoff = ql.PlainVanillaPayoff(ql.Option.Call, k)
    heston_option = ql.VanillaOption(heston_payoff, exercise)
    heston_option.setPricingEngine(heston_engine)
    heston_price_k = heston_option.NPV()
    
    # Calculate Heston implied volatility by inverting Black-Scholes
    try:
        heston_iv = ql.blackFormulaImpliedVol(
            optionType=ql.Option.Call,
            strike=k,
            forward=spot * np.exp((risk_free_rate - dividend_yield) * (expiry - ql.Date.todaysDate()) / 365.0),
            redemption=heston_price_k,
            discount=np.exp(-risk_free_rate * (expiry - ql.Date.todaysDate()) / 365.0)
        )
        print(f"{k}\t{bs_vol_constant:.3f}\t{heston_iv:.3f}")
    except:
        print(f"{k}\t{bs_vol_constant:.3f}\tN/A")
```

### Fixed Income Portfolio Risk with Interest Rate Models

```python
# Multi-bond portfolio with Hull-White rate simulation
bonds_data = [
    # (maturity_years, coupon_rate, face_amount)
    (2, 0.03, 1000000),    # 2Y bond, 3% coupon, $1M
    (5, 0.04, 1500000),    # 5Y bond, 4% coupon, $1.5M
    (10, 0.045, 2000000)   # 10Y bond, 4.5% coupon, $2M
]

# Create bonds
bonds = []
for maturity_years, coupon, notional in bonds_data:
    schedule = ql.MakeSchedule(
        effectiveDate=ql.Date(15, 6, 2023),
        terminationDate=ql.Date(15, 6, 2023) + ql.Period(f"{maturity_years}Y"),
        tenor=ql.Period("6M"),
        calendar=ql.TARGET()
    )
    
    bond = ql.FixedRateBond(
        settlementDays=2,
        faceAmount=notional / 10000,  # Scale to 100 face value
        schedule=schedule,
        coupons=[coupon],
        accrualDayCounter=ql.ActualActual()
    )
    bonds.append((bond, notional))

# Current yield curve
yield_curve = ql.FlatForward(0, ql.TARGET(), 0.035, ql.Actual365Fixed())
discount_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(yield_curve))

# Current portfolio value
current_portfolio_value = 0.0
for bond, notional in bonds:
    bond.setPricingEngine(discount_engine)
    bond_price = bond.cleanPrice()
    bond_value = (bond_price / 100.0) * notional
    current_portfolio_value += bond_value

print(f"Current portfolio value: ${current_portfolio_value:,.2f}")

# Stress test scenarios: parallel yield curve shifts
rate_shifts = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]  # -200bp to +200bp
scenario_values = []

for shift in rate_shifts:
    shifted_curve = ql.FlatForward(0, ql.TARGET(), 0.035 + shift, ql.Actual365Fixed())
    shifted_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(shifted_curve))
    
    scenario_value = 0.0
    for bond, notional in bonds:
        bond.setPricingEngine(shifted_engine)
        bond_price = bond.cleanPrice()
        bond_value = (bond_price / 100.0) * notional
        scenario_value += bond_value
    
    scenario_values.append(scenario_value)
    pnl = scenario_value - current_portfolio_value
    pnl_pct = (pnl / current_portfolio_value) * 100
    
    print(f"Rate shift {shift*100:+4.0f}bp: Portfolio ${scenario_value:,.2f}, P&L ${pnl:,.2f} ({pnl_pct:+5.2f}%)")

# Calculate portfolio duration (sensitivity to parallel shifts)
small_shift = 0.0001  # 1bp
up_curve = ql.FlatForward(0, ql.TARGET(), 0.035 + small_shift, ql.Actual365Fixed())
down_curve = ql.FlatForward(0, ql.TARGET(), 0.035 - small_shift, ql.Actual365Fixed())

up_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(up_curve))
down_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(down_curve))

up_value = sum((bond.setPricingEngine(up_engine) or bond.cleanPrice()/100.0) * notional for bond, notional in bonds)
down_value = sum((bond.setPricingEngine(down_engine) or bond.cleanPrice()/100.0) * notional for bond, notional in bonds)

portfolio_duration = -(up_value - down_value) / (2 * small_shift * current_portfolio_value)
dv01 = portfolio_duration * current_portfolio_value / 10000  # Dollar duration per bp

print(f"\nPortfolio Risk Metrics:")
print(f"Modified duration: {portfolio_duration:.2f}")
print(f"DV01 ($ per bp): ${dv01:,.2f}")
```

The comprehensive QuantLib Python documentation is now complete! This covers all major functional areas with detailed examples and practical usage patterns.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create bonds and fixed income documentation", "status": "completed", "activeForm": "Creating bonds and fixed income documentation"}, {"content": "Create options and derivatives documentation", "status": "completed", "activeForm": "Creating options and derivatives documentation"}, {"content": "Create mathematical tools documentation", "status": "completed", "activeForm": "Creating mathematical tools documentation"}, {"content": "Create market data documentation", "status": "completed", "activeForm": "Creating market data documentation"}, {"content": "Create volatility and models documentation", "status": "completed", "activeForm": "Creating volatility and models documentation"}]