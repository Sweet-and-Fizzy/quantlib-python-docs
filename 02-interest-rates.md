# Interest Rates and Term Structures

This document covers QuantLib's interest rate modeling and yield curve construction classes.

## InterestRate Class

Represents an interest rate with compounding convention and day count.

### Constructor

```python
InterestRate(rate, day_counter, compounding, frequency)

# Examples
rate = ql.InterestRate(0.05, ql.Actual360(), ql.Compounded, ql.Annual)
continuous_rate = ql.InterestRate(0.048, ql.ActualActual(), ql.Continuous)
```

### Compounding Types

```python
ql.Simple                # Simple interest: (1 + r*t)
ql.Compounded           # Compound interest: (1 + r/n)^(n*t)  
ql.Continuous           # Continuous compounding: e^(r*t)
ql.SimpleThenCompounded # Simple if t≤1, compounded otherwise
```

### Frequency Types

```python
ql.NoFrequency    # For Simple and Continuous compounding
ql.Once           # Annual  
ql.Annual         # Annual
ql.Semiannual     # Semi-annual
ql.EveryFourthMonth  # Every 4 months
ql.Quarterly      # Quarterly
ql.Bimonthly      # Every 2 months
ql.Monthly        # Monthly
ql.EveryFourthWeek   # Every 4 weeks
ql.Biweekly       # Every 2 weeks
ql.Weekly         # Weekly
ql.Daily          # Daily
ql.OtherFrequency # Other
```

### Key Methods

```python
rate = ql.InterestRate(0.05, ql.Actual360(), ql.Compounded, ql.Annual)

# Accessors
rate.rate()         # 0.05
rate.dayCounter()   # Actual360 object
rate.compounding()  # ql.Compounded
rate.frequency()    # ql.Annual

# Discount and compound factors
time_to_maturity = 2.5
discount = rate.discountFactor(time_to_maturity)
compound = rate.compoundFactor(time_to_maturity)

# Rate conversion
equivalent = rate.equivalentRate(ql.Continuous, ql.NoFrequency, time_to_maturity)
simple_rate = rate.equivalentRate(ql.Simple, ql.NoFrequency, time_to_maturity)
```

### Static Methods

```python
# Calculate implied rate from compound factor
compound_factor = 1.1025  # 10.25% total return
time = 2.0
day_counter = ql.Actual360()

implied = ql.InterestRate.impliedRate(
    compound_factor, 
    day_counter, 
    ql.Compounded, 
    ql.Annual, 
    time
)
```

## YieldTermStructure (Base Class)

Base class for all yield curves providing discount factors and rates.

### Key Methods

```python
# All concrete yield curves inherit these methods
curve = ...  # Any YieldTermStructure implementation

# Discount factors
discount = curve.discount(date)
discount = curve.discount(time_to_maturity)

# Zero rates
zero_rate = curve.zeroRate(date, day_counter, compounding, frequency)
zero_rate = curve.zeroRate(time, compounding, frequency)

# Forward rates
forward = curve.forwardRate(date1, date2, day_counter, compounding)
forward = curve.forwardRate(time1, time2, compounding, frequency)

# Curve properties
curve.referenceDate()    # Curve anchor date
curve.maxDate()         # Maximum date for extrapolation
curve.maxTime()         # Maximum time for extrapolation
curve.settlementDays()   # Settlement days

# Extrapolation control
curve.enableExtrapolation()   # Allow extrapolation beyond maxDate
curve.disableExtrapolation()  # Disable extrapolation (default)
curve.allowsExtrapolation()   # Check if extrapolation enabled
```

## Concrete Term Structures

### FlatForward

Constant rate term structure.

```python
# From rate value
curve = ql.FlatForward(
    reference_date=ql.Date(15, 6, 2023),
    forward_rate=0.05,
    day_counter=ql.Actual360(),
    compounding=ql.Compounded,
    frequency=ql.Annual
)

# From quote handle (market data)
rate_handle = ql.QuoteHandle(ql.SimpleQuote(0.05))
curve = ql.FlatForward(
    settlement_days=2,
    calendar=ql.TARGET(), 
    quote_handle=rate_handle,
    day_counter=ql.Actual360(),
    compounding=ql.Compounded,
    frequency=ql.Annual
)

# Using constructor with minimal parameters
curve = ql.FlatForward(2, ql.TARGET(), rate_handle, ql.Actual360())
```

### ZeroCurve

Curve from zero rates and dates.

```python
dates = [
    ql.Date(15, 6, 2023),
    ql.Date(15, 9, 2023),
    ql.Date(15, 12, 2023),
    ql.Date(15, 6, 2024)
]
rates = [0.02, 0.025, 0.03, 0.035]

curve = ql.ZeroCurve(
    dates=dates,
    yields=rates,
    day_counter=ql.Actual360(),
    calendar=ql.TARGET(),
    interpolator=ql.Linear(),
    compounding=ql.Compounded,
    frequency=ql.Annual
)
```

### ForwardCurve

Curve from forward rates.

```python
curve = ql.ForwardCurve(
    dates=dates,
    forwards=forward_rates,
    day_counter=ql.Actual360(),
    calendar=ql.TARGET()
)
```

### DiscountCurve

Curve directly from discount factors.

```python
discount_factors = [1.0, 0.995, 0.988, 0.965]

curve = ql.DiscountCurve(
    dates=dates,
    discounts=discount_factors, 
    day_counter=ql.Actual360(),
    calendar=ql.TARGET()
)
```

## Piecewise Yield Curves

Bootstrap yield curves from market instruments using various interpolation methods.

### Bootstrap Interpolation Types

#### Zero Rate Interpolation
```python
# Linear interpolation on zero rates
ql.PiecewiseLinearZero(reference_date, helpers, day_counter)

# Cubic spline interpolation on zero rates  
ql.PiecewiseCubicZero(reference_date, helpers, day_counter)

# Log-cubic interpolation on zero rates
ql.PiecewiseLogCubicDiscount(reference_date, helpers, day_counter)
```

#### Discount Factor Interpolation
```python
# Linear interpolation on log of discount factors
ql.PiecewiseLogLinearDiscount(reference_date, helpers, day_counter)

# Cubic interpolation on log of discount factors
ql.PiecewiseLogCubicDiscount(reference_date, helpers, day_counter)

# Natural cubic spline on log discount factors
ql.PiecewiseNaturalLogCubicDiscount(reference_date, helpers, day_counter)
```

#### Forward Rate Interpolation
```python
# Flat forward rates between nodes
ql.PiecewiseFlatForward(reference_date, helpers, day_counter)

# Linear forward rates
ql.PiecewiseLinearForward(reference_date, helpers, day_counter)
```

### Construction

```python
# Settlement and reference date
settlement_days = 2
calendar = ql.TARGET()
reference_date = ql.Date(15, 6, 2023)

# Create IBOR index for swaps
ibor_index = ql.Euribor6M()

# Rate helpers (market instruments)
helpers = [
    # Deposits
    ql.DepositRateHelper(0.02, ql.Period(3, ql.Months), 2, calendar, ql.ModifiedFollowing, True, ql.Actual360()),
    
    # FRAs
    ql.FraRateHelper(0.025, 3, 6, 2, calendar, ql.ModifiedFollowing, True, ql.Actual360()),
    
    # Swaps
    ql.SwapRateHelper(0.03, ql.Period(2, ql.Years), calendar, ql.Annual, ql.ModifiedFollowing, ql.Thirty360(), ibor_index),
    ql.SwapRateHelper(0.035, ql.Period(5, ql.Years), calendar, ql.Annual, ql.ModifiedFollowing, ql.Thirty360(), ibor_index)
]

# Build curve
curve = ql.PiecewiseLogLinearDiscount(reference_date, helpers, ql.Actual360())

# Enable extrapolation beyond last helper
curve.enableExtrapolation()
```

### Available Rate Helpers

#### DepositRateHelper
For short-term deposits and cash rates.

```python
ql.DepositRateHelper(
    rate=0.02,                    # Deposit rate
    tenor=ql.Period(3, ql.Months), # Deposit tenor
    fixingDays=2,                 # Settlement days
    calendar=ql.TARGET(),
    convention=ql.ModifiedFollowing,
    endOfMonth=True,
    dayCounter=ql.Actual360()
)

# From quote handle
rate_handle = ql.QuoteHandle(ql.SimpleQuote(0.02))
helper = ql.DepositRateHelper(rate_handle, ql.Period(3, ql.Months), 2, calendar, ql.ModifiedFollowing, True, ql.Actual360())
```

#### FraRateHelper  
For Forward Rate Agreements.

```python
ql.FraRateHelper(
    rate=0.025,
    monthsToStart=3,  # 3x6 FRA starts in 3 months
    monthsToEnd=6,    # and ends in 6 months
    fixingDays=2,
    calendar=ql.TARGET(),
    convention=ql.ModifiedFollowing,
    endOfMonth=True,
    dayCounter=ql.Actual360()
)
```

#### SwapRateHelper
For interest rate swaps.

```python
ql.SwapRateHelper(
    rate=0.03,
    tenor=ql.Period(2, ql.Years),
    calendar=ql.TARGET(),
    fixedLegFrequency=ql.Annual,
    fixedLegConvention=ql.ModifiedFollowing,
    fixedLegDayCounter=ql.Thirty360(),
    iborIndex=ibor_index,
    spread=0.0,                    # Optional spread
    fwdStart=ql.Period(0, ql.Days) # Optional forward start
)
```

#### BondHelper
For bonds (used with FittedBondDiscountCurve).

```python
bond = ql.FixedRateBond(...)  # Create bond first

ql.BondHelper(
    quote_handle=ql.QuoteHandle(ql.SimpleQuote(99.5)),  # Clean price quote
    bond=bond
)
```

## Spreaded Term Structures

Overlay spreads on existing curves.

### ZeroSpreadedTermStructure

Add zero rate spread to base curve.

```python
base_curve_handle = ql.YieldTermStructureHandle(base_curve)
spread_handle = ql.QuoteHandle(ql.SimpleQuote(0.005))  # 50bp spread

spreaded_curve = ql.ZeroSpreadedTermStructure(
    base_curve_handle,
    spread_handle, 
    compounding=ql.Compounded,
    frequency=ql.Annual,
    day_counter=ql.Actual360()
)
```

### ForwardSpreadedTermStructure

Add spread to forward rates.

```python
spreaded_curve = ql.ForwardSpreadedTermStructure(
    base_curve_handle,
    spread_handle
)
```

### ImpliedTermStructure

Create term structure with different reference date.

```python
new_reference_date = ql.Date(20, 6, 2023)
implied_curve = ql.ImpliedTermStructure(base_curve_handle, new_reference_date)
```

## Fitted Bond Discount Curves

Fit parametric curves to bond prices.

### Basic Usage

```python
# Create bond helpers
bond_helpers = [...]  # List of BondHelper objects

# Choose fitting method
method = ql.NelsonSiegelFitting()
# or
method = ql.SvenssonFitting()

# Fit curve
curve = ql.FittedBondDiscountCurve(
    settlementDays=2,
    calendar=ql.TARGET(),
    helpers=bond_helpers,
    dayCounter=ql.ActualActual(),
    fittingMethod=method,
    accuracy=1e-10,
    maxEvaluations=10000,
    guess=ql.Array(),      # Initial parameter guess (optional)
    simplexLambda=1.0
)

# Access fit results
fit_results = curve.fitResults()
parameters = fit_results.solution()
iterations = fit_results.numberOfIterations()
min_cost = fit_results.minimumCostValue()
```

### Fitting Methods

#### NelsonSiegelFitting
Four-parameter Nelson-Siegel model.

```python
method = ql.NelsonSiegelFitting(
    weights=ql.Array(),           # Bond weights (equal if empty)
    optimizationMethod=None,      # Use default optimizer
    l2=ql.Array(),               # L2 regularization
    minCutoffTime=0.0,
    maxCutoffTime=1e10
)
```

#### SvenssonFitting  
Six-parameter Svensson extension of Nelson-Siegel.

```python
method = ql.SvenssonFitting(
    weights=ql.Array(),
    optimizationMethod=None,
    l2=ql.Array(),
    minCutoffTime=0.0,
    maxCutoffTime=1e10
)

# With fixed parameters
fixed_params = ql.Array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Fix beta0 and tau2
initial_values = ql.Array([0.04, -0.01, -0.01, 2.0, 0.01, 5.0])

method = ql.SvenssonFittingWithFixedParams(
    weights=ql.Array(),
    optimizationMethod=None,
    l2=ql.Array(),
    minCutoffTime=0.0,
    maxCutoffTime=1e10,
    fixedParameters=fixed_params,
    initialValues=initial_values
)
```

#### ExponentialSplinesFitting
Exponential splines with flexible number of knots.

```python
method = ql.ExponentialSplinesFitting(
    constrainAtZero=True,
    weights=ql.Array(),
    l2=ql.Array(),
    minCutoffTime=0.0,
    maxCutoffTime=1e10,
    numCoeffs=9,                  # Number of spline coefficients
    fixedKappa=None              # Fixed decay parameter
)
```

## Handle System

QuantLib uses handles for managing market data dependencies and updates.

### Quote and QuoteHandle

```python
# Create a quote
quote = ql.SimpleQuote(0.05)

# Wrap in handle
quote_handle = ql.QuoteHandle(quote)

# Use in term structure
curve = ql.FlatForward(2, ql.TARGET(), quote_handle, ql.Actual360())

# Update quote - curve automatically reflects change
quote.setValue(0.055)
new_discount = curve.discount(ql.Date(15, 6, 2024))  # Uses new rate
```

### RelinkableHandle

Allows changing the underlying object.

```python
# Create relinkable handle
rate_handle = ql.RelinkableQuoteHandle()

# Link to initial quote
initial_quote = ql.SimpleQuote(0.05)
rate_handle.linkTo(initial_quote)

# Use in curve
curve = ql.FlatForward(2, ql.TARGET(), rate_handle, ql.Actual360())

# Later, link to different quote
new_quote = ql.SimpleQuote(0.06) 
rate_handle.linkTo(new_quote)  # Curve now uses 6%
```

### YieldTermStructureHandle

For yield curve references.

```python
curve_handle = ql.YieldTermStructureHandle(curve)

# Relinkable version
curve_handle = ql.RelinkableYieldTermStructureHandle()
curve_handle.linkTo(curve)
```

## Usage Examples

### Building a Multi-Currency Curve System

```python
import QuantLib as ql

# Setup
ql.Settings.instance().evaluationDate = ql.Date(15, 6, 2023)
calendar = ql.TARGET()
settlement_days = 2

# USD curve
usd_helpers = [
    ql.DepositRateHelper(0.02, ql.Period(3, ql.Months), 2, calendar, ql.ModifiedFollowing, True, ql.Actual360()),
    ql.DepositRateHelper(0.022, ql.Period(6, ql.Months), 2, calendar, ql.ModifiedFollowing, True, ql.Actual360())
]

usd_curve = ql.PiecewiseLogLinearDiscount(ql.Date(15, 6, 2023), usd_helpers, ql.Actual360())
usd_curve.enableExtrapolation()

# EUR curve  
eur_helpers = [
    ql.DepositRateHelper(0.015, ql.Period(3, ql.Months), 2, calendar, ql.ModifiedFollowing, True, ql.Actual360()),
    ql.DepositRateHelper(0.018, ql.Period(6, ql.Months), 2, calendar, ql.ModifiedFollowing, True, ql.Actual360())
]

eur_curve = ql.PiecewiseLogLinearDiscount(ql.Date(15, 6, 2023), eur_helpers, ql.Actual360())
eur_curve.enableExtrapolation()

# Compare rates
maturity = ql.Date(15, 6, 2024)
usd_rate = usd_curve.zeroRate(maturity, ql.Actual360(), ql.Compounded, ql.Annual)
eur_rate = eur_curve.zeroRate(maturity, ql.Actual360(), ql.Compounded, ql.Annual)

print(f"USD 1Y rate: {usd_rate.rate():.4f}")
print(f"EUR 1Y rate: {eur_rate.rate():.4f}")
```

### Dynamic Curve Updates

```python
# Create updating rate environment
base_rate = ql.SimpleQuote(0.05)
rate_handle = ql.QuoteHandle(base_rate)

curve = ql.FlatForward(2, ql.TARGET(), rate_handle, ql.Actual360())

# Price some instrument
maturity = ql.Date(15, 6, 2024)
initial_discount = curve.discount(maturity)

# Update rate and see effect
base_rate.setValue(0.055)
new_discount = curve.discount(maturity)

print(f"Rate change effect: {(new_discount/initial_discount - 1)*100:.2f}%")
```

### Curve Interpolation Comparison

```python
# Same market data, different interpolation methods
dates = [ql.Date(15, 6, 2023), ql.Date(15, 12, 2023), ql.Date(15, 6, 2024), ql.Date(15, 6, 2025)]
rates = [0.02, 0.025, 0.03, 0.035]
day_counter = ql.Actual360()

# Linear zero interpolation
linear_curve = ql.ZeroCurve(dates, rates, day_counter, ql.TARGET(), ql.Linear(), ql.Compounded, ql.Annual)

# Cubic spline interpolation
cubic_curve = ql.ZeroCurve(dates, rates, day_counter, ql.TARGET(), ql.Cubic(), ql.Compounded, ql.Annual)

# Compare at intermediate point
test_date = ql.Date(15, 3, 2024)
linear_rate = linear_curve.zeroRate(test_date, day_counter, ql.Compounded, ql.Annual).rate()
cubic_rate = cubic_curve.zeroRate(test_date, day_counter, ql.Compounded, ql.Annual).rate()

print(f"Linear interpolation: {linear_rate:.4f}")
print(f"Cubic interpolation: {cubic_rate:.4f}")
```

### Forward Rate Analysis

```python
curve = ql.FlatForward(ql.Date(15, 6, 2023), 0.05, ql.Actual360())

# Calculate forward rates for different periods
periods = [
    ("3M forward in 6M", ql.Date(15, 12, 2023), ql.Date(15, 3, 2024)),
    ("6M forward in 1Y", ql.Date(15, 6, 2024), ql.Date(15, 12, 2024)),
    ("1Y forward in 2Y", ql.Date(15, 6, 2025), ql.Date(15, 6, 2026))
]

for desc, start, end in periods:
    forward = curve.forwardRate(start, end, ql.Actual360(), ql.Simple)
    print(f"{desc}: {forward.rate():.4f}")
```