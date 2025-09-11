# Options and Derivatives

This document covers QuantLib's option pricing classes, payoffs, exercise styles, and exotic derivatives.

## Core Option Components

### Option Base Class

All option instruments inherit from the `Option` base class.

```python
# Base option methods (available on all options)
option.payoff()         # Get payoff function
option.exercise()       # Get exercise schedule
option.setPricingEngine(engine)  # Set pricing method
option.NPV()           # Option value
option.delta()         # Price sensitivity to underlying
option.gamma()         # Delta sensitivity
option.theta()         # Time decay
option.vega()          # Volatility sensitivity  
option.rho()           # Interest rate sensitivity
```

### Option Types

```python
ql.Option.Call = 1     # Call option
ql.Option.Put = -1     # Put option
```

## Payoff Classes

Payoff functions define the option's cash flows at expiration.

### PlainVanillaPayoff

Standard call/put payoff: max(S-K, 0) for calls, max(K-S, 0) for puts.

```python
PlainVanillaPayoff(option_type, strike)

# Examples
call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
put_payoff = ql.PlainVanillaPayoff(ql.Option.Put, 95.0)

# Calculate payoff at specific spot price
spot = 105.0
call_value = call_payoff(spot)  # max(105-100, 0) = 5.0
put_value = put_payoff(spot)    # max(95-105, 0) = 0.0
```

### StrikedTypePayoff (Base Class)

Base class for payoffs with strike and option type.

```python
# Common methods for all struck payoffs
payoff.strike()        # Strike price
payoff.optionType()    # Call or Put
payoff(spot_price)     # Calculate payoff
```

### Binary/Digital Payoffs

#### CashOrNothingPayoff
Pays fixed cash amount if in-the-money.

```python
CashOrNothingPayoff(option_type, strike, cash_amount)

# Digital call: pays $10 if spot > 100
digital_call = ql.CashOrNothingPayoff(ql.Option.Call, 100.0, 10.0)
```

#### AssetOrNothingPayoff  
Pays underlying asset value if in-the-money.

```python
AssetOrNothingPayoff(option_type, strike)

# Asset-or-nothing call: pays spot if spot > 100
asset_call = ql.AssetOrNothingPayoff(ql.Option.Call, 100.0)
```

### Exotic Payoffs

#### SuperSharePayoff
Pays spot price if within range, zero otherwise.

```python
SuperSharePayoff(lower_strike, upper_strike)

# Pays spot if 90 <= spot <= 110
super_share = ql.SuperSharePayoff(90.0, 110.0)
```

#### GapPayoff
Standard payoff but with different strike for payoff calculation.

```python
GapPayoff(option_type, strike, payoff_strike)

# Call struck at 100, but payoff based on spot - 95
gap_call = ql.GapPayoff(ql.Option.Call, 100.0, 95.0)
```

#### PercentageStrikePayoff
Strike defined as percentage of spot price.

```python
PercentageStrikePayoff(option_type, percentage_strike)

# Put with strike at 90% of current spot
percentage_put = ql.PercentageStrikePayoff(ql.Option.Put, 0.9)
```

## Exercise Styles

Define when options can be exercised.

### EuropeanExercise
Exercise only at expiration.

```python
EuropeanExercise(expiry_date)

# European option expiring June 15, 2024
european = ql.EuropeanExercise(ql.Date(15, 6, 2024))
```

### AmericanExercise
Exercise any time between earliest and latest dates.

```python
AmericanExercise(earliest_date, latest_date)

# American option, exercise from today until June 15, 2024
american = ql.AmericanExercise(
    ql.Date.todaysDate(),
    ql.Date(15, 6, 2024)
)

# American option with single date (exercise any time until expiry)
american = ql.AmericanExercise(ql.Date(15, 6, 2024))
```

### BermudanExercise
Exercise on specific dates only.

```python
BermudanExercise(exercise_dates)

# Bermudan option exercisable quarterly
exercise_dates = [
    ql.Date(15, 9, 2023),
    ql.Date(15, 12, 2023), 
    ql.Date(15, 3, 2024),
    ql.Date(15, 6, 2024)
]
bermudan = ql.BermudanExercise(exercise_dates)
```

## Vanilla Options

### VanillaOption

Standard European and American options.

```python
VanillaOption(payoff, exercise)

# European call
call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
european_exercise = ql.EuropeanExercise(ql.Date(15, 6, 2024))
european_call = ql.VanillaOption(call_payoff, european_exercise)

# American put
put_payoff = ql.PlainVanillaPayoff(ql.Option.Put, 95.0)
american_exercise = ql.AmericanExercise(ql.Date(15, 6, 2024))
american_put = ql.VanillaOption(put_payoff, american_exercise)
```

### Complete European Option Example

```python
import QuantLib as ql

# Market data
spot = 100.0
strike = 105.0
risk_free_rate = 0.05
dividend_yield = 0.02
volatility = 0.25
expiry = ql.Date(15, 6, 2024)

# Evaluation date
ql.Settings.instance().evaluationDate = ql.Date(15, 6, 2023)

# Create option
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
exercise = ql.EuropeanExercise(expiry)
option = ql.VanillaOption(payoff, exercise)

# Market data handles
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), risk_free_rate, ql.Actual360()))
dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), dividend_yield, ql.Actual360()))
vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.TARGET(), volatility, ql.Actual360()))

# Black-Scholes process
process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)

# Pricing engine
engine = ql.AnalyticEuropeanEngine(process)
option.setPricingEngine(engine)

# Results
option_value = option.NPV()
delta = option.delta()
gamma = option.gamma()
theta = option.theta()
vega = option.vega()
rho = option.rho()

print(f"Option Value: {option_value:.4f}")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Vega: {vega:.4f}")
print(f"Rho: {rho:.4f}")
```

## Barrier Options

Options with payoff dependent on underlying reaching barrier levels.

### BarrierOption

```python
BarrierOption(
    barrier_type,    # Type of barrier
    barrier_level,   # Barrier price level
    rebate,         # Rebate if barrier hit
    payoff,         # Underlying payoff
    exercise        # Exercise style
)
```

### Barrier Types

```python
# Knock-out options (become worthless if barrier hit)
ql.Barrier.UpOut    # Knocked out if spot rises above barrier
ql.Barrier.DownOut  # Knocked out if spot falls below barrier

# Knock-in options (only become active if barrier hit)  
ql.Barrier.UpIn     # Activated if spot rises above barrier
ql.Barrier.DownIn   # Activated if spot falls below barrier
```

### Example - Down-and-Out Call

```python
# Down-and-out call: worthless if spot falls below 80
barrier_level = 80.0
rebate = 5.0  # Rebate if barrier hit

down_out_call = ql.BarrierOption(
    barrierType=ql.Barrier.DownOut,
    barrier=barrier_level,
    rebate=rebate,
    payoff=ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
    exercise=ql.EuropeanExercise(ql.Date(15, 6, 2024))
)

# Pricing with Black-Scholes
barrier_engine = ql.AnalyticBarrierEngine(process)
down_out_call.setPricingEngine(barrier_engine)
barrier_value = down_out_call.NPV()
```

### Double Barrier Options

Options with two barrier levels.

```python
DoubleBarrierOption(
    barrier_type,     # ql.DoubleBarrier.KnockOut or KnockIn
    barrier_lo,       # Lower barrier
    barrier_hi,       # Upper barrier  
    rebate,
    payoff,
    exercise
)

# Double knock-out: worthless if spot hits either 80 or 120
double_barrier = ql.DoubleBarrierOption(
    barrierType=ql.DoubleBarrier.KnockOut,
    barrier_lo=80.0,
    barrier_hi=120.0,
    rebate=0.0,
    payoff=ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
    exercise=ql.EuropeanExercise(ql.Date(15, 6, 2024))
)
```

## Asian Options

Options with payoff based on average underlying price.

### DiscreteAveragingAsianOption

Asian option with discrete averaging (specific observation dates).

```python
DiscreteAveragingAsianOption(
    average_type,    # ql.Average.Arithmetic or Geometric
    runningAccumulator, # Running sum of past observations
    pastFixings,     # Number of past fixings
    fixingDates,     # Future averaging dates  
    payoff,
    exercise
)

# Arithmetic average Asian call
fixing_dates = [
    ql.Date(15, 7, 2023),
    ql.Date(15, 8, 2023),
    ql.Date(15, 9, 2023),
    ql.Date(15, 10, 2023),
    ql.Date(15, 11, 2023),
    ql.Date(15, 12, 2023)
]

asian_call = ql.DiscreteAveragingAsianOption(
    averageType=ql.Average.Arithmetic,
    runningAccumulator=0.0,  # No past fixings
    pastFixings=0,
    fixingDates=fixing_dates,
    payoff=ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
    exercise=ql.EuropeanExercise(ql.Date(15, 12, 2023))
)
```

### ContinuousAveragingAsianOption

Asian option with continuous averaging.

```python
ContinuousAveragingAsianOption(
    average_type,
    payoff,
    exercise
)

# Geometric average Asian put
geometric_asian = ql.ContinuousAveragingAsianOption(
    averageType=ql.Average.Geometric,
    payoff=ql.PlainVanillaPayoff(ql.Option.Put, 95.0),
    exercise=ql.EuropeanExercise(ql.Date(15, 6, 2024))
)

# Pricing
asian_engine = ql.AnalyticContinuousGeometricAveragePriceAsianEngine(process)
geometric_asian.setPricingEngine(asian_engine)
```

## Lookback Options

Options with payoff based on maximum or minimum underlying price.

### ContinuousFixedLookbackOption

Lookback option with fixed strike, continuous monitoring.

```python
ContinuousFixedLookbackOption(
    minmax,         # ql.MinMax.Min or Max
    payoff,
    exercise
)

# Lookback call: max(S_max - K, 0)
lookback_call = ql.ContinuousFixedLookbackOption(
    minmax=ql.MinMax.Max,
    payoff=ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
    exercise=ql.EuropeanExercise(ql.Date(15, 6, 2024))
)
```

### ContinuousFloatingLookbackOption

Lookback option with floating strike.

```python
ContinuousFloatingLookbackOption(
    minmax,
    payoff,         # Use FloatingTypePayoff
    exercise
)

# Floating strike lookback call: S - S_min
floating_payoff = ql.FloatingTypePayoff(ql.Option.Call)
floating_lookback = ql.ContinuousFloatingLookbackOption(
    minmax=ql.MinMax.Min,
    payoff=floating_payoff,
    exercise=ql.EuropeanExercise(ql.Date(15, 6, 2024))
)
```

## Basket Options

Options on multiple underlyings.

### BasketOption

```python
BasketOption(payoff, exercise)

# Requires multi-asset payoff
max_payoff = ql.MaxBasketPayoff(ql.PlainVanillaPayoff(ql.Option.Call, 100.0))
basket_option = ql.BasketOption(max_payoff, ql.EuropeanExercise(ql.Date(15, 6, 2024)))
```

## Cliquet Options

Forward-starting options with periodic resets.

### CliquetOption

```python
CliquetOption(
    payoff,
    exercise, 
    resetDates,      # Reset dates
    resetType=ql.CliquetOption.Reset
)

# Quarterly reset cliquet
reset_dates = [
    ql.Date(15, 9, 2023),
    ql.Date(15, 12, 2023),
    ql.Date(15, 3, 2024)
]

cliquet = ql.CliquetOption(
    payoff=ql.PercentageStrikePayoff(ql.Option.Call, 1.0),  # At-the-money
    exercise=ql.EuropeanExercise(ql.Date(15, 6, 2024)),
    resetDates=reset_dates
)
```

## Swaptions

Options on interest rate swaps.

### Swaption

```python
Swaption(
    swap,           # Underlying VanillaSwap
    exercise,       # Exercise schedule
    type_=ql.Settlement.Physical  # Physical or Cash settlement
)

# Create underlying swap
underlying_swap = ql.MakeVanillaSwap(
    swapTenor=ql.Period("5Y"),
    index=ql.Euribor6M(),
    fixedRate=0.03
)

# European swaption
swaption_exercise = ql.EuropeanExercise(ql.Date(15, 6, 2024))
swaption = ql.Swaption(underlying_swap, swaption_exercise)

# Pricing
swaption_engine = ql.BlackSwaptionEngine(
    discountCurve=discount_curve_handle,
    vol=vol_handle
)
swaption.setPricingEngine(swaption_engine)
```

### Bermudan Swaption

```python
# Bermudan swaption (callable on coupon dates)
bermudan_exercise = ql.BermudanExercise(exercise_dates)
bermudan_swaption = ql.Swaption(underlying_swap, bermudan_exercise)

# Tree pricing
tree_engine = ql.TreeSwaptionEngine(hw_model, 50)  # Hull-White with 50 steps
bermudan_swaption.setPricingEngine(tree_engine)
```

## Cap and Floor Options

Interest rate caps and floors.

### Cap

Series of caplets on floating rate.

```python
Cap(
    floatingLeg,    # Leg of floating rate coupons
    strikes         # Cap rates
)

# Or using MakeCap
cap = ql.MakeCap(
    capTenor=ql.Period("5Y"),
    index=ql.Euribor3M(),
    strike=0.05     # 5% cap rate
).withNominal(1000000)

# Pricing
cap_engine = ql.BlackCapFloorEngine(
    discountCurve=discount_curve_handle,
    vol=vol_handle
)
cap.setPricingEngine(cap_engine)
```

### Floor

Series of floorlets.

```python
floor = ql.MakeFloor(
    floorTenor=ql.Period("3Y"),
    index=ql.Euribor6M(),
    strike=0.01     # 1% floor rate
).withNominal(1000000)
```

### CapFloor

General cap/floor instrument.

```python
CapFloor(
    type_,          # ql.CapFloor.Cap or Floor
    floatingLeg,
    strikes
)

# Cap and floor on same leg
ibor_leg = [...] # Floating rate leg
cap = ql.CapFloor(ql.CapFloor.Cap, ibor_leg, [0.05])
floor = ql.CapFloor(ql.CapFloor.Floor, ibor_leg, [0.01])
```

## Pricing Engines

Different numerical methods for option valuation.

### Analytic Engines

Closed-form solutions when available.

```python
# Black-Scholes for European vanilla options
ql.AnalyticEuropeanEngine(process)

# European barrier options
ql.AnalyticBarrierEngine(process)

# European digital options  
ql.AnalyticDigitalAmericanEngine(process)

# Geometric Asian options
ql.AnalyticContinuousGeometricAveragePriceAsianEngine(process)

# Heston model options
ql.AnalyticHestonEngine(heston_model)
```

### Finite Difference Engines

PDE-based pricing methods.

```python
# American vanilla options
ql.FDAmericanEngine(process, timeSteps=100, gridPoints=100)

# European options with PDE
ql.FDEuropeanEngine(process, timeSteps=50, gridPoints=50)

# Bermudan options
ql.FDBermudanEngine(process, timeSteps=100, gridPoints=100)
```

### Monte Carlo Engines

Simulation-based pricing.

```python
# European options via Monte Carlo
ql.MCEuropeanEngine(
    process=process,
    traits="PseudoRandom",  # or "LowDiscrepancy"
    timeSteps=1,
    timeStepsPerYear=None,
    requiredSamples=None,
    requiredTolerance=0.02,
    maxSamples=1000000,
    seed=0
)

# American options via Monte Carlo
ql.MCAmericanEngine(
    process=process,
    traits="PseudoRandom",
    timeSteps=100,
    requiredSamples=None,
    requiredTolerance=0.02,
    maxSamples=1000000,
    seed=0
)
```

### Tree/Lattice Engines

Binomial and trinomial tree methods.

```python
# Binomial tree for American options
ql.BinomialVanillaEngine(process, "crr", steps=100)    # Cox-Ross-Rubinstein
ql.BinomialVanillaEngine(process, "jr", steps=100)     # Jarrow-Rudd
ql.BinomialVanillaEngine(process, "eqp", steps=100)    # Equal probabilities

# Trinomial tree
ql.BinomialVanillaEngine(process, "additive", steps=100)
ql.BinomialVanillaEngine(process, "trigeorgiadis", steps=100)
```

## Usage Examples

### Multi-Asset Rainbow Option

```python
# Max of two assets option: max(max(S1, S2) - K, 0)
import QuantLib as ql
import numpy as np

# Market data for two assets
spot1, spot2 = 100.0, 110.0
vol1, vol2 = 0.25, 0.30
correlation = 0.7
risk_free_rate = 0.05

# Individual processes
process1 = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot1)),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), 0.0, ql.Actual360())),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), risk_free_rate, ql.Actual360())),
    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.TARGET(), vol1, ql.Actual360()))
)

process2 = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot2)),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), 0.0, ql.Actual360())),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), risk_free_rate, ql.Actual360())),
    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.TARGET(), vol2, ql.Actual360()))
)

# Multi-asset process would require more complex setup
# This is conceptual - actual implementation requires StochasticProcessArray
```

### Volatility Smile Calibration

```python
# Calibrate Black-Scholes implied volatility smile
strikes = [80, 90, 100, 110, 120]
market_vols = [0.35, 0.28, 0.25, 0.27, 0.32]  # Volatility smile
expiry = ql.Date(15, 6, 2024)

calibrated_vols = []
spot = 100.0
process_template = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot)),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), 0.02, ql.Actual360())),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), 0.05, ql.Actual360())),
    None  # Will be set for each strike
)

for strike, market_vol in zip(strikes, market_vols):
    # Create option
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(expiry)
    option = ql.VanillaOption(payoff, exercise)
    
    # Set volatility and price
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(0, ql.TARGET(), market_vol, ql.Actual360())
    )
    process = ql.BlackScholesMertonProcess(
        process_template.stateVariable(),
        process_template.dividendYield(), 
        process_template.riskFreeRate(),
        vol_handle
    )
    
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)
    
    option_price = option.NPV()
    calibrated_vols.append((strike, market_vol, option_price))

for strike, vol, price in calibrated_vols:
    print(f"Strike {strike}: Vol {vol:.3f}, Price {price:.4f}")
```

### American Put Option with Finite Differences

```python
# American put option with early exercise premium
spot = 100.0
strike = 105.0  # Out-of-the-money put
volatility = 0.25
risk_free_rate = 0.05
dividend_yield = 0.03

# Market data handles
process = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot)),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), dividend_yield, ql.Actual360())),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), risk_free_rate, ql.Actual360())),
    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.TARGET(), volatility, ql.Actual360()))
)

# Create American and European puts for comparison
payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)
american_exercise = ql.AmericanExercise(ql.Date(15, 6, 2024))
european_exercise = ql.EuropeanExercise(ql.Date(15, 6, 2024))

american_put = ql.VanillaOption(payoff, american_exercise)
european_put = ql.VanillaOption(payoff, european_exercise)

# Different engines
fd_engine = ql.FDAmericanEngine(process, timeSteps=100, gridPoints=100)
analytic_engine = ql.AnalyticEuropeanEngine(process)
binomial_engine = ql.BinomialVanillaEngine(process, "crr", 200)

# Price with different methods
american_put.setPricingEngine(fd_engine)
american_fd_price = american_put.NPV()

american_put.setPricingEngine(binomial_engine)  
american_tree_price = american_put.NPV()

european_put.setPricingEngine(analytic_engine)
european_price = european_put.NPV()

early_exercise_premium = american_fd_price - european_price

print(f"European Put: {european_price:.4f}")
print(f"American Put (FD): {american_fd_price:.4f}")
print(f"American Put (Tree): {american_tree_price:.4f}")
print(f"Early Exercise Premium: {early_exercise_premium:.4f}")
```

### Interest Rate Cap Analysis

```python
# 5-year cap on 3-month EURIBOR
euribor3m = ql.Euribor3M()
cap_rate = 0.04  # 4% cap
nominal = 1000000.0

# Build cap
cap = ql.MakeCap(
    capTenor=ql.Period("5Y"),
    index=euribor3m,
    strike=cap_rate
).withNominal(nominal)

# Market curves
discount_curve = ql.FlatForward(0, ql.TARGET(), 0.03, ql.Actual360())
forecast_curve = ql.FlatForward(0, ql.TARGET(), 0.035, ql.Actual360())

# Link index to forecast curve  
euribor3m = ql.Euribor3M(ql.YieldTermStructureHandle(forecast_curve))

# Volatility structure
vol_surface = ql.ConstantOptionletVolatility(
    0, ql.TARGET(), 
    ql.ModifiedFollowing, 
    0.20,  # 20% volatility
    ql.Actual360()
)

# Pricing engine
cap_engine = ql.BlackCapFloorEngine(
    ql.YieldTermStructureHandle(discount_curve),
    ql.OptionletVolatilityStructureHandle(vol_surface)
)

cap.setPricingEngine(cap_engine)

# Results
cap_value = cap.NPV()
cap_vega = cap.vega()

# Individual caplet analysis
caplet_values = []
for i in range(cap.floatingLeg().size()):
    if i < len(cap):  # Access individual caplets if available
        caplet = cap[i]
        caplet_values.append(caplet.NPV())

print(f"Cap Value: ${cap_value:,.2f}")
print(f"Cap Vega: ${cap_vega:,.2f}")
print(f"Average Caplet Value: ${np.mean(caplet_values):,.2f}" if caplet_values else "")
```