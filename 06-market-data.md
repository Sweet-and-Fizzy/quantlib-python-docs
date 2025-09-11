# Market Data and Quote Handling

This document covers QuantLib's market data infrastructure including quotes, handles, indexes, and currencies.

## Quote System

QuantLib uses a sophisticated quote and handle system to manage market data dependencies and automatic updates.

### Quote Base Class

All quote types inherit from the `Quote` base class.

```python
# Base quote interface (all quotes implement these)
quote.value()         # Current quote value
quote.isValid()       # True if quote has valid value
```

### SimpleQuote

Most common quote type - a mutable quote that can be updated.

```python
# Constructors
SimpleQuote()              # Invalid quote (no value)
SimpleQuote(value)         # Quote with initial value

# Examples
rate_quote = ql.SimpleQuote(0.05)    # 5% interest rate
price_quote = ql.SimpleQuote(99.5)   # Bond price
vol_quote = ql.SimpleQuote()         # Invalid initially

# Operations
rate_quote.setValue(0.055)           # Update to 5.5%
current_rate = rate_quote.value()    # Get current value
is_set = rate_quote.isValid()        # Check if valid

# Reset quote
rate_quote.reset()                   # Makes quote invalid again
```

### Derived Quotes

Quotes that depend on other quotes through mathematical relationships.

#### CompositeQuote

Quote based on function of other quotes.

```python
# Create underlying quotes
spot_quote = ql.SimpleQuote(100.0)
strike_quote = ql.SimpleQuote(105.0)

# Composite quote: spot - strike (intrinsic value)
def intrinsic_value(spot, strike):
    return max(spot - strike, 0.0)

# Note: CompositeQuote requires more complex setup in practice
# Often easier to use derived classes or custom quotes
```

#### DeltaVolQuote

Quote for delta-neutral volatility (common in FX markets).

```python
delta_vol_quote = ql.DeltaVolQuote(
    delta=ql.QuoteHandle(ql.SimpleQuote(0.25)),      # 25-delta
    vol=ql.QuoteHandle(ql.SimpleQuote(0.20)),        # Volatility  
    maturity=1.0,                                    # Time to maturity
    deltaType=ql.DeltaVolQuote.Spot                  # Delta type
)
```

### ForwardValueQuote

Quote representing forward value of an underlying.

```python
# Forward quote based on spot, rates, and dividends
spot_handle = ql.QuoteHandle(ql.SimpleQuote(100.0))
rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), 0.05, ql.Actual360()))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), 0.02, ql.Actual360()))

forward_quote = ql.ForwardValueQuote(
    index=spot_handle,
    fixingDate=ql.Date(15, 6, 2024)
)
```

## Handle System

Handles provide automatic dependency tracking and updates throughout QuantLib. **Important**: Handles are essential for connecting market data to instruments - you cannot pass quotes or curves directly to most QuantLib objects.

### QuoteHandle

Wrapper for quotes that enables automatic recalculation.

```python
# Create quote and handle
quote = ql.SimpleQuote(0.05)
quote_handle = ql.QuoteHandle(quote)

# Use handle in term structure
curve = ql.FlatForward(2, ql.TARGET(), quote_handle, ql.Actual360())

# Update quote - curve automatically updates
quote.setValue(0.055)
new_discount = curve.discount(ql.Date(15, 6, 2024))  # Uses new 5.5% rate

# Handle methods  
quote_handle.isValid()        # True if underlying quote is valid
quote_handle.empty()          # True if no quote linked
current_value = quote_handle.value()  # Shortcut to quote.value()
```

### RelinkableQuoteHandle

Handle that can be linked to different quotes during its lifetime.

```python
# Create relinkable handle
relinkable_handle = ql.RelinkableQuoteHandle()

# Initially empty
print(f"Empty handle: {relinkable_handle.empty()}")  # True

# Link to first quote
first_quote = ql.SimpleQuote(0.05)
relinkable_handle.linkTo(first_quote)
print(f"First value: {relinkable_handle.value()}")   # 0.05

# Switch to different quote
second_quote = ql.SimpleQuote(0.06)
relinkable_handle.linkTo(second_quote) 
print(f"Second value: {relinkable_handle.value()}")  # 0.06

# Any objects using this handle automatically see the change
curve = ql.FlatForward(2, ql.TARGET(), relinkable_handle, ql.Actual360())
```

### Specialized Handles

#### YieldTermStructureHandle

For yield curve references.

```python
curve = ql.FlatForward(0, ql.TARGET(), 0.05, ql.Actual360())
curve_handle = ql.YieldTermStructureHandle(curve)

# Relinkable version
relinkable_curve_handle = ql.RelinkableYieldTermStructureHandle()
relinkable_curve_handle.linkTo(curve)

# Later, link to different curve
new_curve = ql.FlatForward(0, ql.TARGET(), 0.055, ql.Actual360())
relinkable_curve_handle.linkTo(new_curve)
```

#### BlackVolTermStructureHandle

For volatility surface references.

```python
vol_surface = ql.BlackConstantVol(0, ql.TARGET(), 0.25, ql.Actual360())
vol_handle = ql.BlackVolTermStructureHandle(vol_surface)
```

## Interest Rate Indexes

Indexes represent reference rates like LIBOR, EURIBOR, etc.

### IborIndex Base Class

Base class for inter-bank offered rates.

```python
# Common properties for all IBOR indexes
index.name()                 # Index name (e.g., "Euribor6M")
index.familyName()          # Family name (e.g., "Euribor")
index.currency()            # Currency (e.g., EUR)
index.dayCounter()          # Day count convention
index.settlementDays()      # Settlement days
index.businessDayConvention()  # Business day adjustment
index.endOfMonth()          # End-of-month adjustment
index.tenor()               # Index tenor (e.g., 6M)
index.fixingCalendar()      # Calendar for fixings
index.isValidFixingDate(date)  # Check if date is valid for fixing

# Fixing operations
index.fixing(date)          # Get historical fixing
index.addFixing(date, rate) # Add historical fixing
index.addFixings(dates, rates)  # Add multiple fixings
```

### Major IBOR Indexes

#### USD LIBOR Family

```python
# USD LIBOR indexes (historical - being phased out)
usd_libor_1m = ql.USDLibor(ql.Period("1M"))
usd_libor_3m = ql.USDLibor(ql.Period("3M"))
usd_libor_6m = ql.USDLibor(ql.Period("6M"))
usd_libor_12m = ql.USDLibor(ql.Period("12M"))

# With forecast curve
forecast_curve = ql.YieldTermStructureHandle(curve)
usd_libor_3m_with_curve = ql.USDLibor(ql.Period("3M"), forecast_curve)

# Index properties
print(f"Name: {usd_libor_3m.name()}")                    # "USDLibor3M"
print(f"Currency: {usd_libor_3m.currency().code()}")     # "USD"
print(f"Day count: {usd_libor_3m.dayCounter().name()}")  # "Actual/360"
print(f"Settlement days: {usd_libor_3m.settlementDays()}")  # 2
```

#### SOFR (Successor to USD LIBOR)

```python
# Secured Overnight Financing Rate
sofr = ql.Sofr()
sofr_with_curve = ql.Sofr(forecast_curve)

# SOFR is an overnight rate
print(f"SOFR tenor: {sofr.tenor()}")  # 1D
```

#### EURIBOR Family

```python
# European inter-bank rates
euribor_1w = ql.Euribor1W()
euribor_1m = ql.Euribor1M() 
euribor_3m = ql.Euribor3M()
euribor_6m = ql.Euribor6M()
euribor_12m = ql.Euribor12M()

# Generic constructor
euribor_9m = ql.Euribor(ql.Period("9M"))  # 9-month EURIBOR

# Properties
print(f"Currency: {euribor_6m.currency().code()}")      # "EUR"
print(f"Calendar: {euribor_6m.fixingCalendar().name()}")  # "TARGET"
```

#### Other Major Indexes

```python
# British Pound
gbp_libor_3m = ql.GBPLibor(ql.Period("3M"))
sonia = ql.Sonia()  # Sterling Overnight Index Average

# Japanese Yen  
jpy_libor_6m = ql.JPYLibor(ql.Period("6M"))
tonar = ql.Tonar()  # Tokyo Overnight Average Rate

# Swiss Franc
chf_libor_3m = ql.CHFLibor(ql.Period("3M"))

# Canadian Dollar
cad_libor_3m = ql.CADLibor(ql.Period("3M"))

# Australian Dollar
aud_bbsw_3m = ql.AUDLibor(ql.Period("3M"))  # Bank Accepted Bills Swap Rate

# Other European rates
eonia = ql.Eonia()  # Euro Overnight Index Average  
```

### Overnight Indexes

Special category for overnight rates.

```python
# Major overnight indexes
eonia = ql.Eonia()          # Euro area
sofr = ql.Sofr()            # United States  
sonia = ql.Sonia()          # United Kingdom
tonar = ql.Tonar()          # Japan

# Properties of overnight indexes
print(f"EONIA day count: {eonia.dayCounter().name()}")  # "Actual/360"
print(f"SOFR calendar: {sofr.fixingCalendar().name()}")  # "United States"

# Overnight indexes typically have 0 settlement days
print(f"SOFR settlement days: {sofr.settlementDays()}")  # 0
```

### Swap Indexes

Represent swap rates for various tenors.

```python
# Euro area swap indexes
eur_swap_2y = ql.EuriborSwapIsdaFixA(ql.Period("2Y"))
eur_swap_5y = ql.EuriborSwapIsdaFixA(ql.Period("5Y"))
eur_swap_10y = ql.EuriborSwapIsdaFixA(ql.Period("10Y"))

# USD swap indexes
usd_swap_2y = ql.UsdLiborSwapIsdaFixAm(ql.Period("2Y"))
usd_swap_5y = ql.UsdLiborSwapIsdaFixAm(ql.Period("5Y"))

# Swap index properties
print(f"Swap name: {eur_swap_5y.name()}")               # "EuriborSwapIsdaFixA5Y"
print(f"Fixed leg frequency: {eur_swap_5y.fixedLegFrequency()}")  # Annual
print(f"Fixed leg day count: {eur_swap_5y.dayCounter().name()}")  # "30/360"

# Get underlying swap at fixing date
fixing_date = ql.Date(15, 6, 2023)
underlying_swap = eur_swap_5y.underlyingSwap(fixing_date)
```

### Index Management

#### Adding Historical Fixings

```python
# Add historical fixings for index
euribor_3m = ql.Euribor3M()

# Single fixing
fixing_date = ql.Date(15, 6, 2023)
fixing_rate = 0.034
euribor_3m.addFixing(fixing_date, fixing_rate)

# Multiple fixings
fixing_dates = [
    ql.Date(15, 6, 2023),
    ql.Date(16, 6, 2023),
    ql.Date(19, 6, 2023)  # Skip weekend
]
fixing_rates = [0.034, 0.035, 0.0345]
euribor_3m.addFixings(fixing_dates, fixing_rates)

# Check fixing
retrieved_fixing = euribor_3m.fixing(fixing_date)
print(f"EURIBOR 3M on {fixing_date}: {retrieved_fixing:.4f}")
```

#### Index Curves

Link indexes to forecast curves for forward rate projection.

```python
# Create forecast curve
forecast_curve = ql.FlatForward(0, ql.TARGET(), 0.035, ql.Actual360())
curve_handle = ql.YieldTermStructureHandle(forecast_curve)

# Link index to curve
euribor_3m_with_curve = ql.Euribor3M(curve_handle)

# Forward fixing (projected rate)
future_date = ql.Date(15, 12, 2023)
if euribor_3m_with_curve.isValidFixingDate(future_date):
    forward_rate = euribor_3m_with_curve.fixing(future_date)
    print(f"Forward EURIBOR 3M: {forward_rate:.4f}")
```

## Currency Classes

Represent different currencies with their properties.

### Major Currencies

```python
# G10 currencies
usd = ql.USDCurrency()
eur = ql.EURCurrency()
gbp = ql.GBPCurrency()
jpy = ql.JPYCurrency()
chf = ql.CHFCurrency()
cad = ql.CADCurrency()
aud = ql.AUDCurrency()
nzd = ql.NZDCurrency()
sek = ql.SEKCurrency()
nok = ql.NOKCurrency()

# Emerging market currencies
brl = ql.BRLCurrency()  # Brazilian Real
cny = ql.CNYCurrency()  # Chinese Yuan
inr = ql.INRCurrency()  # Indian Rupee
krw = ql.KRWCurrency()  # Korean Won
mxn = ql.MXNCurrency()  # Mexican Peso
zar = ql.ZARCurrency()  # South African Rand
```

### Currency Properties

```python
currency = ql.USDCurrency()

# Basic properties
print(f"Name: {currency.name()}")           # "U.S. dollar"
print(f"Code: {currency.code()}")           # "USD"
print(f"Numeric code: {currency.numericCode()}")  # 840
print(f"Symbol: {currency.symbol()}")       # "$"

# Minor currency properties
print(f"Minor unit symbol: {currency.minorUnitSymbol()}")  # "¢" (cents)
print(f"Fractional symbol: {currency.fractionalSymbol()}")

# Currency comparison
eur = ql.EURCurrency()
print(f"Same currency: {currency == eur}")  # False
print(f"Different currency: {currency != eur}")  # True
```

### Currency Triangulation

For currencies that need to be converted through a common base.

```python
# Some currency pairs trade through USD triangulation
# This is handled automatically by QuantLib's currency conversion
```

## Exchange Rate Management

### Exchange Rate

Represent FX rates between currency pairs.

```python
# Direct exchange rate: 1 USD = 0.85 EUR
usd = ql.USDCurrency()
eur = ql.EURCurrency()
exchange_rate = ql.ExchangeRate(usd, eur, 0.85)

# Convert amounts
usd_amount = 1000.0
eur_amount = exchange_rate.exchange(ql.Money(usd_amount, usd)).amount()
print(f"{usd_amount} USD = {eur_amount} EUR")

# Rate properties
print(f"Source: {exchange_rate.source().code()}")  # "USD"
print(f"Target: {exchange_rate.target().code()}")  # "EUR"  
print(f"Rate: {exchange_rate.rate()}")             # 0.85
```

### ExchangeRateManager

Global registry for exchange rates.

```python
# Get singleton instance
rate_manager = ql.ExchangeRateManager.instance()

# Add exchange rate
usd_eur_rate = ql.ExchangeRate(ql.USDCurrency(), ql.EURCurrency(), 0.85)
rate_manager.add(usd_eur_rate)

# Lookup exchange rate
retrieved_rate = rate_manager.lookup(ql.USDCurrency(), ql.EURCurrency())
print(f"USD/EUR rate: {retrieved_rate.rate()}")

# Clear all rates
rate_manager.clear()
```

## Money Class

Represent monetary amounts with currency.

```python
# Create money amounts
usd_amount = ql.Money(1000.0, ql.USDCurrency())
eur_amount = ql.Money(850.0, ql.EURCurrency())

# Money properties
print(f"Amount: {usd_amount.value()}")         # 1000.0
print(f"Currency: {usd_amount.currency().code()}")  # "USD"

# Money arithmetic (same currency)
usd_amount2 = ql.Money(500.0, ql.USDCurrency())
total_usd = usd_amount + usd_amount2           # 1500 USD
difference = usd_amount - usd_amount2          # 500 USD
scaled = usd_amount * 1.5                      # 1500 USD

# Comparison
print(f"Equal: {usd_amount == usd_amount2}")   # False
print(f"Greater: {usd_amount > usd_amount2}")  # True

# Currency conversion requires exchange rates to be set up
# usd_plus_eur = usd_amount + eur_amount  # Would need exchange rate
```

## Market Data Observables

### Observable Pattern

QuantLib implements the Observer pattern for automatic updates.

```python
# SimpleQuote is Observable
quote = ql.SimpleQuote(0.05)

# Term structures are Observers that depend on quotes
curve = ql.FlatForward(2, ql.TARGET(), ql.QuoteHandle(quote), ql.Actual360())

# When quote changes, curve automatically recalculates
initial_discount = curve.discount(ql.Date(15, 6, 2024))
quote.setValue(0.055)  # Update quote
new_discount = curve.discount(ql.Date(15, 6, 2024))  # Automatically uses new rate

print(f"Discount factor change: {(new_discount/initial_discount - 1)*100:.2f}%")
```

## Usage Examples

### Multi-Currency Bond Portfolio

```python
import QuantLib as ql

# Setup evaluation date
ql.Settings.instance().evaluationDate = ql.Date(15, 6, 2023)

# Create bonds in different currencies
usd_bond_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("6M"),
    calendar=ql.UnitedStates()
)

eur_bond_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2030),
    tenor=ql.Period("1Y"),
    calendar=ql.TARGET()
)

# USD bond
usd_bond = ql.FixedRateBond(
    settlementDays=2,
    faceAmount=100.0,
    schedule=usd_bond_schedule,
    coupons=[0.04],
    accrualDayCounter=ql.ActualActual(),
    paymentConvention=ql.Following
)

# EUR bond  
eur_bond = ql.FixedRateBond(
    settlementDays=2,
    faceAmount=100.0,
    schedule=eur_bond_schedule,
    coupons=[0.025],
    accrualDayCounter=ql.ActualActual(),
    paymentConvention=ql.Following
)

# Create yield curves
usd_curve = ql.FlatForward(0, ql.UnitedStates(), 0.045, ql.Actual360())
eur_curve = ql.FlatForward(0, ql.TARGET(), 0.02, ql.Actual360())

# Price bonds in their native currencies
usd_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(usd_curve))
eur_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(eur_curve))

usd_bond.setPricingEngine(usd_engine)
eur_bond.setPricingEngine(eur_engine)

usd_price = usd_bond.cleanPrice()
eur_price = eur_bond.cleanPrice()

# Portfolio values in native currencies
usd_position = ql.Money(usd_price * 1000, ql.USDCurrency())  # 1000 bonds
eur_position = ql.Money(eur_price * 1500, ql.EURCurrency())  # 1500 bonds

print(f"USD position: {usd_position.value():.2f} {usd_position.currency().code()}")
print(f"EUR position: {eur_position.value():.2f} {eur_position.currency().code()}")

# To aggregate in single currency, need exchange rate
usd_eur_rate = 0.85
rate_manager = ql.ExchangeRateManager.instance()
rate_manager.add(ql.ExchangeRate(ql.USDCurrency(), ql.EURCurrency(), usd_eur_rate))

# Convert USD position to EUR
eur_equivalent = usd_position.value() * usd_eur_rate
total_eur = eur_position.value() + eur_equivalent

print(f"Total portfolio value: {total_eur:.2f} EUR")
```

### Interest Rate Index Curve Building

```python
# Build EURIBOR curve from market rates
market_data = {
    'deposits': [
        ('1W', 0.025),
        ('1M', 0.026),
        ('3M', 0.028)
    ],
    'swaps': [
        ('2Y', 0.030),
        ('5Y', 0.032),
        ('10Y', 0.035)
    ]
}

# Create rate helpers
helpers = []
calendar = ql.TARGET()
euribor_3m = ql.Euribor3M()

# Deposit helpers
for tenor_str, rate in market_data['deposits']:
    tenor = ql.Period(tenor_str)
    helper = ql.DepositRateHelper(
        rate, tenor, 2, calendar, ql.ModifiedFollowing, True, ql.Actual360()
    )
    helpers.append(helper)

# Swap helpers
for tenor_str, rate in market_data['swaps']:
    tenor = ql.Period(tenor_str)
    helper = ql.SwapRateHelper(
        rate, tenor, calendar, ql.Annual, ql.ModifiedFollowing,
        ql.Thirty360(), euribor_3m
    )
    helpers.append(helper)

# Bootstrap curve
curve = ql.PiecewiseLogLinearDiscount(ql.Date(15, 6, 2023), helpers, ql.Actual360())
curve.enableExtrapolation()

# Link EURIBOR index to forecasting curve
curve_handle = ql.YieldTermStructureHandle(curve)
euribor_3m_with_curve = ql.Euribor3M(curve_handle)

# Test forward rates
test_dates = [
    ql.Date(15, 9, 2023),   # 3M forward
    ql.Date(15, 12, 2023),  # 6M forward
    ql.Date(15, 6, 2024)    # 1Y forward
]

for date in test_dates:
    if euribor_3m_with_curve.isValidFixingDate(date):
        forward_rate = euribor_3m_with_curve.fixing(date)
        print(f"3M EURIBOR forward to {date}: {forward_rate:.4f}")
```

### Dynamic Rate Environment

```python
# Create dynamic rate environment that updates automatically
base_rate_quote = ql.SimpleQuote(0.03)
spread_quote = ql.SimpleQuote(0.02)

# Create relinkable handles
base_handle = ql.QuoteHandle(base_rate_quote)
spread_handle = ql.QuoteHandle(spread_quote)

# Build curves that depend on these quotes
risk_free_curve = ql.FlatForward(2, ql.TARGET(), base_handle, ql.Actual360())
credit_curve = ql.ZeroSpreadedTermStructure(
    ql.YieldTermStructureHandle(risk_free_curve),
    spread_handle
)

# Create instrument that depends on credit curve
bond_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("1Y"),
    calendar=ql.TARGET()
)

bond = ql.FixedRateBond(
    settlementDays=2,
    faceAmount=100.0,
    schedule=bond_schedule,
    coupons=[0.045],
    accrualDayCounter=ql.Actual360()
)

bond.setPricingEngine(ql.DiscountingBondEngine(ql.YieldTermStructureHandle(credit_curve)))

# Initial valuation
initial_price = bond.cleanPrice()
print(f"Initial bond price: {initial_price:.3f}")

# Scenario analysis: rates up 50bp
base_rate_quote.setValue(0.035)  # Risk-free rate up 50bp
rate_up_price = bond.cleanPrice()

# Credit spread tightens 25bp  
spread_quote.setValue(0.0175)    # Spread down 25bp
final_price = bond.cleanPrice()

print(f"After rate rise: {rate_up_price:.3f}")
print(f"After spread tightening: {final_price:.3f}")
print(f"Net price change: {final_price - initial_price:.3f}")
```

### Cross-Currency Swap Analytics

```python
# Create cross-currency swap (USD/EUR)
notional_usd = 1000000.0
notional_eur = 850000.0  # At initial exchange rate of 1.18

# USD leg (floating)
usd_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("3M"),
    calendar=ql.UnitedStates()
)

# EUR leg (fixed)
eur_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("6M"),
    calendar=ql.TARGET()
)

# Market curves
usd_curve = ql.FlatForward(0, ql.UnitedStates(), 0.04, ql.Actual360())
eur_curve = ql.FlatForward(0, ql.TARGET(), 0.025, ql.Actual360())

# USD LIBOR index
usd_libor_3m = ql.USDLibor(ql.Period("3M"), ql.YieldTermStructureHandle(usd_curve))

# Create legs (simplified - actual implementation more complex)
usd_floating_leg = ql.IborLeg([notional_usd], usd_schedule, usd_libor_3m)
usd_floating_leg = usd_floating_leg.withNotionals([notional_usd])

eur_fixed_leg = ql.FixedRateLeg(eur_schedule)
eur_fixed_leg = eur_fixed_leg.withNotionals([notional_eur]).withCouponRates([0.025], ql.Actual360())

# Value legs separately in their currencies
usd_leg_npv = ql.CashFlows.npv(usd_floating_leg, ql.YieldTermStructureHandle(usd_curve), ql.Date(15, 6, 2023))
eur_leg_npv = ql.CashFlows.npv(eur_fixed_leg, ql.YieldTermStructureHandle(eur_curve), ql.Date(15, 6, 2023))

print(f"USD floating leg NPV: ${usd_leg_npv:,.2f}")
print(f"EUR fixed leg NPV: €{eur_leg_npv:,.2f}")

# For cross-currency swap valuation, need FX forwards/volatility
# This requires more sophisticated modeling
```