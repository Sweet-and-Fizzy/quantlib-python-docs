# Bonds and Fixed Income Instruments

This document covers QuantLib's bond instruments, cash flows, and fixed income analytics.

## Table of Contents
- [Bond Base Class](#bond-base-class)
- [ZeroCouponBond](#zerocouponbond)
- [FixedRateBond](#fixedratebond)
- [FloatingRateBond](#floatingratebond)
- [CallableBond and PuttableBond](#callablebond-and-puttablebond)
- [ConvertibleBond](#convertiblebond)
- [Cash Flow Classes](#cash-flow-classes)
- [Interest Rate Swaps](#interest-rate-swaps)
- [Bond Pricing Engines](#bond-pricing-engines)
- [Usage Examples](#usage-examples)

## Bond Base Class

**Inherits from**: `Instrument` → `LazyObject` → `Observable`

All bond instruments inherit from the `Bond` base class, which provides common functionality for pricing, yield calculations, and cash flow analysis.

### Constructors

```python
# Constructor with explicit cash flows
Bond(
    settlementDays,    # Natural: settlement delay in days
    calendar,         # Calendar: payment calendar
    faceAmount,       # Real: face/notional amount
    maturityDate,     # Date: bond maturity
    issueDate=Date(), # Date: issue date (optional)
    cashflows=Leg()   # Leg: vector of cash flows (optional)
)

# Constructor with just cash flows
Bond(
    settlementDays,    # Natural: settlement delay in days
    calendar,         # Calendar: payment calendar
    issueDate=Date(), # Date: issue date (optional)
    coupons=Leg()     # Leg: coupon cash flows
)
```

### Key Methods

#### Pricing and Valuation
```python
bond = ...  # Any Bond subclass

# Clean and dirty prices
bond.cleanPrice()           # Market price without accrued interest
bond.dirtyPrice()          # Price including accrued interest
bond.accruedAmount()       # Accrued interest at settlement
bond.accruedAmount(settlement_date)  # Accrued at specific date

# Set pricing engine for valuation
bond.setPricingEngine(pricing_engine)
bond.NPV()                 # Net present value
```

#### Yield Calculations
```python
# Yield to maturity
ytm = bond.yield_(
    day_counter=ql.ActualActual(),
    compounding=ql.Compounded,
    frequency=ql.Semiannual,
    accuracy=1e-8,
    max_evaluations=100,
    guess=0.05
)

# Yield from price
ytm_from_price = bond.yield_(
    clean_price=99.5,
    day_counter=ql.ActualActual(),
    compounding=ql.Compounded,
    frequency=ql.Semiannual
)
```

#### Risk Metrics
```python
# Duration and convexity
bond.duration()                    # Modified duration
bond.duration(ytm)                # Duration at specific yield
bond.convexity()                  # Convexity
bond.convexity(ytm)               # Convexity at specific yield

# From yield and settlement date
bond.duration(ytm, day_counter, compounding, frequency, settlement_date)
bond.convexity(ytm, day_counter, compounding, frequency, settlement_date)
```

#### Bond Properties
```python
bond.settlementDate()             # Settlement date for current evaluation date
bond.settlementDate(trade_date)   # Settlement for specific trade date
bond.cashflows()                  # Leg of cash flows
bond.maturityDate()              # Bond maturity
bond.issueDate()                 # Issue date
bond.faceAmount()                # Face/par value
bond.calendar()                  # Bond calendar
bond.settlementDays()            # Settlement period
```

## Specific Bond Types

### ZeroCouponBond

Pure discount bond with single payment at maturity.

```python
ZeroCouponBond(
    settlementDays,
    calendar,
    faceAmount,
    maturityDate,
    convention=ql.Following,
    redemption=100.0,
    issueDate=None
)

# Example
zero_bond = ql.ZeroCouponBond(
    settlementDays=3,
    calendar=ql.TARGET(),
    faceAmount=100.0,
    maturityDate=ql.Date(15, 6, 2030),
    convention=ql.Following,
    redemption=100.0,
    issueDate=ql.Date(15, 6, 2023)
)
```

## FixedRateBond

**Inherits from**: [`Bond`](#bond-base-class) → `Instrument` → `LazyObject` → `Observable`

Bond with fixed coupon payments throughout its life.

<details>
<summary><b>Inheritance Details</b> (click to expand)</summary>

- **From [`Bond`](#bond-base-class)**: `cleanPrice()`, `dirtyPrice()`, `bondYield()`, `accruedAmount()`, `duration()`, `convexity()`, `settlementDate()`, `maturityDate()`, `cashflows()`, etc.
- **From `Instrument`**: `NPV()`, `setPricingEngine()`, `isExpired()`, `errorEstimate()`
- **From `LazyObject`**: `calculate()`, `freeze()`, `unfreeze()`
- **From `Observable`**: Observer pattern support for automatic recalculation

</details>

### Constructors

```python
# Primary constructor - only first 5 parameters are required
FixedRateBond(
    settlementDays,                          # Integer: REQUIRED - settlement period in days
    faceAmount,                              # Real: REQUIRED - face/notional amount  
    schedule,                                # Schedule: REQUIRED - payment schedule
    coupons,                                 # List[Rate]: REQUIRED - coupon rates (can vary by period)
    paymentDayCounter,                       # DayCounter: REQUIRED - day count for coupon accrual
    paymentConvention=ql.Following,          # BusinessDayConvention: payment adjustment (default: Following)
    redemption=100.0,                        # Real: redemption amount at maturity (default: 100.0)
    issueDate=ql.Date(),                     # Date: bond issue date (default: null Date)
    paymentCalendar=ql.Calendar(),           # Calendar: payment calendar (default: null, uses schedule calendar)
    exCouponPeriod=ql.Period(),              # Period: ex-coupon period (default: no ex-coupon)
    exCouponCalendar=ql.Calendar(),          # Calendar: ex-coupon calendar (default: null)
    exCouponConvention=ql.Unadjusted,        # BusinessDayConvention: ex-coupon adjustment (default: Unadjusted)
    exCouponEndOfMonth=False                 # Boolean: ex-coupon end-of-month rule (default: False)
)

# Minimal constructor example - only required parameters
FixedRateBond(
    settlementDays=2,                        # Integer: settlement days
    faceAmount=100.0,                        # Real: face amount
    schedule=my_schedule,                    # Schedule: payment schedule (see Schedule class)
    coupons=[0.04],                          # List[Rate]: coupon rates
    paymentDayCounter=ql.ActualActual()      # DayCounter: day count convention
)

# Full constructor example - all parameters specified
FixedRateBond(
    settlementDays=3,                        # Integer: settlement days
    faceAmount=1000.0,                       # Real: face amount
    schedule=my_schedule,                    # Schedule: payment schedule
    coupons=[0.045],                         # List[Rate]: coupon rates
    paymentDayCounter=ql.ActualActual(ql.ActualActual.Bond),  # DayCounter: day count convention
    paymentConvention=ql.ModifiedFollowing,  # BusinessDayConvention: payment adjustment
    redemption=105.0,                        # Real: premium redemption
    issueDate=ql.Date(15, 6, 2023),          # Date: issue date
    paymentCalendar=ql.TARGET(),             # Calendar: payment calendar
    exCouponPeriod=ql.Period("5D"),          # Period: ex-coupon period
    exCouponCalendar=ql.TARGET(),            # Calendar: ex-coupon calendar
    exCouponConvention=ql.Following,         # BusinessDayConvention: ex-coupon adjustment
    exCouponEndOfMonth=True                  # Boolean: ex-coupon end-of-month rule
)
```

**Related Classes**: [`Schedule`](01-date-time.md#schedule-class), [`DayCounter`](01-date-time.md#daycounter-classes), [`Calendar`](01-date-time.md#calendar-classes), [`BusinessDayConvention`](01-date-time.md#business-day-conventions)

### Key Methods (inherited from Bond)

```python
bond = ql.FixedRateBond(...)

# Pricing methods (require pricing engine - see Bond Pricing Engines section)
bond.cleanPrice()                            # Real: market price without accrued interest
bond.dirtyPrice()                            # Real: price including accrued interest  
bond.accruedAmount()                         # Real: accrued interest at settlement
bond.NPV()                                   # Real: net present value (requires setPricingEngine())

# Yield calculations
bond.bondYield(                              # InterestRate: yield to maturity
    price,                                   # Real: clean price
    dayCounter,                              # DayCounter: yield day count convention
    compounding,                             # Compounding: Simple|Compounded|Continuous
    frequency                                # Frequency: Annual|Semiannual|Quarterly|etc
)

# Alternative yield calculation (from current market price)
bond.bondYield(
    dayCounter,                              # DayCounter: yield calculation day count
    compounding,                             # Compounding: compounding convention  
    frequency                                # Frequency: compounding frequency
)

# Risk metrics  
bond.duration()                              # Real: modified duration (from current price)
bond.convexity()                             # Real: convexity (from current price)
bond.duration(ytm, dayCounter, compounding, frequency)    # Real: duration at specific yield
bond.convexity(ytm, dayCounter, compounding, frequency)   # Real: convexity at specific yield

# Bond properties
bond.settlementDate()                        # Date: settlement date for current evaluation date
bond.settlementDate(trade_date)              # Date: settlement date for specific trade date
bond.maturityDate()                          # Date: bond maturity
bond.issueDate()                             # Date: bond issue date
bond.faceAmount()                            # Real: face/par value  
bond.cashflows()                             # Leg: vector of cash flows (see Cash Flow Classes)
bond.notional(date=ql.Date())                # Real: notional at specific date
bond.calendar()                              # Calendar: bond's payment calendar

# Cash flow analysis
bond.nextCashFlowDate(date=ql.Date())        # Date: next coupon date after given date  
bond.previousCashFlowDate(date=ql.Date())    # Date: previous coupon date before given date
bond.nextCashFlowAmount(date=ql.Date())      # Real: next coupon amount
bond.previousCashFlowAmount(date=ql.Date())  # Real: previous coupon amount
```

**See Also**: [`InterestRate`](02-interest-rates.md#interestrate-class), [`Compounding`](02-interest-rates.md#compounding-types), [`Frequency`](02-interest-rates.md#frequency-types), [Cash Flow Classes](#cash-flow-classes), [Bond Pricing Engines](#bond-pricing-engines)

### FixedRateBond-Specific Methods

```python
# Additional methods specific to FixedRateBond
bond.frequency()                   # Frequency: coupon payment frequency
bond.dayCounter()                 # DayCounter: coupon day counter
```

### Usage Example

```python
import QuantLib as ql

# Setup
ql.Settings.instance().evaluationDate = ql.Date(15, 6, 2023)

# Create payment schedule
schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("6M"),                    # Semi-annual payments
    calendar=ql.TARGET(),
    convention=ql.ModifiedFollowing
)

# Create bond
fixed_bond = ql.FixedRateBond(
    settlementDays=3,
    faceAmount=100.0,
    schedule=schedule,
    coupons=[0.04],                                    # 4% annual coupon
    paymentDayCounter=ql.ActualActual(ql.ActualActual.Bond),  # REQUIRED parameter
    paymentConvention=ql.ModifiedFollowing,
    redemption=100.0
)

# Set pricing engine and get price
yield_curve = ql.FlatForward(2, ql.TARGET(), 0.03, ql.Actual365Fixed())
engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(yield_curve))
fixed_bond.setPricingEngine(engine)

price = fixed_bond.cleanPrice()
print(f"Bond price: {price:.3f}")
```

#### Step-up/Step-down Bonds

Bonds with changing coupon rates over time.

```python
# Different coupon rates for different periods
coupons = [0.03, 0.035, 0.04, 0.045]  # Stepping up each year
stepup_bond = ql.FixedRateBond(
    settlementDays=3, 
    faceAmount=100.0, 
    schedule=schedule, 
    coupons=coupons, 
    paymentDayCounter=ql.Thirty360(),
    paymentConvention=ql.ModifiedFollowing
)
```

### FloatingRateBond

Bond with variable coupon payments linked to a reference rate.

```python
FloatingRateBond(
    settlementDays,                    # Size: settlement period in days
    faceAmount,                        # Real: face/notional amount
    schedule,                          # Schedule: payment schedule
    index,                            # IborIndex: reference index (e.g., LIBOR)
    paymentDayCounter,                # DayCounter: REQUIRED day count for payments
    paymentConvention=ql.Following,   # Business day convention
    fixingDays=None,                  # Size: fixing days (None = index default)
    gearings=[],                      # List[Real]: multipliers for index rate
    spreads=[],                       # List[Spread]: spreads over index
    caps=[],                          # List[Rate]: rate caps
    floors=[],                        # List[Rate]: rate floors
    inArrears=False,                  # Boolean: in-arrears fixing
    redemption=100.0,                 # Real: redemption amount
    issueDate=ql.Date()               # Date: issue date
)

# Example - Floating rate note linked to 6M EURIBOR
import QuantLib as ql

euribor6m = ql.Euribor6M()
floating_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("6M"),
    calendar=ql.TARGET()
)

frn = ql.FloatingRateBond(
    settlementDays=2,
    faceAmount=100.0,
    schedule=floating_schedule,
    index=euribor6m,
    paymentDayCounter=ql.Actual360(),           # REQUIRED parameter
    paymentConvention=ql.ModifiedFollowing,
    fixingDays=2,
    gearings=[1.0],     # 1x EURIBOR
    spreads=[0.002],    # +20bp spread
    caps=[],            # No cap
    floors=[0.0],       # 0% floor
    inArrears=False
)
```

### CallableBond and PuttableBond

Bonds with embedded options.

```python
# Callable bond
CallableFixedRateBond(
    settlementDays,
    faceAmount,
    schedule,
    coupons,
    accrualDayCounter,
    paymentConvention,
    redemption,
    issueDate,
    callability  # CallabilitySchedule
)

# Puttable bond  
PuttableFixedRateBond(
    settlementDays,
    faceAmount,
    schedule,
    coupons,
    accrualDayCounter,
    paymentConvention,
    redemption,
    issueDate,
    putability  # CallabilitySchedule (put rights)
)

# Example with call schedule
call_dates = [ql.Date(15, 6, 2025), ql.Date(15, 6, 2026), ql.Date(15, 6, 2027)]
call_prices = [102.0, 101.0, 100.5]

call_schedule = ql.CallabilitySchedule()
for date, price in zip(call_dates, call_prices):
    call_schedule.append(
        ql.Callability(
            price=ql.BondPrice(price, ql.BondPrice.Clean),
            type_=ql.Callability.Call,
            date=date
        )
    )

callable_bond = ql.CallableFixedRateBond(
    settlementDays=3,
    faceAmount=100.0,
    schedule=schedule,
    coupons=[0.045],
    accrualDayCounter=ql.Thirty360(),
    paymentConvention=ql.ModifiedFollowing,
    redemption=100.0,
    issueDate=ql.Date(15, 6, 2023),
    callability=call_schedule
)
```

### ConvertibleBond

Bonds convertible to equity.

```python
# Requires more complex setup with equity process
convertible = ql.ConvertibleFixedCouponBond(
    exercise=exercise_schedule,
    conversionRatio=conversion_ratio,
    dividends=[],
    callability=call_schedule,
    creditSpread=credit_spread_handle,
    issueDate=issue_date,
    settlementDays=settlement_days,
    dayCounter=day_counter,
    schedule=coupon_schedule,
    coupons=coupon_rates
)
```

## Cash Flow Classes

Cash flows represent individual payments from bonds and other instruments.

### SimpleCashFlow

Basic cash flow with fixed amount and date.

```python
SimpleCashFlow(amount, date)

# Example
redemption = ql.SimpleCashFlow(100.0, ql.Date(15, 6, 2028))
```

### Coupon (Base Class)

Base class for interest payments.

```python
# All coupons have these methods
coupon.amount()                # Coupon amount
coupon.date()                 # Payment date
coupon.accrualStartDate()     # Accrual period start
coupon.accrualEndDate()       # Accrual period end
coupon.accrualPeriod()        # Accrual period length
coupon.accrualDays()          # Days in accrual period
coupon.dayCounter()           # Day count convention
coupon.rate()                 # Coupon rate
coupon.nominal()              # Notional amount
```

### FixedRateCoupon

Fixed rate interest payment.

```python
FixedRateCoupon(
    paymentDate,
    nominal,
    rate,
    dayCounter,
    accrualStartDate,
    accrualEndDate,
    refPeriodStart=None,
    refPeriodEnd=None
)

# Example
coupon = ql.FixedRateCoupon(
    paymentDate=ql.Date(15, 12, 2023),
    nominal=100.0,
    rate=0.04,
    dayCounter=ql.ActualActual(),
    accrualStartDate=ql.Date(15, 6, 2023),
    accrualEndDate=ql.Date(15, 12, 2023)
)

coupon_amount = coupon.amount()  # Calculate coupon payment
```

### IborCoupon

Floating rate coupon linked to IBOR index.

```python
IborCoupon(
    paymentDate,
    nominal,
    startDate,
    endDate,
    fixingDays,
    index,           # IborIndex
    gearing=1.0,
    spread=0.0,
    refPeriodStart=None,
    refPeriodEnd=None,
    dayCounter=None,
    isInArrears=False
)

# Example
euribor3m = ql.Euribor3M()
ibor_coupon = ql.IborCoupon(
    paymentDate=ql.Date(15, 9, 2023),
    nominal=100.0,
    startDate=ql.Date(15, 6, 2023),
    endDate=ql.Date(15, 9, 2023),
    fixingDays=2,
    index=euribor3m,
    gearing=1.0,
    spread=0.001  # 10bp spread
)
```

### CappedFlooredCoupon

Floating coupon with cap and/or floor.

```python
CappedFlooredCoupon(
    underlying_coupon,  # IborCoupon or other floating coupon
    cap=None,          # Maximum rate
    floor=None         # Minimum rate
)

# Example - LIBOR coupon with 5% cap and 1% floor
underlying = ql.IborCoupon(...)
capped_coupon = ql.CappedFlooredCoupon(underlying, cap=0.05, floor=0.01)
```

### CmsCoupon

Coupon linked to Constant Maturity Swap rate.

```python
CmsCoupon(
    paymentDate,
    nominal,
    startDate,
    endDate,
    fixingDays,
    swapIndex,       # SwapIndex
    gearing=1.0,
    spread=0.0,
    refPeriodStart=None,
    refPeriodEnd=None,
    dayCounter=None,
    isInArrears=False
)
```

## Cash Flow Analysis

### Leg Class

Collection of cash flows forming one side of a swap or bond.

```python
# Access cash flows
leg = bond.cashflows()  # Returns Leg object

# Iterate through cash flows
for i, cf in enumerate(leg):
    print(f"Cash flow {i}: {cf.amount()} on {cf.date()}")
    
    # Check if it's a coupon
    if hasattr(cf, 'rate'):
        coupon = ql.as_coupon(cf)  # Cast to coupon
        print(f"  Rate: {coupon.rate():.4f}")

# Leg properties
leg.size()              # Number of cash flows
npv = ql.CashFlows.npv(leg, yield_curve, settlement_date)
```

### Cash Flow Functions

Utility functions for analyzing cash flow streams.

```python
# NPV calculations
npv = ql.CashFlows.npv(cashflows, discount_curve, settlement_date)
npv_yield = ql.CashFlows.npv(cashflows, yield_rate, day_counter, compounding, frequency, settlement_date)

# BPS (basis point sensitivity)
bps = ql.CashFlows.bps(cashflows, yield_curve, settlement_date)

# Duration and convexity
duration = ql.CashFlows.duration(cashflows, yield_rate, day_counter, compounding, frequency, settlement_date, ql.Duration.Modified)
convexity = ql.CashFlows.convexity(cashflows, yield_rate, day_counter, compounding, frequency, settlement_date)

# Yield calculation
yield_rate = ql.CashFlows.yield_(cashflows, present_value, day_counter, compounding, frequency, settlement_date)

# Z-spread calculation
z_spread = ql.CashFlows.zSpread(cashflows, present_value, discount_curve, day_counter, compounding, frequency, settlement_date)
```

## Interest Rate Swaps

Interest rate swaps exchange fixed and floating rate cash flows.

### VanillaSwap

Standard fixed-for-floating interest rate swap.

```python
VanillaSwap(
    type_,              # ql.VanillaSwap.Payer or ql.VanillaSwap.Receiver
    nominal,
    fixedSchedule,      # Fixed leg payment schedule
    fixedRate,
    fixedDayCounter,
    floatSchedule,      # Floating leg payment schedule  
    index,             # Floating rate index
    spread,            # Spread over index
    floatingDayCounter,
    paymentConvention=ql.Following
)

# Example - 5-year USD swap, pay 3% fixed vs 3M LIBOR
fixed_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("1Y"),
    calendar=ql.UnitedStates(),
    convention=ql.ModifiedFollowing
)

float_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("3M"),
    calendar=ql.UnitedStates(),
    convention=ql.ModifiedFollowing
)

usd_libor_3m = ql.USDLibor(ql.Period("3M"))

swap = ql.VanillaSwap(
    type_=ql.VanillaSwap.Payer,  # Pay fixed, receive floating
    nominal=1000000.0,
    fixedSchedule=fixed_schedule,
    fixedRate=0.03,
    fixedDayCounter=ql.Thirty360(),
    floatSchedule=float_schedule,
    index=usd_libor_3m,
    spread=0.0,
    floatingDayCounter=ql.Actual360()
)

# Price the swap
discount_curve_handle = ql.YieldTermStructureHandle(discount_curve)
swap_engine = ql.DiscountingSwapEngine(discount_curve_handle)
swap.setPricingEngine(swap_engine)

swap_npv = swap.NPV()
fair_rate = swap.fairRate()  # Par fixed rate
fair_spread = swap.fairSpread()  # Par floating spread
```

### Swap Analytics

```python
# Swap legs
fixed_leg = swap.fixedLeg()
float_leg = swap.floatingLeg()

# Individual leg NPVs
fixed_leg_npv = swap.fixedLegNPV()  
float_leg_npv = swap.floatingLegNPV()

# BPS (01 sensitivity)
fixed_leg_bps = swap.fixedLegBPS()
float_leg_bps = swap.floatingLegBPS()

# Start and maturity dates
swap.startDate()
swap.maturityDate()
```

### MakeVanillaSwap Builder

Simplified swap construction.

```python
swap = ql.MakeVanillaSwap(
    swapTenor=ql.Period("5Y"),
    index=usd_libor_3m,
    fixedRate=0.03,
    forwardStart=ql.Period("0D")
).withDiscountingTermStructure(discount_curve_handle)

# Or receive fixed
receive_swap = ql.MakeVanillaSwap(
    swapTenor=ql.Period("10Y"),
    index=euribor_6m,
    fixedRate=0.025,
    forwardStart=ql.Period("1Y")
).receiveFixed().withDiscountingTermStructure(discount_curve_handle)
```

### OvernightIndexedSwap (OIS)

Swap against overnight index (e.g., SOFR, EONIA).

```python
OvernightIndexedSwap(
    type_,
    nominal,
    schedule,       # Payment schedule (typically annual)
    fixedRate,
    fixedDC,
    overnightIndex, # e.g., SOFR, EONIA
    spread=0.0,
    paymentLag=0,
    paymentAdjustment=ql.Following,
    paymentCalendar=None,
    telescopicValueDates=False
)

# Example - 2-year SOFR OIS
sofr = ql.Sofr()
ois_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2025),
    tenor=ql.Period("1Y"),
    calendar=ql.UnitedStates()
)

ois = ql.OvernightIndexedSwap(
    type_=ql.OvernightIndexedSwap.Payer,
    nominal=1000000.0,
    schedule=ois_schedule,
    fixedRate=0.025,
    fixedDC=ql.Actual360(),
    overnightIndex=sofr,
    spread=0.0
)
```

## Bond Pricing Engines

### DiscountingBondEngine

Standard bond pricing using discount curve.

```python
engine = ql.DiscountingBondEngine(discount_curve_handle)
bond.setPricingEngine(engine)
bond_npv = bond.NPV()
```

### TreeCallableFixedRateBondEngine

For callable/puttable bonds using interest rate trees.

```python
# Hull-White model
hw_model = ql.HullWhite(yield_curve_handle)

# Tree engine
tree_engine = ql.TreeCallableFixedRateBondEngine(
    model=hw_model,
    size=100,  # Tree size
    discount_curve=discount_curve_handle
)

callable_bond.setPricingEngine(tree_engine)
bond_price = callable_bond.cleanPrice()
```

### BlackCallableFixedRateBondEngine

Black model for callable bonds.

```python
black_engine = ql.BlackCallableFixedRateBondEngine(
    fwdYieldVol=vol_handle,
    discountCurve=discount_curve_handle
)
```

## Usage Examples

### Corporate Bond Analysis

```python
import QuantLib as ql

# Setup - ALWAYS set evaluation date first
ql.Settings.instance().evaluationDate = ql.Date(15, 6, 2023)
calendar = ql.TARGET()

# Create 5-year corporate bond with 4.5% coupon
schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("6M"),
    calendar=calendar,
    convention=ql.ModifiedFollowing
)

corporate_bond = ql.FixedRateBond(
    settlementDays=3,
    faceAmount=100.0,
    schedule=schedule,
    coupons=[0.045],
    accrualDayCounter=ql.ActualActual(ql.ActualActual.Bond)
)

# Create discount curve (Treasury + credit spread)
treasury_curve = ql.FlatForward(ql.Date(15, 6, 2023), 0.03, ql.Actual360())
credit_spread = 0.015  # 150bp credit spread
corporate_curve = ql.ZeroSpreadedTermStructure(
    ql.YieldTermStructureHandle(treasury_curve),
    ql.QuoteHandle(ql.SimpleQuote(credit_spread))
)

# Price the bond
engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(corporate_curve))
corporate_bond.setPricingEngine(engine)

# Analytics
clean_price = corporate_bond.cleanPrice()
ytm = corporate_bond.yield_(ql.ActualActual(), ql.Compounded, ql.Semiannual)
duration = corporate_bond.duration()
convexity = corporate_bond.convexity()

print(f"Clean Price: {clean_price:.3f}")
print(f"Yield to Maturity: {ytm.rate():.4f}")
print(f"Modified Duration: {duration:.3f}")
print(f"Convexity: {convexity:.3f}")
```

### Floating Rate Note Valuation

```python
# Create FRN linked to 3M EURIBOR
euribor3m = ql.Euribor3M()
frn_schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2026),
    tenor=ql.Period("3M"),
    calendar=ql.TARGET()
)

frn = ql.FloatingRateBond(
    settlementDays=2,
    faceAmount=100.0,
    schedule=frn_schedule,
    index=euribor3m,
    accrualDayCounter=ql.Actual360(),
    fixingDays=2,
    gearings=[1.0],
    spreads=[0.0025],  # 25bp spread
    caps=[0.06],       # 6% cap
    floors=[0.0]       # 0% floor
)

# Set up curve with forward rates
euribor_curve = ql.FlatForward(ql.Date(15, 6, 2023), 0.035, ql.Actual360())
euribor3m.addFixing(ql.Date(13, 6, 2023), 0.034)  # Current fixing

# Link index to curve
euribor_handle = ql.YieldTermStructureHandle(euribor_curve)
euribor3m = ql.Euribor3M(euribor_handle)

# Price FRN
discount_curve = ql.FlatForward(ql.Date(15, 6, 2023), 0.032, ql.Actual360())
frn_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(discount_curve))
frn.setPricingEngine(frn_engine)

frn_price = frn.cleanPrice()
print(f"FRN Clean Price: {frn_price:.3f}")

# Analyze individual coupons
for i, cf in enumerate(frn.cashflows()):
    if hasattr(cf, 'rate'):
        coupon = ql.as_coupon(cf)
        if hasattr(coupon, 'index'):  # Floating coupon
            ibor_coupon = ql.as_ibor_coupon(coupon)
            print(f"Coupon {i}: {ibor_coupon.rate():.4f} on {cf.date()}")
```

### Interest Rate Swap Pricing

```python
# 10-year USD IRS: Pay 3.5% fixed vs 3M LIBOR
usd_libor_3m = ql.USDLibor(ql.Period("3M"))

swap = ql.MakeVanillaSwap(
    swapTenor=ql.Period("10Y"),
    index=usd_libor_3m,
    fixedRate=0.035,
    forwardStart=ql.Period("0D")
)

# Set up curves
discount_curve = ql.FlatForward(ql.Date(15, 6, 2023), 0.04, ql.Actual360())
forecast_curve = ql.FlatForward(ql.Date(15, 6, 2023), 0.038, ql.Actual360())

# Link index to forecast curve
usd_libor_3m = ql.USDLibor(ql.Period("3M"), ql.YieldTermStructureHandle(forecast_curve))

# Price swap
swap_engine = ql.DiscountingSwapEngine(ql.YieldTermStructureHandle(discount_curve))
swap.setPricingEngine(swap_engine)

# Analytics
swap_npv = swap.NPV()
fair_rate = swap.fairRate()
fixed_leg_npv = swap.fixedLegNPV()
float_leg_npv = swap.floatingLegNPV()

print(f"Swap NPV: ${swap_npv:,.2f}")
print(f"Fair Fixed Rate: {fair_rate:.4f}")
print(f"Fixed Leg NPV: ${fixed_leg_npv:,.2f}")
print(f"Floating Leg NPV: ${float_leg_npv:,.2f}")
```

### Bond Portfolio Analysis

```python
# Create portfolio of bonds
bonds = []

# 2-year Treasury
treasury_2y = ql.FixedRateBond(
    settlementDays=1,
    faceAmount=100.0,
    schedule=ql.MakeSchedule(
        effectiveDate=ql.Date(15, 6, 2023),
        terminationDate=ql.Date(15, 6, 2025),
        tenor=ql.Period("6M"),
        calendar=ql.UnitedStates()
    ),
    coupons=[0.025],
    accrualDayCounter=ql.ActualActual()
)

# 10-year Treasury  
treasury_10y = ql.FixedRateBond(
    settlementDays=1,
    faceAmount=100.0,
    schedule=ql.MakeSchedule(
        effectiveDate=ql.Date(15, 6, 2023),
        terminationDate=ql.Date(15, 6, 2033),
        tenor=ql.Period("6M"),
        calendar=ql.UnitedStates()
    ),
    coupons=[0.04],
    accrualDayCounter=ql.ActualActual()
)

bonds = [treasury_2y, treasury_10y]
weights = [0.4, 0.6]  # Portfolio weights

# Price bonds
treasury_curve = ql.FlatForward(ql.Date(15, 6, 2023), 0.035, ql.Actual360())
engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(treasury_curve))

portfolio_value = 0.0
portfolio_duration = 0.0
portfolio_convexity = 0.0

for bond, weight in zip(bonds, weights):
    bond.setPricingEngine(engine)
    
    bond_price = bond.cleanPrice()
    bond_duration = bond.duration()
    bond_convexity = bond.convexity()
    
    portfolio_value += weight * bond_price
    portfolio_duration += weight * bond_duration * bond_price
    portfolio_convexity += weight * bond_convexity * bond_price

# Weight by market value
portfolio_duration /= portfolio_value
portfolio_convexity /= portfolio_value

print(f"Portfolio Price: {portfolio_value:.3f}")
print(f"Portfolio Duration: {portfolio_duration:.3f}")
print(f"Portfolio Convexity: {portfolio_convexity:.3f}")
```