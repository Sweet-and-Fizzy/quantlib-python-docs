# QuantLib Python Documentation

This directory contains comprehensive, structured documentation for the QuantLib Python bindings, organized by functional categories.

## Version Information

This documentation is based on **QuantLib 1.40.0** (development version) and the corresponding QuantLib-SWIG Python bindings.

**Important Links:**
- [QuantLib Official Website](https://www.quantlib.org/)
- [QuantLib GitHub Repository](https://github.com/lballabio/QuantLib)
- [QuantLib-SWIG Python Bindings](https://github.com/lballabio/QuantLib-SWIG)
- [Official QuantLib Documentation](https://quantlib-python-docs.readthedocs.io/)

**Compatibility:** This documentation is written for Python 3.6+ and requires the QuantLib Python package to be installed.

## Documentation Structure

Each document follows a consistent API reference style with:
- **Table of Contents** for easy navigation
- **Inheritance chains** showing parent classes
- **Complete constructor listings** with all available signatures
- **Method documentation** grouped by inheritance source
- **Usage examples** showing practical implementation

### Core Documentation Files

- **[Date & Time](01-date-time.md)** - Date, Calendar, Schedule, Period, DayCounter classes
- **[Interest Rates & Curves](02-interest-rates.md)** - Term structures, yield curves, interest rate models  
- **[Bonds & Fixed Income](03-bonds.md)** - Bond instruments, cash flows, swaps
- **[Options & Derivatives](04-options.md)** - Option pricing, payoffs, exercise styles
- **[Mathematical Tools](05-math.md)** - Arrays, matrices, interpolation, optimization
- **[Market Data](06-market-data.md)** - Quotes, handles, indexes, currencies
- **[Volatility & Models](07-volatility.md)** - Volatility surfaces, stochastic processes, pricing models

## Key Features

- **Complete API Coverage**: All 83 SWIG interface files analyzed
- **Practical Examples**: Real-world usage patterns and code samples
- **Python Integration**: SWIG-specific features and Python enhancements
- **Clear Organization**: Grouped by financial domain and use case

## Quick Start Examples

### Basic Date Operations
```python
import QuantLib as ql

today = ql.Date.todaysDate()
calendar = ql.TARGET()
maturity = calendar.advance(today, ql.Period(1, ql.Years))
```

### Building a Yield Curve
```python
helpers = [ql.DepositRateHelper(0.02, ql.Period(3, ql.Months), 2, calendar, ql.ModifiedFollowing, True, day_counter)]
curve = ql.PiecewiseLogLinearDiscount(2, calendar, helpers, day_counter)
```

### Option Pricing
```python
payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
exercise = ql.EuropeanExercise(maturity)
option = ql.VanillaOption(payoff, exercise)
```

## Contributing

This documentation is generated from analysis of the QuantLib-SWIG source code. 

**Report Issues**: Found an error or have suggestions? Please report issues at: https://github.com/Sweet-and-Fizzy/quantlib-python-docs/issues

**Pull Requests Welcome**: Contributions and improvements to the documentation are encouraged! Please submit pull requests to the same repository.

For QuantLib library issues, see the main [QuantLib project](https://github.com/lballabio/QuantLib/issues).