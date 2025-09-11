# Date and Time Classes

This document covers QuantLib's core date and time handling classes, essential for all financial calculations.

## ⚠️ Important Setup Note

**Always set the global evaluation date first** when using QuantLib:
```python
import QuantLib as ql
ql.Settings.instance().evaluationDate = ql.Date(15, 6, 2023)  # Set your "today"
```
This global date is used by all QuantLib calculations. Many errors occur when this isn't set properly.

## Date Class

The `Date` class is QuantLib's fundamental date representation with optional intraday support.

### Constructors

```python
# Null/empty date
Date()

# Basic date creation
Date(day, month, year)
Date(15, 6, 2023)  # June 15, 2023

# Intraday dates (if compiled with intraday support)
Date(day, month, year, hours, minutes, seconds, millisec=0, microsec=0)
Date(15, 6, 2023, 14, 30, 0)  # June 15, 2023 at 2:30 PM

# From serial number (days since epoch)
Date(serial_number)

# String parsing
Date(date_string, format_string)
Date("2023-06-15", "%Y-%m-%d")
```

### Key Methods

#### Date Components
```python
date = ql.Date(15, 6, 2023)
date.dayOfMonth()    # 15
date.month()         # 6 (or ql.June)
date.year()          # 2023
date.weekday()       # ql.Thursday
date.dayOfYear()     # Day of year (1-366)
```

#### Formatting and Conversion
```python
date.serialNumber()  # Internal representation
date.weekdayNumber() # Weekday as integer (1=Sunday)
date.ISO()          # ISO format string "2023-06-15"

# Python integration
date.to_date()      # Convert to Python datetime.date
Date.from_date(python_date)  # Create from Python datetime.date
```

#### Date Arithmetic
```python
date = ql.Date(15, 6, 2023)

# Add/subtract days
new_date = date + 30
days_diff = date2 - date1

# Add periods
future_date = date + ql.Period(6, ql.Months)
past_date = date - ql.Period(1, ql.Years)
```

#### Comparison Operators
```python
date1 == date2
date1 != date2  
date1 < date2
date1 <= date2
date1 > date2
date1 >= date2
```

### Static Methods

```python
# Current system date
ql.Date.todaysDate()

# Date limits
ql.Date.minDate()  # Minimum representable date
ql.Date.maxDate()  # Maximum representable date

# Utility functions
ql.Date.isLeap(2024)  # True for leap years
ql.Date.endOfMonth(date)    # Last day of month
ql.Date.startOfMonth(date)  # First day of month

# Weekday operations
ql.Date.nextWeekday(date, ql.Friday)  # Next Friday after date
ql.Date.nthWeekday(3, ql.Friday, ql.June, 2023)  # 3rd Friday of June 2023
```

## Period Class

Represents time periods like "3 months", "1 year", "2 weeks".

### Constructors

```python
# Null period
Period()

# From length and time unit
Period(3, ql.Months)
Period(1, ql.Years) 
Period(2, ql.Weeks)
Period(5, ql.Days)

# From frequency
Period(ql.Quarterly)  # 3 months
Period(ql.Annual)     # 1 year

# String parsing
Period("3M")   # 3 months
Period("1Y")   # 1 year
Period("2W")   # 2 weeks
Period("5D")   # 5 days
```

### Key Methods

```python
period = ql.Period(6, ql.Months)
period.length()    # 6
period.units()     # ql.Months
period.frequency() # ql.Semiannual (if applicable)

# Normalization (convert to canonical form)
ql.Period(24, ql.Months).normalized()  # 2Y
```

### Arithmetic Operations

```python
period1 = ql.Period(6, ql.Months)
period2 = ql.Period(1, ql.Years)

# Addition/subtraction
total = period1 + period2  # 18M

# Multiplication by integer
doubled = period1 * 2      # 12M

# Comparison
period1 == period2
period1 < period2
```

## Calendar Classes

Calendars handle business day calculations and holiday schedules for different markets.

### Base Calendar Methods

```python
calendar = ql.TARGET()  # Example: European Central Bank calendar

# Business day checks
calendar.isBusinessDay(date)
calendar.isHoliday(date)

# Date adjustment
calendar.adjust(date, ql.Following)  # Move to next business day
calendar.adjust(date, ql.Preceding) # Move to previous business day
calendar.adjust(date, ql.ModifiedFollowing)  # Following, unless crosses month

# Date advancement
future_date = calendar.advance(
    date, 
    ql.Period(3, ql.Months),
    ql.ModifiedFollowing,
    end_of_month=False
)

# Business day counting
days = calendar.businessDaysBetween(start_date, end_date, include_first=True, include_last=False)

# Holiday lists
holidays = calendar.holidayList(start_date, end_date, include_weekends=False)
business_days = calendar.businessDayList(start_date, end_date)
```

### Business Day Conventions

- `Following` - Move to next business day
- `ModifiedFollowing` - Following, unless it crosses month boundary
- `Preceding` - Move to previous business day  
- `ModifiedPreceding` - Preceding, unless it crosses month boundary
- `Unadjusted` - No adjustment

### Available Calendars

#### Major Markets
```python
ql.UnitedStates(ql.UnitedStates.NYSE)        # New York Stock Exchange
ql.UnitedStates(ql.UnitedStates.GovernmentBond)  # US Treasury
ql.UnitedStates(ql.UnitedStates.NERC)        # North American Electric Reliability Corp
ql.UnitedKingdom()                           # London Stock Exchange
ql.TARGET()                                  # European Central Bank
ql.Japan()                                   # Tokyo Stock Exchange
```

#### Americas
```python
ql.Brazil()                    # Brazil
ql.Canada(ql.Canada.TSX)       # Toronto Stock Exchange
ql.Mexico(ql.Mexico.BMV)       # Mexican Stock Exchange
ql.Argentina(ql.Argentina.Merval)  # Buenos Aires
ql.Chile()                     # Santiago Stock Exchange
```

#### Europe
```python
ql.Germany(ql.Germany.Eurex)   # Frankfurt/Eurex
ql.Germany(ql.Germany.Xetra)   # Xetra trading
ql.France()                    # Paris
ql.Italy()                     # Milan
ql.Switzerland()               # Zurich
ql.Sweden()                    # Stockholm
ql.Norway()                    # Oslo
ql.Denmark()                   # Copenhagen
```

#### Asia-Pacific
```python
ql.Australia(ql.Australia.ASX)  # Australian Securities Exchange
ql.HongKong()                   # Hong Kong Stock Exchange
ql.Singapore()                  # Singapore Exchange
ql.India(ql.India.NSE)          # National Stock Exchange of India
ql.SouthKorea(ql.SouthKorea.KRX) # Korea Exchange
ql.Taiwan()                     # Taiwan Stock Exchange
```

### Special Calendars

```python
# No holidays (only weekends)
ql.WeekendsOnly()

# No holidays, no weekends
ql.NullCalendar()

# Combine multiple calendars
joint_cal = ql.JointCalendar(ql.UnitedStates(), ql.UnitedKingdom())
joint_cal = ql.JointCalendar(ql.UnitedStates(), ql.UnitedKingdom(), ql.JointCalendar.JoinHolidays)

# Custom calendar
bespoke = ql.BespokeCalendar()
bespoke.addWeekend(ql.Saturday)
bespoke.addWeekend(ql.Sunday)
bespoke.addHoliday(ql.Date(4, ql.July, 2023))  # Add July 4th as holiday
```

## DayCounter Classes

Day counters calculate year fractions and day counts between dates for interest calculations.

### Common Day Counters

```python
# Actual day count variants
ql.Actual360()                    # Actual/360
ql.Actual365Fixed()               # Actual/365 Fixed
ql.Actual366()                    # Actual/366
ql.ActualActual(ql.ActualActual.ISDA)      # Actual/Actual ISDA
ql.ActualActual(ql.ActualActual.Bond)      # Actual/Actual Bond
ql.ActualActual(ql.ActualActual.Historical) # Actual/Actual Historical

# 30/360 variants  
ql.Thirty360(ql.Thirty360.USA)           # 30/360 US
ql.Thirty360(ql.Thirty360.BondBasis)     # 30/360 Bond Basis
ql.Thirty360(ql.Thirty360.European)      # 30/360 European

# Business day counts
ql.Business252(calendar)          # 252 business days per year

# Special cases
ql.OneDayCounter()               # Always returns 1 day, 1/365 year
ql.SimpleDayCounter()            # Simple subtraction
```

### Key Methods

```python
dc = ql.Actual360()
start_date = ql.Date(15, 6, 2023)
end_date = ql.Date(15, 12, 2023)

# Calculate days between dates
days = dc.dayCount(start_date, end_date)  # Integer day count

# Calculate year fraction
year_frac = dc.yearFraction(start_date, end_date)  # Float year fraction

# Description
name = dc.name()  # "Actual/360"
```

## Schedule Class

Generates sequences of dates for payment schedules, coupon dates, etc.

### Constructors

```python
# From explicit date vector
dates = [ql.Date(15, 6, 2023), ql.Date(15, 12, 2023), ql.Date(15, 6, 2024)]
schedule = ql.Schedule(dates, calendar, ql.Following)

# Generated schedule
schedule = ql.Schedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period(6, ql.Months),
    calendar=ql.TARGET(),
    convention=ql.ModifiedFollowing,
    terminationDateConvention=ql.ModifiedFollowing,
    rule=ql.DateGeneration.Backward,
    endOfMonth=False,
    firstDate=ql.Date(),  # Optional
    nextToLastDate=ql.Date()  # Optional
)
```

### MakeSchedule Builder (Recommended)

The `MakeSchedule` class provides a more Python-friendly interface:

```python
schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028),
    tenor=ql.Period("6M"),
    calendar=ql.TARGET(),
    convention=ql.ModifiedFollowing
).backwards().endOfMonth().withFirstDate(ql.Date(15, 7, 2023))
```

### Date Generation Rules

- `DateGeneration.Forward` - Start from effective date, go forward
- `DateGeneration.Backward` - Start from termination date, go backward  
- `DateGeneration.Zero` - Single period from effective to termination
- `DateGeneration.ThirdWednesday` - Third Wednesday of each month
- `DateGeneration.Twentieth` - 20th of each month
- `DateGeneration.TwentiethIMM` - 20th of IMM months (Mar, Jun, Sep, Dec)

### Key Methods

```python
schedule = ql.MakeSchedule(...)

# Access dates
schedule.size()         # Number of dates
schedule.date(i)        # Get i-th date (0-indexed)
schedule.dates()        # Vector of all dates
schedule[i]             # Python indexing

# Schedule properties
schedule.calendar()
schedule.businessDayConvention()
schedule.terminationDateBusinessDayConvention()  
schedule.tenor()
schedule.rule()
schedule.endOfMonth()

# Truncate schedule
partial = schedule.until(cutoff_date)
partial = schedule.after(start_date)

# Check if date is in schedule
schedule.isRegular(i)   # True if i-th period is regular length
```

## Usage Examples

### Basic Date Operations

```python
import QuantLib as ql

# Create dates
today = ql.Date.todaysDate()
start = ql.Date(15, 6, 2023)
end = ql.Date(15, 6, 2028)

# Period arithmetic
maturity = today + ql.Period(1, ql.Years)
print(f"One year from today: {maturity}")

# Business day adjustment
calendar = ql.TARGET()
adjusted = calendar.adjust(ql.Date(17, 6, 2023), ql.Following)  # If Saturday, move to Monday
```

### Creating a Payment Schedule

```python
# Semi-annual payment schedule for 5-year bond
schedule = ql.MakeSchedule(
    effectiveDate=ql.Date(15, 6, 2023),
    terminationDate=ql.Date(15, 6, 2028), 
    tenor=ql.Period("6M"),
    calendar=ql.TARGET(),
    convention=ql.ModifiedFollowing
).backwards()

# Print all payment dates
for i in range(schedule.size()):
    print(f"Payment {i+1}: {schedule.date(i)}")
```

### Year Fraction Calculations

```python
# Calculate interest for different day count conventions
start = ql.Date(15, 1, 2023)
end = ql.Date(15, 7, 2023)

day_counters = [
    ("Act/360", ql.Actual360()),
    ("Act/365", ql.Actual365Fixed()),
    ("30/360", ql.Thirty360()),
    ("Act/Act ISDA", ql.ActualActual(ql.ActualActual.ISDA))
]

for name, dc in day_counters:
    year_frac = dc.yearFraction(start, end)
    print(f"{name}: {year_frac:.6f} years")
```

### Multi-Currency Calendar Operations

```python
# Joint calendar for US and UK holidays
us_calendar = ql.UnitedStates()
uk_calendar = ql.UnitedKingdom()
joint_calendar = ql.JointCalendar(us_calendar, uk_calendar)

# Check if date is business day in both countries
date = ql.Date(4, ql.July, 2023)  # US Independence Day
print(f"Business day in US: {us_calendar.isBusinessDay(date)}")      # False
print(f"Business day in UK: {uk_calendar.isBusinessDay(date)}")      # True  
print(f"Business day in both: {joint_calendar.isBusinessDay(date)}") # False
```