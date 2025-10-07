# QuantLib Python Interpolation Reference

Complete reference for all interpolation classes available in QuantLib Python bindings.

## Table of Contents

- [Quick Reference](#quick-reference)
- [1D Interpolation Classes](#1d-interpolation-classes)
  - [Linear Interpolation](#linear-interpolation)
  - [Logarithmic Interpolation](#logarithmic-interpolation)
  - [Flat Interpolation](#flat-interpolation)
  - [Cubic Splines](#cubic-splines)
  - [Parabolic Interpolation](#parabolic-interpolation)
  - [Specialized Methods](#specialized-methods)
- [2D Interpolation Classes](#2d-interpolation-classes)
- [Extrapolation and Advanced Features](#extrapolation-and-advanced-features)
- [Choosing the Right Interpolation](#choosing-the-right-interpolation)

## Quick Reference

### Method Availability Matrix

| Interpolation Class | `()` | `derivative()` | `secondDerivative()` | `primitive()` | Monotonic | Log-space |
|---------------------|------|----------------|---------------------|---------------|-----------|-----------|
| **LinearInterpolation** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **LogLinearInterpolation** | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **BackwardFlatInterpolation** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **ForwardFlatInterpolation** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **CubicNaturalSpline** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **LogCubicNaturalSpline** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **MonotonicCubicNaturalSpline** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **MonotonicLogCubicNaturalSpline** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **KrugerCubic** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **KrugerLogCubic** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **FritschButlandCubic** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **FritschButlandLogCubic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Parabolic** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **LogParabolic** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **MonotonicParabolic** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **MonotonicLogParabolic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ConvexMonotoneInterpolation** | ✅ | ❌ | ❌ | ❌* | ✅ | ❌ |
| **LagrangeInterpolation** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **ChebyshevInterpolation** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

*ConvexMonotone has `primitive()` in C++ but not currently exposed in Python bindings.

### 2D Interpolation

| Class | Description | Use Case |
|-------|-------------|----------|
| **BilinearInterpolation** | Linear interpolation in 2D | Fast, simple surface interpolation |
| **BicubicSpline** | Cubic spline in 2D | Smooth surfaces, volatility surfaces |

---

## 1D Interpolation Classes

### Linear Interpolation

#### LinearInterpolation

**Description**: Simple linear interpolation between adjacent points.

**Constructor**:
```python
LinearInterpolation(x, y)
```

**Parameters**:
- `x`: Array of x-coordinates (must be strictly increasing)
- `y`: Array of y-values

**Methods**:
- `interp(x, allowExtrapolation=False)`: Evaluate at point x

**Properties**:
- ✅ Continuous (C⁰)
- ✅ Monotonic if data is monotonic
- ❌ First derivative is discontinuous at knots
- ⚡ Very fast
- 💾 Minimal memory

**Use Cases**:
- Default choice for simple interpolation
- When speed is critical
- When you don't need smooth derivatives
- Discount curve interpolation (simple cases)

**Example**:
```python
import QuantLib as ql

x = ql.Array([1.0, 2.0, 3.0, 4.0, 5.0])
y = ql.Array([1.0, 1.5, 1.8, 2.0, 2.1])

interp = ql.LinearInterpolation(x, y)
value = interp(2.5)  # Returns ~1.65
```

---

### Logarithmic Interpolation

#### LogLinearInterpolation

**Description**: Linear interpolation in log-space. Interpolates `log(y)` linearly, then exponentiates.

**Constructor**:
```python
LogLinearInterpolation(x, y)
```

**Mathematical Formula**:
```
y(x) = exp(linear_interp(x, log(y_data)))
```

**Properties**:
- ✅ Ensures positive values (as long as input y > 0)
- ✅ Monotonic if data is monotonic
- ✅ Suitable for exponential growth/decay
- ⚡ Very fast

**Use Cases**:
- Discount factors (always positive, exponential decay)
- Volatility surfaces (volatility must be positive)
- Interest rates (when enforcing positivity)
- Any data that grows/decays exponentially

**Example**:
```python
import QuantLib as ql

times = ql.Array([0.5, 1.0, 2.0, 5.0, 10.0])
discount_factors = ql.Array([0.99, 0.98, 0.95, 0.88, 0.75])

interp = ql.LogLinearInterpolation(times, discount_factors)
df_3y = interp(3.0)  # Interpolate discount factor at 3 years
```

**When to Use**:
- Prefer over LinearInterpolation when data is strictly positive
- Particularly good for discount curves
- Better than linear for exponentially decaying quantities

---

### Flat Interpolation

#### BackwardFlatInterpolation

**Description**: Step function that takes the value of the previous point.

**Constructor**:
```python
BackwardFlatInterpolation(x, y)
```

**Behavior**:
```
y(x) = y[i-1]  for  x[i-1] < x <= x[i]
```

**Properties**:
- ✅ Simple, no oscillations
- ✅ Preserves positivity
- ❌ Discontinuous at knot points
- ⚡ Extremely fast

**Use Cases**:
- Exchange-traded derivatives (LIBOR curve construction)
- When market convention requires backward flat
- Simple piecewise constant curves

**Example**:
```python
import QuantLib as ql

dates = ql.Array([1.0, 2.0, 3.0, 5.0])
rates = ql.Array([0.02, 0.025, 0.03, 0.035])

interp = ql.BackwardFlatInterpolation(dates, rates)
rate_2_5 = interp(2.5)  # Returns 0.025 (value at x=2.0)
```

#### ForwardFlatInterpolation

**Description**: Step function that takes the value of the next point.

**Constructor**:
```python
ForwardFlatInterpolation(x, y)
```

**Behavior**:
```
y(x) = y[i]  for  x[i-1] < x <= x[i]
```

**Use Cases**:
- Forward rate curves
- When market convention requires forward flat
- Simple piecewise constant extrapolation

**Example**:
```python
import QuantLib as ql

dates = ql.Array([1.0, 2.0, 3.0, 5.0])
forwards = ql.Array([0.02, 0.025, 0.03, 0.035])

interp = ql.ForwardFlatInterpolation(dates, forwards)
fwd_2_5 = interp(2.5)  # Returns 0.03 (value at x=3.0)
```

---

### Cubic Splines

Cubic splines provide smooth interpolation with continuous second derivatives.

#### CubicNaturalSpline

**Description**: Natural cubic spline with zero second derivatives at boundaries.

**Constructor**:
```python
CubicNaturalSpline(x, y)
```

**Properties**:
- ✅ Smooth (C² continuity)
- ✅ Continuous second derivative
- ❌ Can oscillate between points
- ❌ Not monotonic
- ❌ Can produce negative values

**Methods**:
- `interp(x, allowExtrapolation=False)`: Evaluate at x
- `derivative(x, allowExtrapolation=False)`: First derivative at x
- `secondDerivative(x, allowExtrapolation=False)`: Second derivative at x
- `primitive(x, allowExtrapolation=False)`: Integral from x[0] to x

**Use Cases**:
- When you need smooth derivatives
- Curve fitting where smoothness is more important than monotonicity
- Scientific/engineering applications

**Example**:
```python
import QuantLib as ql

x = ql.Array([0.0, 1.0, 2.0, 3.0, 4.0])
y = ql.Array([1.0, 2.5, 3.0, 3.2, 3.5])

spline = ql.CubicNaturalSpline(x, y)

# Evaluate at points
value = spline(1.5)
slope = spline.derivative(1.5)
curvature = spline.secondDerivative(1.5)
area = spline.primitive(2.0)  # Integral from 0 to 2
```

#### LogCubicNaturalSpline

**Description**: Natural cubic spline in log-space.

**Constructor**:
```python
LogCubicNaturalSpline(x, y)
```

**Mathematical Formula**:
```
y(x) = exp(cubic_spline(x, log(y_data)))
```

**Properties**:
- ✅ Smooth (C² in log-space)
- ✅ Ensures positive values
- ❌ Can oscillate

**Methods**: Same as CubicNaturalSpline (derivative, secondDerivative, primitive)

**Use Cases**:
- Smooth discount curves
- Volatility interpolation requiring smoothness
- Any positive-valued smooth curve

**Example**:
```python
import QuantLib as ql

times = ql.Array([0.5, 1.0, 2.0, 5.0, 10.0])
vols = ql.Array([0.15, 0.18, 0.20, 0.22, 0.21])

interp = ql.LogCubicNaturalSpline(times, vols)
vol_3y = interp(3.0)
vol_slope = interp.derivative(3.0)
```

#### MonotonicCubicNaturalSpline

**Description**: Cubic spline with monotonicity preservation (Hyman filter applied).

**Constructor**:
```python
MonotonicCubicNaturalSpline(x, y)
```

**Properties**:
- ✅ Smooth (C²)
- ✅ Monotonic (if input data is monotonic)
- ✅ No spurious oscillations
- ✅ Preserves monotonic trends

**Methods**: derivative, secondDerivative, primitive

**Use Cases**:
- Yield curves where monotonicity is required
- Forward rate curves
- Any application requiring both smoothness and monotonicity

**Example**:
```python
import QuantLib as ql

times = ql.Array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
rates = ql.Array([0.01, 0.015, 0.02, 0.025, 0.03, 0.032])

interp = ql.MonotonicCubicNaturalSpline(times, rates)
rate = interp(3.0)
```

#### MonotonicLogCubicNaturalSpline

**Description**: Monotonic cubic spline in log-space.

**Constructor**:
```python
MonotonicLogCubicNaturalSpline(x, y)
```

**Properties**:
- ✅ Smooth in log-space
- ✅ Monotonic
- ✅ Positive values guaranteed
- ✅ No oscillations

**Methods**: derivative, secondDerivative, primitive

**Use Cases**:
- Premium choice for discount curves
- Smooth, monotonic, positive volatility curves
- Best of both worlds: smoothness + monotonicity + positivity

**Example**:
```python
import QuantLib as ql

times = ql.Array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
discount_factors = ql.Array([0.998, 0.995, 0.98, 0.95, 0.87, 0.73])

interp = ql.MonotonicLogCubicNaturalSpline(times, discount_factors)
df = interp(3.0)
instantaneous_forward = -interp.derivative(3.0) / interp(3.0)
```

---

### Kruger Cubic Splines

Kruger method uses local information to determine derivatives, avoiding some issues with global splines.

#### KrugerCubic

**Description**: Kruger's local cubic interpolation method.

**Constructor**:
```python
KrugerCubic(x, y)
```

**Properties**:
- ✅ Smooth (C²)
- ✅ Local (changes to data affect nearby points only)
- ✅ Generally well-behaved

**Methods**: derivative, secondDerivative, primitive

**Use Cases**:
- Alternative to natural cubic splines
- When local control is desired
- Avoiding global effects of natural splines

#### KrugerLogCubic

**Description**: Kruger interpolation in log-space.

**Constructor**:
```python
KrugerLogCubic(x, y)
```

**Properties**:
- ✅ Smooth in log-space
- ✅ Positive values
- ✅ Local method

**Methods**: derivative, secondDerivative, primitive

---

### Fritsch-Butland Cubic Splines

Fritsch-Butland method ensures monotonicity by modifying derivatives.

#### FritschButlandCubic

**Description**: Monotonicity-preserving cubic interpolation (Fritsch & Butland, 1984).

**Constructor**:
```python
FritschButlandCubic(x, y)
```

**Properties**:
- ✅ C¹ continuity (smooth first derivative)
- ✅ Monotonic (guaranteed)
- ✅ No overshooting
- ✅ Well-tested, widely used

**Methods**: derivative, secondDerivative, primitive

**Use Cases**:
- Industry standard for monotonic interpolation
- Yield curve construction
- When derivatives must be smooth AND monotonic

**Example**:
```python
import QuantLib as ql

x = ql.Array([1.0, 2.0, 3.0, 4.0, 5.0])
y = ql.Array([1.0, 1.5, 2.2, 3.0, 3.5])  # Monotonically increasing

interp = ql.FritschButlandCubic(x, y)
value = interp(2.5)  # Will not overshoot
slope = interp.derivative(2.5)  # Smooth derivative
```

#### FritschButlandLogCubic

**Description**: Fritsch-Butland in log-space.

**Constructor**:
```python
FritschButlandLogCubic(x, y)
```

**Properties**:
- ✅ Monotonic
- ✅ Positive values
- ✅ Smooth first derivative
- ✅ Excellent for yield curves

**Methods**: derivative, secondDerivative, primitive

**Use Cases**:
- **Recommended choice for discount curves**
- Premium interpolation for forward rates
- Volatility curves requiring smoothness and positivity

**Example**:
```python
import QuantLib as ql

# Yield curve example
maturities = ql.Array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
zero_rates = ql.Array([0.015, 0.018, 0.020, 0.022, 0.028, 0.032, 0.035])

interp = ql.FritschButlandLogCubic(maturities, zero_rates)

# Interpolate 7-year rate
rate_7y = interp(7.0)

# Calculate instantaneous forward rate
# f(t) = r(t) + t * r'(t)
r = interp(7.0)
dr = interp.derivative(7.0)
forward_7y = r + 7.0 * dr
```

---

### Parabolic Interpolation

Parabolic interpolation uses quadratic polynomials.

#### Parabolic

**Description**: Piecewise parabolic interpolation.

**Constructor**:
```python
Parabolic(x, y)
```

**Properties**:
- ✅ Smooth (C¹)
- ❌ Can oscillate
- ❌ Not monotonic

**Methods**: derivative, secondDerivative, primitive

**Use Cases**:
- Alternative to cubic splines
- When quadratic is more appropriate than cubic

#### LogParabolic

**Description**: Parabolic interpolation in log-space.

**Constructor**:
```python
LogParabolic(x, y)
```

**Properties**:
- ✅ Positive values
- ✅ Smooth in log-space

**Methods**: derivative, secondDerivative, primitive

#### MonotonicParabolic

**Description**: Monotonicity-preserving parabolic interpolation.

**Constructor**:
```python
MonotonicParabolic(x, y)
```

**Properties**:
- ✅ Monotonic
- ✅ Smooth (C¹)
- ✅ Simpler than cubic

**Methods**: derivative, secondDerivative, primitive

**Use Cases**:
- Simpler alternative to Fritsch-Butland
- When quadratic is sufficient

#### MonotonicLogParabolic

**Description**: Monotonic parabolic in log-space.

**Constructor**:
```python
MonotonicLogParabolic(x, y)
```

**Properties**:
- ✅ Monotonic
- ✅ Positive values
- ✅ Smooth

**Methods**: derivative, secondDerivative, primitive

---

### Specialized Methods

#### ConvexMonotoneInterpolation

**Description**: Hagan-West convex monotone method for yield curves. See [detailed documentation](convex-monotone-interpolation.md).

**Constructor**:
```python
ConvexMonotoneInterpolation(x, y, quadraticity=0.3, monotonicity=0.7, forcePositive=True)
```

**Parameters**:
- `quadraticity`: Blend between quadratic (1.0) and convex monotone (0.0)
- `monotonicity`: Monotonicity enforcement (0.0 to 1.0)
- `forcePositive`: Ensure non-negative values

**Properties**:
- ✅ Excellent for forward rate curves
- ✅ Configurable smoothness vs monotonicity
- ✅ Can enforce positivity
- ⚠️ C⁰ only (discontinuous derivatives)
- ❌ No derivative() or secondDerivative() methods
- ❌ primitive() available in C++ but not exposed to Python

**Use Cases**:
- Forward rate curve interpolation
- When you need strict control over monotonicity and positivity
- Yield curve bootstrapping (via ConvexMonotone trait)

**Example**:
```python
import QuantLib as ql

times = ql.Array([0.5, 1.0, 2.0, 5.0, 10.0])
forwards = ql.Array([0.02, 0.025, 0.028, 0.030, 0.032])

# Standard settings
interp = ql.ConvexMonotoneInterpolation(times, forwards)

# More smoothing
smooth_interp = ql.ConvexMonotoneInterpolation(
    times, forwards,
    quadraticity=0.5,    # More quadratic character
    monotonicity=0.5,    # Less strict monotonicity
    forcePositive=True
)

forward_3y = interp(3.0)
```

**Reference**: See [convex-monotone-interpolation.md](convex-monotone-interpolation.md) for full details.

#### LagrangeInterpolation

**Description**: Polynomial interpolation using Lagrange form.

**Constructor**:
```python
LagrangeInterpolation(x, y)
```

**Properties**:
- ✅ Exact at all data points
- ✅ Smooth (polynomial)
- ❌ Can oscillate wildly (Runge's phenomenon)
- ❌ Numerical instability with many points
- ⚠️ Use with caution for n > 10 points

**Use Cases**:
- Mathematical applications requiring polynomial interpolation
- Small datasets (< 10 points)
- When exact polynomial fit is required

**Warning**: Lagrange interpolation can produce extreme oscillations between data points, especially with equally-spaced points.

**Example**:
```python
import QuantLib as ql

# Use only with small datasets
x = ql.Array([1.0, 2.0, 3.0, 4.0])
y = ql.Array([1.0, 4.0, 9.0, 16.0])

interp = ql.LagrangeInterpolation(x, y)
value = interp(2.5)
```

#### ChebyshevInterpolation

**Description**: Polynomial interpolation using Chebyshev nodes or from function.

**Constructors**:
```python
# From data array
ChebyshevInterpolation(f_values, pointsType=ChebyshevInterpolation.SecondKind)

# From function (Python)
ChebyshevInterpolation(n, function, pointsType=ChebyshevInterpolation.SecondKind)
```

**Point Types**:
- `ChebyshevInterpolation.FirstKind`: Chebyshev points of first kind
- `ChebyshevInterpolation.SecondKind`: Chebyshev points of second kind (default)

**Properties**:
- ✅ Minimizes interpolation error (optimal polynomial approximation)
- ✅ Better numerical stability than Lagrange
- ✅ Can interpolate functions directly
- ⚠️ Requires Chebyshev nodes for optimal properties

**Static Methods**:
- `ChebyshevInterpolation.nodes(n, pointsType)`: Get Chebyshev nodes

**Use Cases**:
- Function approximation
- Numerical analysis
- When you need polynomial approximation with controlled error

**Example**:
```python
import QuantLib as ql
import math

# From function
def my_function(x):
    return math.exp(-x * x)

# Create interpolation from function
n = 10
interp = ql.ChebyshevInterpolation(n, my_function)

# Evaluate
value = interp(0.5)

# Get Chebyshev nodes
nodes = ql.ChebyshevInterpolation.nodes(10, ql.ChebyshevInterpolation.SecondKind)
```

#### RichardsonExtrapolation

**Description**: Numerical extrapolation technique to improve convergence.

**Constructor**:
```python
RichardsonExtrapolation(function, delta_h, n=None)
```

**Parameters**:
- `function`: Python callable f(h) to extrapolate
- `delta_h`: Step size reduction factor
- `n`: Scaling exponent (optional)

**Methods**:
- `extrapolation(t=2.0)`: Extrapolate with t scaling
- `extrapolation(t, s)`: Extrapolate with custom t and s

**Use Cases**:
- Improving numerical integration accuracy
- Accelerating convergence of sequences
- Richardson extrapolation for numerical derivatives

**Example**:
```python
import QuantLib as ql

# Function to extrapolate (e.g., numerical integration with step size h)
def trapezoid_rule(h):
    # Your numerical method that depends on step size h
    return some_approximation(h)

# Create Richardson extrapolation
richardson = ql.RichardsonExtrapolation(trapezoid_rule, delta_h=0.5)

# Get extrapolated value
improved_value = richardson(2.0)  # Richardson extrapolation
```

---

## 2D Interpolation Classes

### BilinearInterpolation

**Description**: Bilinear interpolation on a rectangular grid.

**Constructor**:
```python
BilinearInterpolation(x_array, y_array, z_matrix)
```

**Parameters**:
- `x_array`: Array of x-coordinates
- `y_array`: Array of y-coordinates
- `z_matrix`: Matrix of z-values (rows=y, cols=x)

**Properties**:
- ✅ Fast
- ✅ Continuous (C⁰)
- ❌ Discontinuous first derivatives
- ⚡ Very efficient

**Use Cases**:
- Volatility surface interpolation (simple)
- 2D lookup tables
- Fast 2D interpolation

**Example**:
```python
import QuantLib as ql

# Strike and expiry axes
strikes = ql.Array([90.0, 95.0, 100.0, 105.0, 110.0])
expiries = ql.Array([0.25, 0.5, 1.0, 2.0])

# Volatility matrix (4x5)
vols = ql.Matrix(4, 5)
vols[0][0] = 0.20  # expiry=0.25, strike=90
vols[0][1] = 0.18  # expiry=0.25, strike=95
# ... fill in all values ...

interp = ql.BilinearInterpolation(strikes, expiries, vols)

# Interpolate volatility at strike=102, expiry=0.75
vol = interp(102.0, 0.75)
```

### BicubicSpline

**Description**: Bicubic spline interpolation on a rectangular grid.

**Constructor**:
```python
BicubicSpline(x_array, y_array, z_matrix)
```

**Properties**:
- ✅ Smooth (C²)
- ✅ Continuous second derivatives
- ❌ Can oscillate
- 🐌 Slower than bilinear

**Use Cases**:
- Smooth volatility surfaces
- When smoothness is critical
- Surface fitting requiring smooth derivatives

**Example**:
```python
import QuantLib as ql

strikes = ql.Array([90.0, 95.0, 100.0, 105.0, 110.0])
expiries = ql.Array([0.25, 0.5, 1.0, 2.0])
vols = ql.Matrix(4, 5)  # Fill with data

interp = ql.BicubicSpline(strikes, expiries, vols)

# Smooth interpolation
vol = interp(102.0, 0.75)
```

---

## Extrapolation and Advanced Features

### Controlling Extrapolation

Most interpolation methods support extrapolation control:

```python
interp = ql.LinearInterpolation(x, y)

# By default, extrapolation is disabled
try:
    value = interp(10.0)  # If 10.0 > max(x), this throws an error
except:
    print("Extrapolation not allowed")

# Enable extrapolation
value = interp(10.0, allowExtrapolation=True)  # OK
```

### Common Patterns

#### Using Interpolation Traits with Curves

Interpolation traits are used with PiecewiseYieldCurve and InterpolatedDiscountCurve:

```python
import QuantLib as ql

# Define helpers (deposits, swaps, etc.)
helpers = [...]

# Use different interpolation methods via traits
calendar = ql.TARGET()
day_counter = ql.Actual360()

# Linear interpolation
curve_linear = ql.PiecewiseLogLinearDiscount(0, calendar, helpers, day_counter)

# Cubic interpolation
curve_cubic = ql.PiecewiseLogCubicDiscount(0, calendar, helpers, day_counter)

# Convex monotone (via trait)
# Note: Need to use specific curve constructors that support ConvexMonotone
```

Available interpolation traits (for use with curves):
- `Linear`
- `LogLinear`
- `BackwardFlat`
- `ForwardFlat`
- `Cubic` (with parameters)
- `ConvexMonotone` (with parameters)
- `MonotonicCubic`
- `SplineCubic`
- `SplineLogCubic`
- `Kruger`
- `KrugerLog`

---

## Choosing the Right Interpolation

### Decision Tree

```
Do you need 2D interpolation?
├─ Yes → BilinearInterpolation (fast) or BicubicSpline (smooth)
└─ No (1D) → Continue below

Is data strictly positive (e.g., discount factors, volatility)?
├─ Yes → Use Log* variant
└─ No → Use regular variant

Do you need monotonicity?
├─ Yes → Continue
│   ├─ Need smooth derivatives?
│   │   ├─ Yes → FritschButlandCubic / FritschButlandLogCubic ⭐
│   │   └─ No → MonotonicCubicNaturalSpline
│   └─ Special case (forward rates) → ConvexMonotoneInterpolation
└─ No → Continue
    ├─ Need smooth (C²) interpolation?
    │   ├─ Yes → CubicNaturalSpline / LogCubicNaturalSpline
    │   └─ Smoothness not critical → LinearInterpolation / LogLinearInterpolation
    └─ Simple step function → BackwardFlat or ForwardFlat
```

### Recommendations by Use Case

| Use Case | Recommended Method | Alternative |
|----------|-------------------|-------------|
| **Discount Curves** | FritschButlandLogCubic ⭐ | MonotonicLogCubicNaturalSpline |
| **Forward Rate Curves** | ConvexMonotoneInterpolation | FritschButlandCubic |
| **Zero Rate Curves** | MonotonicCubicNaturalSpline | FritschButlandCubic |
| **Volatility Curves** | LogLinearInterpolation | FritschButlandLogCubic |
| **Volatility Surfaces** | BilinearInterpolation | BicubicSpline |
| **Simple Positive Data** | LogLinearInterpolation | LinearInterpolation |
| **Exchange Conventions** | BackwardFlat or ForwardFlat | N/A |
| **Maximum Smoothness** | CubicNaturalSpline | KrugerCubic |
| **Function Approximation** | ChebyshevInterpolation | LagrangeInterpolation (small n) |

### Performance Considerations

**Fastest** (in order):
1. BackwardFlat / ForwardFlat
2. LinearInterpolation / LogLinearInterpolation
3. Parabolic variants
4. Cubic variants
5. ConvexMonotoneInterpolation (complex internal logic)

**Memory Usage** (minimal to high):
1. Flat interpolations
2. Linear interpolations
3. Cubic splines
4. ConvexMonotone (stores section helpers)

---

## Common Pitfalls and Tips

### 1. Input Data Requirements

**All interpolations require**:
- x-values must be **strictly increasing**
- x and y arrays must have the **same length**
- Minimum number of points varies (usually 2+)

```python
# ❌ WRONG - not strictly increasing
x = ql.Array([1.0, 2.0, 2.0, 3.0])  # Duplicate!
y = ql.Array([1.0, 2.0, 3.0, 4.0])

# ✅ CORRECT
x = ql.Array([1.0, 2.0, 2.5, 3.0])
y = ql.Array([1.0, 2.0, 3.0, 4.0])
```

### 2. Logarithmic Interpolations Require Positive Data

```python
# ❌ WRONG - negative or zero values
x = ql.Array([1.0, 2.0, 3.0])
y = ql.Array([0.0, -1.0, 2.0])  # Has zero and negative!
interp = ql.LogLinearInterpolation(x, y)  # Will fail or give NaN

# ✅ CORRECT - all positive
y = ql.Array([0.1, 1.0, 2.0])
interp = ql.LogLinearInterpolation(x, y)
```

### 3. Extrapolation Default is Disabled

```python
x = ql.Array([1.0, 2.0, 3.0])
y = ql.Array([1.0, 2.0, 3.0])
interp = ql.LinearInterpolation(x, y)

# ❌ Will throw error (default: no extrapolation)
value = interp(5.0)

# ✅ Explicitly allow extrapolation
value = interp(5.0, allowExtrapolation=True)
```

### 4. Choose Monotonic Methods for Monotonic Data

```python
# Data is monotonically increasing
x = ql.Array([1.0, 2.0, 3.0, 4.0])
y = ql.Array([1.0, 2.0, 2.8, 3.5])

# ❌ BAD - CubicNaturalSpline might overshoot/oscillate
interp = ql.CubicNaturalSpline(x, y)

# ✅ GOOD - Guarantees monotonicity
interp = ql.MonotonicCubicNaturalSpline(x, y)
# or
interp = ql.FritschButlandCubic(x, y)
```

### 5. Derivative and Primitive Availability

Not all interpolations support derivative/primitive:

```python
# ✅ Has derivative, secondDerivative, primitive
interp = ql.CubicNaturalSpline(x, y)
deriv = interp.derivative(2.0)

# ❌ No derivative method
interp = ql.LinearInterpolation(x, y)
deriv = interp.derivative(2.0)  # AttributeError!

# ❌ ConvexMonotone has no derivative in Python
interp = ql.ConvexMonotoneInterpolation(x, y)
deriv = interp.derivative(2.0)  # AttributeError!
```

Check the [Method Availability Matrix](#method-availability-matrix) above.

---

## Complete Example: Yield Curve Interpolation Comparison

```python
import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np

# Market data: time vs zero rate
times = ql.Array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
rates = ql.Array([0.015, 0.017, 0.020, 0.023, 0.028, 0.031, 0.033])

# Create different interpolations
linear = ql.LinearInterpolation(times, rates)
log_linear = ql.LogLinearInterpolation(times, rates)
cubic = ql.CubicNaturalSpline(times, rates)
monotonic_cubic = ql.MonotonicCubicNaturalSpline(times, rates)
fritsch = ql.FritschButlandCubic(times, rates)
convex_mono = ql.ConvexMonotoneInterpolation(times, rates)

# Evaluation points
eval_times = np.linspace(0.25, 30.0, 200)

# Evaluate all methods
methods = {
    'Linear': linear,
    'LogLinear': log_linear,
    'Cubic': cubic,
    'MonotonicCubic': monotonic_cubic,
    'FritschButland': fritsch,
    'ConvexMonotone': convex_mono
}

results = {}
for name, interp in methods.items():
    results[name] = [interp(float(t), True) for t in eval_times]

# Plot
plt.figure(figsize=(12, 6))
for name, values in results.items():
    plt.plot(eval_times, values, label=name)

# Plot original points
plt.scatter(list(times), list(rates), color='red', s=50,
            zorder=5, label='Market Data')

plt.xlabel('Time (years)')
plt.ylabel('Zero Rate')
plt.title('Yield Curve Interpolation Methods Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Version Information

Based on QuantLib development version and QuantLib-SWIG bindings as of January 2025.

## See Also

- [ConvexMonotone Interpolation Deep Dive](convex-monotone-interpolation.md)
- [QuantLib Mathematical Tools](05-math.md)
- [Interest Rates & Curves](02-interest-rates.md)
