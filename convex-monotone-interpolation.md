# ConvexMonotoneInterpolation in QuantLib Python

## Overview

ConvexMonotoneInterpolation is a specialized interpolation method designed for yield curve construction. It was developed by Hagan & West and published in "Interpolation Methods for Curve Construction" (AMF Vol 13, No2 2006).

### Key Features

- **Monotonicity preservation**: Ensures smooth, monotonic interpolation suitable for financial data
- **Positive value enforcement**: Optional constraint to prevent negative forward rates
- **Smooth value interpolation**: Produces continuous forward rate curves
- **Configurable behavior**: Adjustable `quadraticity` and `monotonicity` parameters for curve smoothness

### Mathematical Properties

Unlike cubic splines which prioritize derivative continuity, ConvexMonotoneInterpolation:
- Produces **piecewise smooth functions** with potential derivative discontinuities at knot points
- Focuses on **value smoothness** rather than derivative smoothness
- Uses multiple helper functions internally (ConvexMonotone2Helper, ConvexMonotone3Helper, ConvexMonotone4Helper, QuadraticHelper) depending on the local gradient conditions
- Can compute **analytical integrals** (primitives) for each section

## Python API (Current Implementation)

### Constructor

```python
ConvexMonotoneInterpolation(
    x_values,           # Array: knot points (e.g., time values)
    y_values,           # Array: data values (e.g., forward rates)
    quadraticity=0.3,   # Real: blend between quadratic and convex monotone (0.0-1.0)
    monotonicity=0.7,   # Real: monotonicity enforcement level (0.0-1.0)
    forcePositive=True  # bool: enforce positive values (prevents negative rates)
)
```

#### Parameters Explained

- **`quadraticity`** (default: 0.3)
  - Range: 0.0 to 1.0
  - 0.0 = pure convex monotone method
  - 1.0 = pure quadratic interpolation
  - Values between blend the two approaches for smoother curves

- **`monotonicity`** (default: 0.7)
  - Range: 0.0 to 1.0
  - Controls how strictly monotonicity is enforced
  - Setting both `monotonicity=1.0` and `quadraticity=0.0` reproduces the basic Hagan/West method
  - Lower values produce smoother gradients, reducing P&L volatility for some curves

- **`forcePositive`** (default: True)
  - When True, ensures interpolated values never go negative
  - Essential for forward rate curves where negative rates are not desired
  - Adds special handling to avoid negative values in sections

### Methods

```python
# Create interpolation
x = ql.Array([0.0, 1.0, 2.0, 5.0, 10.0])
y = ql.Array([0.02, 0.025, 0.03, 0.035, 0.04])
interp = ql.ConvexMonotoneInterpolation(x, y)

# Evaluate at a point
value = interp(2.5)                    # Interpolate at x=2.5
value = interp(2.5, True)              # Allow extrapolation beyond range
```

## Missing Methods (Not Currently Exposed)

### 1. `primitive()` Method

**Purpose**: Computes the integral (antiderivative) of the interpolated function.

**C++ Signature**:
```cpp
Real primitive(Real x, bool extrapolate = false) const
```

**Use Cases**:
- **Discount factor calculations**: Integrating forward rates to compute discount factors
  ```
  DF(t) = exp(-∫₀ᵗ f(s) ds)
  ```
- **Average rate calculations**: Computing time-weighted average rates over periods
- **Bond pricing**: Integrating cash flows with interpolated discount curves
- **Present value calculations**: Any scenario requiring integration of the curve

**Implementation Details**:
- Each section helper (ConvexMonotone2Helper, ConvexMonotone3Helper, etc.) has analytical primitive formulas
- The implementation handles piecewise integration across different section types
- For example, ConvexMonotone2Helper uses quadratic primitives with conditional logic based on eta values
- Primitives accumulate from section to section, maintaining continuity

**Why It's Valuable**:
The primitive is often more important than the interpolated values themselves in fixed income calculations. For yield curves, you frequently need integrals rather than point values.

### 2. `getExistingHelpers()` Method

**Purpose**: Retrieves internal section helpers for incremental curve construction.

**C++ Signature**:
```cpp
std::map<Real, ext::shared_ptr<detail::SectionHelper>> getExistingHelpers()
```

**Use Cases**:
- **Bootstrapping yield curves**: Building curves point-by-point as new market data arrives
- **Incremental updates**: Adding new instruments without recalculating the entire curve
- **Performance optimization**: Reusing calculations from previous interpolations
- **Piecewise curve construction**: Used by PiecewiseYieldCurve with ConvexMonotone interpolation

**How It Works**:
- When constructing a ConvexMonotoneInterpolation, you can pass `preExistingHelpers`
- The interpolation reuses these helpers for already-calculated sections
- Only new sections need to be computed
- This is critical for the `localInterpolate()` method used in bootstrapping

**Implementation Note**:
The constructor accepts a `helper_map` parameter (see convexmonotoneinterpolation.hpp:62-63) which allows passing in pre-computed helpers.

## Methods That Will Never Be Available

### `derivative()` and `secondDerivative()`

**Status**: Intentionally not implemented in C++

**C++ Implementation** (convexmonotoneinterpolation.hpp:216-222):
```cpp
Real derivative(Real) const override {
    QL_FAIL("Convex-monotone spline derivative not implemented");
}
Real secondDerivative(Real) const override {
    QL_FAIL("Convex-monotone spline second derivative not implemented");
}
```

**Why They're Not Available**:

1. **Mathematical Nature**: ConvexMonotoneInterpolation uses piecewise functions with different formulas in different sections. At transition points (knot points), the derivative can be discontinuous - it has different left and right limits.

2. **Multiple Section Types**: The implementation dynamically chooses between:
   - EverywhereConstantHelper (constant values)
   - ConstantGradHelper (linear sections)
   - ConvexMonotone2Helper, ConvexMonotone3Helper, ConvexMonotone4Helper (various convex monotone formulas)
   - QuadraticHelper (quadratic sections)
   - ComboHelper (blends of the above)

   Each transition between different helper types can create a kink in the derivative.

3. **Design Philosophy**: The method prioritizes:
   - **Smooth interpolated values** (forward rates themselves)
   - **Monotonicity** in the values
   - **Absence of oscillations** that plague cubic splines

   NOT:
   - Smooth derivatives
   - Continuous rate of change

4. **Practical Considerations**: In yield curve work:
   - You care about forward rates (the values)
   - You care about integrals of forward rates (discount factors via primitives)
   - You rarely care about derivatives of forward rates (rate of change of forward rates)

**Contrast with Cubic Splines**:
- Cubic splines are specifically designed to have continuous second derivatives
- This makes them smooth but can cause oscillations and negative values
- The SWIG bindings expose `derivative()` and `secondDerivative()` for cubic splines (see interpolation.i:85-94)
- ConvexMonotone trades derivative continuity for better value behavior

## Typical Use Cases

### Yield Curve Construction

```python
import QuantLib as ql

# Time points (in years) and forward rates
times = ql.Array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
forwards = ql.Array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])

# Create interpolation with default settings
# (good balance of smoothness and monotonicity)
interp = ql.ConvexMonotoneInterpolation(times, forwards)

# Interpolate forward rate at 3 years
forward_3y = interp(3.0)
```

### With Custom Parameters

```python
# More aggressive smoothing (reduce P&L volatility)
smooth_interp = ql.ConvexMonotoneInterpolation(
    times,
    forwards,
    quadraticity=0.5,   # More quadratic character
    monotonicity=0.5,   # Less strict monotonicity
    forcePositive=True
)

# Strict Hagan-West method
strict_interp = ql.ConvexMonotoneInterpolation(
    times,
    forwards,
    quadraticity=0.0,   # Pure convex monotone
    monotonicity=1.0,   # Strict monotonicity
    forcePositive=True
)
```

### Usage with Yield Curve Traits

The ConvexMonotone trait can be used with PiecewiseYieldCurve:

```python
# Define the interpolation trait
interp_trait = ql.ConvexMonotone(
    quadraticity=0.3,
    monotonicity=0.7,
    forcePositive=True
)

# Use in piecewise curve construction
# Note: This uses the C++ localInterpolate method internally
# which leverages getExistingHelpers for efficient bootstrapping
```

## Comparison with Other Interpolation Methods

| Method | Derivative Continuity | Monotonic | Positive Values | Best For |
|--------|----------------------|-----------|----------------|----------|
| Linear | C⁰ (discontinuous 1st derivative) | Yes | Conditional | Simple, guaranteed stability |
| Cubic Spline | C² (smooth 2nd derivative) | No | No | Smooth derivatives needed |
| Fritsch-Butland | C¹ (smooth 1st derivative) | Yes | No | Monotonic data, smooth derivatives |
| ConvexMonotone | C⁰ (value continuity only) | Yes | Optional | Yield curves, forward rates |
| BackwardFlat/ForwardFlat | Discontinuous | Yes | Yes | Exchange conventions, simple |

### When to Use ConvexMonotone

**Best for**:
- Forward rate curves where monotonicity is important
- Cases where you need to enforce positive values
- Situations where derivative smoothness is not critical
- Yield curve bootstrapping with ConvexMonotone trait

**Avoid when**:
- You need smooth derivatives
- You need to compute derivative() or secondDerivative()
- Simple linear or flat interpolation would suffice
- You have very few data points (requires at least 2 points)

## Exposing the Missing Methods

### To Add `primitive()` Support

In `SWIG/interpolation.i`, after the SafeConvexMonotoneInterpolation class definition (line 396), add:

```cpp
%extend SafeConvexMonotoneInterpolation {
    Real primitive(Real x, bool extrapolate = false) {
        return self->f_.primitive(x, extrapolate);
    }
}
```

This would enable:
```python
integral_value = interp.primitive(5.0)  # Integral from 0 to 5.0
```

### To Add `getExistingHelpers()` Support

This is more complex as it requires:
1. Wrapping the `SectionHelper` class hierarchy
2. Wrapping `std::map<Real, ext::shared_ptr<detail::SectionHelper>>`
3. Modifying the constructor to accept existing helpers

Given the complexity and the fact that bootstrapping is typically done through the PiecewiseYieldCurve interface (which handles this internally), exposing this method has limited value for Python users.

## References

- **Paper**: Hagan, P.S., & West, G. (2006). "Interpolation Methods for Curve Construction." *Applied Mathematical Finance*, 13(2), 89-129.
- **C++ Header**: `QuantLib/ql/math/interpolations/convexmonotoneinterpolation.hpp`
- **SWIG Interface**: `QuantLib-SWIG/SWIG/interpolation.i` (lines 274-278, 331-396)
- **Related Classes**: Used internally by `PiecewiseYieldCurve` when using `ConvexMonotone` trait

## Version Information

Based on QuantLib development version and QuantLib-SWIG bindings as of January 2025.
