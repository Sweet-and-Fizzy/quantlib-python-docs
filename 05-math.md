# Mathematical Tools and Numerical Methods

This document covers QuantLib's mathematical infrastructure including linear algebra, interpolation, optimization, and statistical classes.

## Linear Algebra Classes

### Array Class

One-dimensional array for mathematical operations.

```python
# Constructors
Array()                    # Empty array
Array(size)               # Array of given size (initialized to 0)
Array(size, value)        # Array filled with value
Array([1, 2, 3, 4])      # From Python list

# Examples
arr = ql.Array(5)         # [0, 0, 0, 0, 0]
arr = ql.Array(3, 2.5)    # [2.5, 2.5, 2.5]
arr = ql.Array([1, 2, 3]) # [1, 2, 3]
```

#### Array Operations

```python
arr = ql.Array([1.0, 2.0, 3.0, 4.0])

# Element access
arr[0] = 5.0              # Set element
value = arr[1]            # Get element
arr.size()                # Number of elements
len(arr)                  # Python length

# Array properties  
arr.empty()               # True if empty
arr.front()               # First element
arr.back()                # Last element

# Mathematical operations
arr2 = ql.Array([2.0, 3.0, 4.0, 5.0])

# Element-wise operations
sum_arr = arr + arr2      # [7, 5, 7, 9]
diff_arr = arr - arr2     # [3, -1, -1, -1]
prod_arr = arr * arr2     # [10, 6, 12, 20]

# Scalar operations
scaled = arr * 2.0        # [10, 4, 6, 8]
shifted = arr + 1.0       # [6, 3, 4, 5]

# In-place operations
arr += arr2
arr -= 1.0
arr *= 2.0
```

#### Array Functions

```python
# Statistical functions
arr = ql.Array([1.0, 2.0, 3.0, 4.0, 5.0])

mean = ql.mean(arr)              # 3.0
variance = ql.variance(arr)      # Sample variance
std_dev = ql.standardDeviation(arr)  # Standard deviation

# Norms
l1_norm = ql.norm(arr, 1)        # L1 norm (sum of absolute values)  
l2_norm = ql.norm(arr, 2)        # L2 norm (Euclidean)
inf_norm = ql.norm(arr, float('inf'))  # Maximum absolute value

# Dot product
arr2 = ql.Array([1, 1, 1, 1, 1])
dot_product = ql.DotProduct(arr, arr2)  # 15.0

# Sorting and extrema
sorted_arr = ql.sort(arr)        # Sort in ascending order
min_val = ql.min(arr)            # Minimum value
max_val = ql.max(arr)            # Maximum value
```

### Matrix Class

Two-dimensional matrix for linear algebra.

```python
# Constructors
Matrix()                         # Empty matrix
Matrix(rows, cols)              # Matrix of given size (zeros)
Matrix(rows, cols, value)       # Matrix filled with value

# Examples
mat = ql.Matrix(3, 3)           # 3x3 zero matrix
mat = ql.Matrix(2, 3, 1.0)      # 2x3 matrix filled with 1.0
```

#### Matrix Operations

```python
mat = ql.Matrix(3, 3)

# Element access
mat[0][1] = 2.5              # Set element at row 0, col 1
value = mat[1][2]            # Get element
mat.rows()                   # Number of rows
mat.columns()                # Number of columns

# Matrix properties
mat.empty()                  # True if empty
mat.diagonal()               # Extract diagonal as Array

# Create special matrices
identity = ql.IdentityMatrix(3)      # 3x3 identity matrix
zeros = ql.ZeroMatrix(2, 4)          # 2x4 zero matrix

# Matrix arithmetic
mat2 = ql.Matrix(3, 3, 2.0)
sum_mat = mat + mat2         # Matrix addition
diff_mat = mat - mat2        # Matrix subtraction
prod_mat = mat * mat2        # Matrix multiplication

# Scalar operations
scaled = mat * 3.0           # Scale matrix
```

#### Matrix Decompositions

```python
# SVD (Singular Value Decomposition)
mat = ql.Matrix(3, 3)
# Fill matrix with data...

svd = ql.SVD(mat)
u = svd.U()                  # Left singular vectors
s = svd.s()                  # Singular values (Array)
v = svd.V()                  # Right singular vectors

# Solve linear system Ax = b using SVD
b = ql.Array([1, 2, 3])
x = svd.solveFor(b)          # Solution vector

# Pseudo-inverse
pseudo_inv = ql.pseudoInverse(mat, tol=1e-12)
```

#### Linear Algebra Functions

```python
# Matrix-vector operations
mat = ql.Matrix(3, 2)
vec = ql.Array([1, 2])
result = mat * vec           # Matrix-vector multiplication

# Transpose
transposed = ql.transpose(mat)

# Determinant (for square matrices)
square_mat = ql.Matrix(3, 3)
det = ql.determinant(square_mat)

# Matrix norms
frobenius = ql.normFrobenius(mat)    # Frobenius norm
```

## Interpolation Classes

QuantLib provides various interpolation methods for curve construction and data fitting. See [interpolation-reference.md](interpolation-reference.md) for detailed guidance on choosing interpolation methods.

### Common Interface

All 1D interpolation classes share a common interface:

```python
# Constructor pattern
interp = ql.InterpolationClass(x_array, y_array, [additional_params])

# Evaluation
value = interp(x_point)
value = interp(x_point, allowExtrapolation=True)

# Some interpolations also provide (where applicable):
# derivative = interp.derivative(x_point, allowExtrapolation=False)
# second_deriv = interp.secondDerivative(x_point, allowExtrapolation=False)
# integral = interp.primitive(x_point, allowExtrapolation=False)
```

### Linear and Flat Interpolations

#### LinearInterpolation

Linear interpolation between adjacent points.

```python
LinearInterpolation(x, y)

# Example
x = ql.Array([1.0, 2.0, 3.0, 4.0, 5.0])
y = ql.Array([1.0, 1.5, 1.8, 2.0, 2.1])
interp = ql.LinearInterpolation(x, y)
value = interp(2.5)
```

**Methods**: `operator()`

#### LogLinearInterpolation

Linear interpolation in log-space. Ensures positive values.

```python
LogLinearInterpolation(x, y)

# Example
times = ql.Array([0.5, 1.0, 2.0, 5.0, 10.0])
discount_factors = ql.Array([0.99, 0.98, 0.95, 0.88, 0.75])
interp = ql.LogLinearInterpolation(times, discount_factors)
df_3y = interp(3.0)
```

**Methods**: `operator()`

#### BackwardFlatInterpolation

Step function using previous point's value.

```python
BackwardFlatInterpolation(x, y)

# Example
dates = ql.Array([1.0, 2.0, 3.0, 5.0])
rates = ql.Array([0.02, 0.025, 0.03, 0.035])
interp = ql.BackwardFlatInterpolation(dates, rates)
rate = interp(2.5)  # Returns 0.025
```

**Methods**: `operator()`

#### ForwardFlatInterpolation

Step function using next point's value.

```python
ForwardFlatInterpolation(x, y)

# Example
dates = ql.Array([1.0, 2.0, 3.0, 5.0])
forwards = ql.Array([0.02, 0.025, 0.03, 0.035])
interp = ql.ForwardFlatInterpolation(dates, forwards)
fwd = interp(2.5)  # Returns 0.03
```

**Methods**: `operator()`

### Cubic Spline Interpolations

#### CubicNaturalSpline

Natural cubic spline with zero second derivatives at boundaries.

```python
CubicNaturalSpline(x, y)

# Example
x = ql.Array([0.0, 1.0, 2.0, 3.0, 4.0])
y = ql.Array([1.0, 2.5, 3.0, 3.2, 3.5])
spline = ql.CubicNaturalSpline(x, y)
value = spline(1.5)
slope = spline.derivative(1.5)
curvature = spline.secondDerivative(1.5)
area = spline.primitive(2.0)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

#### LogCubicNaturalSpline

Natural cubic spline in log-space. Ensures positive values.

```python
LogCubicNaturalSpline(x, y)

# Example
times = ql.Array([0.5, 1.0, 2.0, 5.0, 10.0])
vols = ql.Array([0.15, 0.18, 0.20, 0.22, 0.21])
interp = ql.LogCubicNaturalSpline(times, vols)
vol_3y = interp(3.0)
vol_slope = interp.derivative(3.0)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

#### MonotonicCubicNaturalSpline

Cubic spline with monotonicity preservation (Hyman filter).

```python
MonotonicCubicNaturalSpline(x, y)

# Example
times = ql.Array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
rates = ql.Array([0.01, 0.015, 0.02, 0.025, 0.03, 0.032])
interp = ql.MonotonicCubicNaturalSpline(times, rates)
rate = interp(3.0)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

#### MonotonicLogCubicNaturalSpline

Monotonic cubic spline in log-space.

```python
MonotonicLogCubicNaturalSpline(x, y)

# Example
times = ql.Array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
discount_factors = ql.Array([0.998, 0.995, 0.98, 0.95, 0.87, 0.73])
interp = ql.MonotonicLogCubicNaturalSpline(times, discount_factors)
df = interp(3.0)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

### Kruger Cubic Interpolations

Kruger's local cubic interpolation method.

#### KrugerCubic

```python
KrugerCubic(x, y)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

#### KrugerLogCubic

Kruger interpolation in log-space.

```python
KrugerLogCubic(x, y)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

### Fritsch-Butland Cubic Interpolations

Monotonicity-preserving cubic interpolation (Fritsch & Butland, 1984).

#### FritschButlandCubic

```python
FritschButlandCubic(x, y)

# Example - recommended for monotonic curves
x = ql.Array([1.0, 2.0, 3.0, 4.0, 5.0])
y = ql.Array([1.0, 1.5, 2.2, 3.0, 3.5])
interp = ql.FritschButlandCubic(x, y)
value = interp(2.5)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

#### FritschButlandLogCubic

Fritsch-Butland in log-space. Recommended for discount curves.

```python
FritschButlandLogCubic(x, y)

# Example - excellent for yield curves
maturities = ql.Array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
zero_rates = ql.Array([0.015, 0.018, 0.020, 0.022, 0.028, 0.032, 0.035])
interp = ql.FritschButlandLogCubic(maturities, zero_rates)
rate_7y = interp(7.0)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

### Parabolic Interpolations

Piecewise parabolic (quadratic) interpolation.

#### Parabolic

```python
Parabolic(x, y)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

#### LogParabolic

```python
LogParabolic(x, y)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

#### MonotonicParabolic

```python
MonotonicParabolic(x, y)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

#### MonotonicLogParabolic

```python
MonotonicLogParabolic(x, y)
```

**Methods**: `operator()`, `derivative()`, `secondDerivative()`, `primitive()`

### Specialized Interpolations

#### ConvexMonotoneInterpolation

Hagan-West convex monotone method for yield curves. See [convex-monotone-interpolation.md](convex-monotone-interpolation.md) for details.

```python
ConvexMonotoneInterpolation(x, y, quadraticity=0.3, monotonicity=0.7, forcePositive=True)

# Example
times = ql.Array([0.5, 1.0, 2.0, 5.0, 10.0])
forwards = ql.Array([0.02, 0.025, 0.028, 0.030, 0.032])
interp = ql.ConvexMonotoneInterpolation(times, forwards)
forward_3y = interp(3.0)

# With custom parameters for more smoothing
smooth_interp = ql.ConvexMonotoneInterpolation(
    times, forwards,
    quadraticity=0.5,
    monotonicity=0.5,
    forcePositive=True
)
```

**Methods**: `operator()` only (no derivative or primitive in Python bindings)

#### LagrangeInterpolation

Polynomial interpolation using Lagrange form. Use with caution for large datasets (Runge's phenomenon).

```python
LagrangeInterpolation(x, y)

# Example - only for small datasets
x = ql.Array([1.0, 2.0, 3.0, 4.0])
y = ql.Array([1.0, 4.0, 9.0, 16.0])
interp = ql.LagrangeInterpolation(x, y)
value = interp(2.5)
```

**Methods**: `operator()` only

#### ChebyshevInterpolation

Polynomial interpolation using Chebyshev nodes or from function.

```python
# From data array
ChebyshevInterpolation(f_values, pointsType=ChebyshevInterpolation.SecondKind)

# From function
ChebyshevInterpolation(n, function, pointsType=ChebyshevInterpolation.SecondKind)

# Example
import math
def my_function(x):
    return math.exp(-x * x)

n = 10
interp = ql.ChebyshevInterpolation(n, my_function)
value = interp(0.5)

# Get Chebyshev nodes
nodes = ql.ChebyshevInterpolation.nodes(10, ql.ChebyshevInterpolation.SecondKind)
```

**Point Types**: `ChebyshevInterpolation.FirstKind`, `ChebyshevInterpolation.SecondKind`

**Methods**: `operator()`, static `nodes()`

#### RichardsonExtrapolation

Numerical extrapolation technique to improve convergence.

```python
RichardsonExtrapolation(function, delta_h, n=None)

# Example
def trapezoid_rule(h):
    # Your numerical method that depends on step size h
    return some_approximation(h)

richardson = ql.RichardsonExtrapolation(trapezoid_rule, delta_h=0.5)
improved_value = richardson(2.0)
```

**Methods**: `operator()(t)`, `operator()(t, s)`

### 2D Interpolations

#### BilinearInterpolation

Bilinear interpolation on rectangular grid.

```python
BilinearInterpolation(x_array, y_array, z_matrix)

# Example
strikes = ql.Array([90.0, 95.0, 100.0, 105.0, 110.0])
expiries = ql.Array([0.25, 0.5, 1.0, 2.0])
vols = ql.Matrix(4, 5)  # 4x5 matrix (rows=expiries, cols=strikes)
# Fill vols with data...

interp = ql.BilinearInterpolation(strikes, expiries, vols)
vol = interp(102.0, 0.75)  # Interpolate at strike=102, expiry=0.75
```

**Methods**: `operator()(x, y, allowExtrapolation=False)`

#### BicubicSpline

Bicubic spline interpolation on rectangular grid.

```python
BicubicSpline(x_array, y_array, z_matrix)

# Example
strikes = ql.Array([90.0, 95.0, 100.0, 105.0, 110.0])
expiries = ql.Array([0.25, 0.5, 1.0, 2.0])
vols = ql.Matrix(4, 5)
# Fill vols with data...

interp = ql.BicubicSpline(strikes, expiries, vols)
vol = interp(102.0, 0.75)
```

**Methods**: `operator()(x, y, allowExtrapolation=False)`

## Optimization and Root Finding

### 1D Root Finding

Find zeros of univariate functions.

#### Root-Finding Algorithms

```python
# Brent's method (recommended for most cases)
brent = ql.Brent()
brent.setMaxEvaluations(100)
brent.setLowerBound(-10.0)
brent.setUpperBound(10.0)

# Define target function (example: f(x) = x^2 - 4)
def target_function(x):
    return x * x - 4.0

# Find root
root = brent.solve(target_function, 1e-8, 1.0, -3.0, 3.0)  # Should find x = 2

# Other algorithms
bisection = ql.Bisection()
false_position = ql.FalsePosition()
newton = ql.Newton()               # Requires derivative
newton_safe = ql.NewtonSafe()      # Newton with bisection backup
ridder = ql.Ridder()
secant = ql.Secant()
```

#### Usage Pattern

```python
solver = ql.Brent()
solver.setMaxEvaluations(1000)

try:
    root = solver.solve(
        f=target_function,      # Function to solve
        accuracy=1e-8,          # Required accuracy
        guess=0.0,              # Initial guess
        step=1.0                # Initial step size
    )
    print(f"Root found: {root:.6f}")
    print(f"Function evaluations: {solver.evaluationNumber()}")
except RuntimeError as e:
    print(f"Root finding failed: {e}")
```

### Multi-Dimensional Optimization

Minimize functions of several variables.

#### Optimization Algorithms

```python
# Levenberg-Marquardt (for least squares)
lm = ql.LevenbergMarquardt(
    epsfcn=1e-8,           # Function tolerance
    xtol=1e-8,             # Parameter tolerance  
    gtol=1e-8              # Gradient tolerance
)

# Simplex method (Nelder-Mead)
simplex = ql.Simplex(lambda_initial=0.1)

# Conjugate gradient
conj_grad = ql.ConjugateGradient()

# Steepest descent
steepest = ql.SteepestDescent()

# Differential Evolution
diff_evolution = ql.DifferentialEvolution()
```

#### Cost Functions

Define objective function to minimize.

```python
class QuadraticCostFunction(ql.CostFunction):
    """Example: minimize (x-2)^2 + (y-3)^2"""
    
    def value(self, x):
        # x is ql.Array with parameters
        return (x[0] - 2.0)**2 + (x[1] - 3.0)**2
    
    def values(self, x):
        # Return residuals for least-squares
        residuals = ql.Array(2)
        residuals[0] = x[0] - 2.0
        residuals[1] = x[1] - 3.0
        return residuals

# Optimization
cost_function = QuadraticCostFunction()
constraint = ql.NoConstraint()  # Unconstrained
initial_guess = ql.Array([0.0, 0.0])

# End criteria
end_criteria = ql.EndCriteria(
    maxIterations=1000,
    maxStationaryStateIterations=100,
    rootEpsilon=1e-8,
    functionEpsilon=1e-8,
    gradientNormEpsilon=1e-8
)

# Optimize
problem = ql.Problem(cost_function, constraint, initial_guess)
result = lm.minimize(problem, end_criteria)

optimized_params = problem.currentValue()
minimum_value = problem.functionValue()
```

#### Constraints

```python
# Box constraints (parameter bounds)
lower_bounds = ql.Array([-5.0, -5.0])
upper_bounds = ql.Array([5.0, 5.0])
box_constraint = ql.BoundaryConstraint(lower_bounds, upper_bounds)

# Positive constraint
positive_constraint = ql.PositiveConstraint()

# Non-negative constraint  
nonnegative_constraint = ql.NoConstraint()  # For illustration

# Composite constraint
composite = ql.CompositeConstraint(box_constraint, positive_constraint)
```

### End Criteria

Control when optimization terminates.

```python
end_criteria = ql.EndCriteria(
    maxIterations=1000,                    # Maximum iterations
    maxStationaryStateIterations=100,     # Max iterations without improvement
    rootEpsilon=1e-8,                     # Root tolerance
    functionEpsilon=1e-8,                 # Function value tolerance
    gradientNormEpsilon=1e-8              # Gradient norm tolerance
)

# Check termination reason
result = optimizer.minimize(problem, end_criteria)
if result == ql.EndCriteria.MaxIterations:
    print("Terminated due to maximum iterations")
elif result == ql.EndCriteria.StationaryFunctionValue:
    print("Terminated due to stationary function value")
```

## Statistics and Random Numbers

### Statistics Accumulators

Compute running statistics on data streams.

#### GeneralStatistics

```python
stats = ql.GeneralStatistics()

# Add data points
data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
for value in data:
    stats.add(value, weight=1.0)  # Equal weights

# Or add weighted data
stats.add(10.0, weight=2.0)  # This point counts twice

# Statistical measures
n = stats.samples()              # Number of data points
mean = stats.mean()              # Sample mean
variance = stats.variance()      # Sample variance
std_dev = stats.standardDeviation()  # Standard deviation
skewness = stats.skewness()      # Skewness
kurtosis = stats.kurtosis()      # Kurtosis
min_val = stats.min()           # Minimum value
max_val = stats.max()           # Maximum value

# Weighted statistics
weighted_mean = stats.weightSum()  # Sum of weights
```

#### IncrementalStatistics

More efficient for online computation.

```python
inc_stats = ql.IncrementalStatistics()

for value in data:
    inc_stats.add(value)

# Same interface as GeneralStatistics
mean = inc_stats.mean()
variance = inc_stats.variance()
```

#### RiskStatistics

Specialized for risk metrics.

```python
risk_stats = ql.RiskStatistics()

# Add returns data
returns = [-0.02, 0.01, -0.005, 0.015, -0.01, 0.008]
for ret in returns:
    risk_stats.add(ret)

# Risk metrics
var_95 = risk_stats.gaussianPercentile(0.05)     # 5% VaR (Gaussian)
var_99 = risk_stats.gaussianPercentile(0.01)     # 1% VaR
expected_shortfall = risk_stats.gaussianExpectedShortfall(0.05)  # Expected shortfall
downside_variance = risk_stats.downsideVariance()  # Downside variance
semi_deviation = risk_stats.semiDeviation()       # Semi-deviation
```

### Probability Distributions

#### Normal Distribution

```python
# Standard normal
standard_normal = ql.NormalDistribution()
pdf_value = standard_normal(0.0)        # Probability density at 0
cdf_value = ql.CumulativeNormalDistribution()(0.0)  # P(X <= 0) = 0.5

# Normal with mean and std dev
normal = ql.NormalDistribution(mean=2.0, sigma=1.5)
pdf_at_2 = normal(2.0)                  # Maximum density

# Cumulative normal
cum_normal = ql.CumulativeNormalDistribution()
prob = cum_normal(1.96)                 # P(Z <= 1.96) ≈ 0.975

# Inverse normal
inv_normal = ql.InverseCumulativeNormal()
quantile = inv_normal(0.975)            # 97.5% quantile ≈ 1.96

# Bivariate normal
bivariate = ql.BivariateCumulativeNormalDistribution(rho=0.5)
prob_2d = bivariate(1.0, 1.0)          # P(X <= 1, Y <= 1)
```

#### Other Distributions

```python
# Chi-squared distribution
chi_squared = ql.NonCentralChiSquaredDistribution(df=5, lambda_=2.0)

# Poisson distribution  
poisson = ql.PoissonDistribution(mean=3.0)

# Gamma distribution
gamma = ql.GammaDistribution(alpha=2.0, beta=1.0)
```

### Random Number Generators

#### Pseudo-Random Generators

```python
# Mersenne Twister (recommended)
mt = ql.MersenneTwisterUniformRsg(dimension=1, seed=12345)
sample = mt.nextSequence()              # Returns SampleRealVector
random_value = sample.value[0]          # Extract value

# Knuth uniform
knuth = ql.KnuthUniformRsg(dimension=1, seed=12345)

# L'Ecuyer uniform
lecuyer = ql.LecuyerUniformRsg(dimension=1, seed=12345)
```

#### Quasi-Random (Low-Discrepancy) Generators

```python
# Halton sequence
halton = ql.HaltonRsg(dimension=2, seed=0, skip=1000)

# Sobol sequence (most common for finance)
sobol = ql.SobolRsg(dimension=3, seed=0)

# Faure sequence
faure = ql.FaureRsg(dimension=2)

# Generate samples
for i in range(100):
    sample = sobol.nextSequence()
    point = sample.value  # Array with 3 quasi-random numbers
```

#### Random Number Transformations

```python
# Box-Muller for normal variates
uniform_rng = ql.MersenneTwisterUniformRsg(2, 12345)
box_muller = ql.BoxMullerGaussianRsg(uniform_rng)

normal_sample = box_muller.nextSequence()
normal_values = normal_sample.value    # Two correlated normal variables

# Inverse transform for normal
inv_gauss = ql.InvGaussianRsg(uniform_rng)
```

## Numerical Integration

### 1D Integration

```python
# Adaptive integration
integrator = ql.GaussKronrodAdaptive(tolerance=1e-10, max_evaluations=1000)

# Define integrand
def integrand(x):
    return x * x  # Integrate x^2 from 0 to 2

result = integrator(integrand, 0.0, 2.0)  # Should give 8/3

# Gauss-Legendre quadrature
gauss_legendre = ql.GaussLegendreIntegration(n=16)
result_gl = gauss_legendre(integrand, 0.0, 2.0)

# Discrete integration (for payoff integration)
discrete_integrator = ql.DiscreteTrapezoidIntegrator()
```

### Multi-Dimensional Integration

```python
# Monte Carlo integration for high dimensions
def multivariate_function(x):
    # x is Array with multiple variables
    return sum(xi*xi for xi in x)  # Sum of squares

# Use with random number generators for MC integration
```

## Usage Examples

### Curve Fitting with Optimization

```python
import QuantLib as ql
import math

# Market data: time to maturity and zero rates
market_times = ql.Array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
market_rates = ql.Array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045])

# Define Nelson-Siegel curve: r(t) = beta0 + beta1*exp(-t/tau) + beta2*(t/tau)*exp(-t/tau)
class NelsonSiegelCost(ql.CostFunction):
    def __init__(self, times, market_rates):
        ql.CostFunction.__init__(self)
        self.times = times
        self.market_rates = market_rates
    
    def values(self, params):
        beta0, beta1, beta2, tau = params[0], params[1], params[2], params[3]
        residuals = ql.Array(self.times.size())
        
        for i in range(self.times.size()):
            t = self.times[i]
            if tau > 0:
                exp_t_tau = math.exp(-t/tau)
                model_rate = beta0 + beta1 * exp_t_tau + beta2 * (t/tau) * exp_t_tau
            else:
                model_rate = beta0  # Fallback for invalid tau
            
            residuals[i] = model_rate - self.market_rates[i]
        
        return residuals
    
    def value(self, params):
        residuals = self.values(params)
        return sum(r*r for r in residuals)  # Sum of squared residuals

# Set up optimization
cost_function = NelsonSiegelCost(market_times, market_rates)
initial_guess = ql.Array([0.04, -0.01, -0.01, 2.0])  # beta0, beta1, beta2, tau
constraint = ql.NoConstraint()

# End criteria
end_criteria = ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)

# Optimize
optimizer = ql.LevenbergMarquardt()
problem = ql.Problem(cost_function, constraint, initial_guess)
result = optimizer.minimize(problem, end_criteria)

# Results
fitted_params = problem.currentValue()
print(f"Fitted parameters: beta0={fitted_params[0]:.6f}, beta1={fitted_params[1]:.6f}, beta2={fitted_params[2]:.6f}, tau={fitted_params[3]:.6f}")
```

### Monte Carlo Portfolio Simulation

```python
# Simulate correlated asset returns
import QuantLib as ql
import numpy as np
import math

# Portfolio of 3 assets
n_assets = 3
n_simulations = 10000
time_horizon = 1.0  # 1 year

# Expected returns and volatilities
mu = ql.Array([0.08, 0.12, 0.15])      # Expected returns
sigma = ql.Array([0.15, 0.25, 0.35])   # Volatilities

# Correlation matrix
correlation = ql.Matrix(3, 3)
correlation[0][0] = correlation[1][1] = correlation[2][2] = 1.0
correlation[0][1] = correlation[1][0] = 0.3
correlation[0][2] = correlation[2][0] = 0.2  
correlation[1][2] = correlation[2][1] = 0.5

# Convert to covariance matrix
covariance = ql.Matrix(3, 3)
for i in range(3):
    for j in range(3):
        covariance[i][j] = correlation[i][j] * sigma[i] * sigma[j]

# Random number generator
rng = ql.MersenneTwisterUniformRsg(n_assets, 12345)
gaussian_rng = ql.BoxMullerGaussianRsg(rng)

# Portfolio weights
weights = ql.Array([0.4, 0.35, 0.25])

# Simulate portfolio returns
portfolio_returns = []
for i in range(n_simulations):
    # Generate correlated normal variables
    sample = gaussian_rng.nextSequence()
    random_normals = sample.value
    
    # Apply correlation (simplified - should use Cholesky decomposition)
    portfolio_return = 0.0
    for j in range(n_assets):
        asset_return = mu[j] * time_horizon + sigma[j] * math.sqrt(time_horizon) * random_normals[j]
        portfolio_return += weights[j] * asset_return
    
    portfolio_returns.append(portfolio_return)

# Analyze results
stats = ql.GeneralStatistics()
for ret in portfolio_returns:
    stats.add(ret)

mean_return = stats.mean()
return_volatility = stats.standardDeviation()
print(f"Simulated portfolio: Mean return {mean_return:.4f}, Volatility {return_volatility:.4f}")
```

### Volatility Surface Interpolation

```python
# Create volatility surface from market data
strikes = [80, 90, 100, 110, 120]
expiries = [0.25, 0.5, 1.0, 2.0]  # Times to expiry

# Market volatilities (rows = strikes, cols = expiries)
vol_matrix = ql.Matrix(len(strikes), len(expiries))
market_vols = [
    [0.35, 0.32, 0.28, 0.26],  # Strike 80
    [0.30, 0.28, 0.25, 0.24],  # Strike 90  
    [0.28, 0.25, 0.23, 0.22],  # Strike 100
    [0.30, 0.27, 0.24, 0.23],  # Strike 110
    [0.33, 0.30, 0.27, 0.25]   # Strike 120
]

for i, strike_vols in enumerate(market_vols):
    for j, vol in enumerate(strike_vols):
        vol_matrix[i][j] = vol

# Create 2D interpolation
strike_array = ql.Array(strikes)
expiry_array = ql.Array(expiries)
bilinear_interp = ql.BilinearInterpolation(strike_array, expiry_array, vol_matrix)

# Interpolate volatility at arbitrary point
strike_interp = 95.0
expiry_interp = 0.75
interpolated_vol = bilinear_interp(strike_interp, expiry_interp)

print(f"Volatility at strike {strike_interp}, expiry {expiry_interp}: {interpolated_vol:.4f}")

# Create full volatility surface for visualization
strike_grid = [75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
expiry_grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

vol_surface = []
for exp in expiry_grid:
    vol_row = []
    for k in strike_grid:
        vol = bilinear_interp(k, exp)
        vol_row.append(vol)
    vol_surface.append(vol_row)
```

### Statistical Analysis of Returns

```python
# Analyze historical returns data
returns = [-0.02, 0.01, -0.005, 0.015, -0.01, 0.008, 0.003, -0.007, 0.012, -0.004]

# Basic statistics
stats = ql.GeneralStatistics()
for ret in returns:
    stats.add(ret)

print(f"Mean return: {stats.mean():.6f}")
print(f"Volatility: {stats.standardDeviation():.6f}")
print(f"Skewness: {stats.skewness():.4f}")
print(f"Kurtosis: {stats.kurtosis():.4f}")

# Risk statistics
risk_stats = ql.RiskStatistics()
for ret in returns:
    risk_stats.add(ret)

# VaR calculations (assuming normal distribution)
var_95 = risk_stats.gaussianPercentile(0.05)
var_99 = risk_stats.gaussianPercentile(0.01)
expected_shortfall_95 = risk_stats.gaussianExpectedShortfall(0.05)

print(f"95% VaR: {var_95:.6f}")
print(f"99% VaR: {var_99:.6f}")
print(f"95% Expected Shortfall: {expected_shortfall_95:.6f}")

# Downside risk measures
downside_deviation = risk_stats.semiDeviation()
downside_variance = risk_stats.downsideVariance()

print(f"Downside deviation: {downside_deviation:.6f}")
print(f"Downside variance: {downside_variance:.6f}")
```