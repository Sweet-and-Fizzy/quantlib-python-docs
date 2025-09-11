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

QuantLib provides various interpolation methods for curve construction and data fitting.

### 1D Interpolation

#### Linear Interpolation

```python
# Data points
x_values = [1.0, 2.0, 3.0, 4.0, 5.0]
y_values = [1.0, 4.0, 9.0, 16.0, 25.0]

# Create interpolation
linear_interp = ql.LinearInterpolation(x_values, y_values)
linear_interp.update()       # Finalize interpolation

# Interpolate at point
x_point = 2.5
y_interpolated = linear_interp(x_point)  # Should be ~6.5

# Check bounds
linear_interp.allowsExtrapolation()     # False by default
linear_interp.enableExtrapolation()     # Allow extrapolation
```

#### Cubic Spline Interpolation

```python
# Natural cubic spline
cubic_interp = ql.CubicInterpolation(
    x_values, y_values,
    CubicInterpolation.Spline,      # Spline type
    True,                           # Monotonic
    CubicInterpolation.SecondDerivative,  # Left boundary condition
    0.0,                            # Left boundary value
    CubicInterpolation.SecondDerivative,  # Right boundary condition  
    0.0                             # Right boundary value
)
cubic_interp.update()

# Cubic interpolation types
CubicInterpolation.Spline          # Cubic spline
CubicInterpolation.SplineOM1       # Spline with first derivative matching
CubicInterpolation.SplineOM2       # Spline with second derivative matching
CubicInterpolation.Akima           # Akima spline
CubicInterpolation.Fritsch_Butland # Fritsch-Butland (monotonic)
CubicInterpolation.Parabolic       # Parabolic
CubicInterpolation.FritschButland  # Monotonic cubic
CubicInterpolation.Kruger          # Kruger spline
```

#### Logarithmic Interpolation

```python
# Log-linear interpolation (interpolates log(y) linearly)
log_interp = ql.LogLinearInterpolation(x_values, y_values)
log_interp.update()

# Log-cubic interpolation
log_cubic = ql.LogCubicInterpolation(x_values, y_values)
log_cubic.update()
```

#### Specialized Interpolations

```python
# Backward flat (step function)
backward_flat = ql.BackwardFlatInterpolation(x_values, y_values)

# Forward flat
forward_flat = ql.ForwardFlatInterpolation(x_values, y_values)

# Convex monotone (for yield curves)
convex_monotone = ql.ConvexMonotoneInterpolation(x_values, y_values)

# SABR interpolation (for volatility)
sabr_interp = ql.SABRInterpolation(
    x_begin=x_values,
    x_end=x_values,
    y_begin=y_values,
    expiry=1.0,      # Time to expiry
    forward=100.0,   # Forward price
    alpha=0.3,       # SABR alpha
    beta=0.7,        # SABR beta  
    nu=0.4,          # SABR nu
    rho=-0.1         # SABR rho
)
```

### 2D Interpolation

For volatility surfaces and other two-dimensional data.

```python
# Bilinear interpolation
x_vals = [1.0, 2.0, 3.0]        # First dimension (e.g., strike)
y_vals = [0.25, 0.5, 1.0]       # Second dimension (e.g., time)
z_matrix = ql.Matrix(3, 3)      # 3x3 data matrix

# Fill z_matrix with data...
bilinear = ql.BilinearInterpolation(x_vals, y_vals, z_matrix)

# Interpolate at point
x_point, y_point = 1.5, 0.75
z_value = bilinear(x_point, y_point)

# Bicubic interpolation
bicubic = ql.BicubicSplineInterpolation(x_vals, y_vals, z_matrix)
```

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