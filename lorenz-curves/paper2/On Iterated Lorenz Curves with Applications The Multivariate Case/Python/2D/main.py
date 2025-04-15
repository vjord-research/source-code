import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, RegularGridInterpolator

# -------------------------------
# 1. Generate a starting bivariate distribution:
# -------------------------------
np.random.seed(0)
N = 10000  # sample size

# Parameters for two lognormals:
mu1, sigma1 = 0, 0.5
mu2, sigma2 = 0, 1.0

# Correlation for the Gaussian copula:
rho = 0.5
cov = [[1, rho], [rho, 1]]

# Generate bivariate normal samples and transform to lognormal:
Z = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=N)
X1 = np.exp(mu1 + sigma1 * Z[:, 0])
X2 = np.exp(mu2 + sigma2 * Z[:, 1])

# Define a grid on [0,1] using 150 points.
n_grid = 150
grid = np.linspace(0, 1, n_grid)
du = grid[1] - grid[0]


def empirical_quantile(data, u):
    """Return the empirical quantile at level u (0<=u<=1)."""
    return np.percentile(data, 100 * u)


def compute_empirical_F(X1, X2, grid):
    """
    Compute the empirical copula F.
    For each (u,v) in grid^2, F(u,v)=P(X1 <= q1(u), X2 <= q2(v)),
    where q1(u) is the u-quantile of X1 and similarly for X2.
    """
    n = len(grid)
    F = np.zeros((n, n))
    for i, u in enumerate(grid):
        q1 = empirical_quantile(X1, u)
        for j, v in enumerate(grid):
            q2 = empirical_quantile(X2, v)
            F[i, j] = np.mean((X1 <= q1) & (X2 <= q2))
    return F


F0 = compute_empirical_F(X1, X2, grid)


# F0 is our starting distribution (empirical copula).

# -------------------------------
# 2. Define the Lorenz transform operator.
#
# For a given CDF M on [0,1]^2 we define:
#
#   L(M)(x1,x2) = ( ∫₀^(s₁)∫₀^(s₂) u₁*u₂ dM(u₁,u₂) ) / D,
#
# where s₁ = M⁻¹(x1,1) and s₂ = M⁻¹(1,x2) are the inverses of the marginal curves,
# and D = ∫₀¹∫₀¹ u₁*u₂ dM(u₁,u₂).
#
# Here we approximate dM via finite differences and compute the cumulative integral via
# a cumulative sum. We then use RegularGridInterpolator for smooth evaluation.
# -------------------------------
def lorenz_transform(M, grid):
    n = len(grid)
    du = grid[1] - grid[0]

    # Compute the mixed partial derivative using finite differences.
    dM_du = np.gradient(M, du, axis=0)
    f_grid = np.gradient(dM_du, du, axis=1)  # approximate joint density

    # Create meshgrid for u and v.
    X, Y = np.meshgrid(grid, grid, indexing='ij')
    integrand = X * Y * f_grid
    A = integrand * (du ** 2)

    # Compute cumulative sum (2D Riemann sum).
    A_cum = np.cumsum(np.cumsum(A, axis=0), axis=1)
    # Build an interpolator for the cumulative integral.
    acum_interp = RegularGridInterpolator((grid, grid), A_cum, bounds_error=False, fill_value=None)
    D = float(acum_interp((1, 1)))  # full integral over [0,1]^2

    # Invert the marginals of M.
    m1 = M[:, -1]  # m₁(x)=M(x,1)
    m2 = M[-1, :]  # m₂(y)=M(1,y)
    inv_m1 = interp1d(m1, grid, kind='cubic', bounds_error=False, fill_value="extrapolate")
    inv_m2 = interp1d(m2, grid, kind='cubic', bounds_error=False, fill_value="extrapolate")

    M_new = np.zeros_like(M)
    for i, x1 in enumerate(grid):
        s1 = float(inv_m1(x1))
        for j, x2 in enumerate(grid):
            s2 = float(inv_m2(x2))
            num = float(acum_interp((s1, s2)))
            M_new[i, j] = num / D if D != 0 else 0.0
    return M_new


# -------------------------------
# 3. Iterate the Lorenz operator.
# -------------------------------
n_iter = 15  # user-specified number of iterations
surfaces = [F0]  # iteration 0 is F

for k in range(1, n_iter):
    M_prev = surfaces[-1]
    M_new = lorenz_transform(M_prev, grid)
    surfaces.append(M_new)


# -------------------------------
# 4. Compute marginals and define copula.
#
# For a given surface M, define:
#   m₁(u)=M(u,1)   and   m₂(v)=M(1,v).
#
# For the copula: for each (u,v) in [0,1]^2,
#   C(u,v) = M( q₁(u), q₂(v) )
# where q₁ and q₂ are the smooth inverses of the marginals.
# -------------------------------
def compute_marginals(M):
    m1 = M[:, -1]
    m2 = M[-1, :]
    return m1, m2


def get_copula(M, grid):
    m1, m2 = compute_marginals(M)
    inv_m1 = interp1d(m1, grid, kind='cubic', bounds_error=False, fill_value="extrapolate")
    inv_m2 = interp1d(m2, grid, kind='cubic', bounds_error=False, fill_value="extrapolate")
    interp_M = RegularGridInterpolator((grid, grid), M, bounds_error=False, fill_value=None)
    C = np.zeros_like(M)
    for i, u in enumerate(grid):
        s1 = float(inv_m1(u))
        for j, v in enumerate(grid):
            s2 = float(inv_m2(v))
            C[i, j] = float(interp_M((s1, s2)))
    return C


# -------------------------------
# 5. Compute Spearman's rho and Kendall's tau for each iteration.
#
# Spearman's rho is approximated as: ρ ≈ 12 * mean(C) - 3.
# Kendall's tau is approximated via finite differences.
# -------------------------------
spearman_list = []
kendall_list = []
for M in surfaces:
    C = get_copula(M, grid)
    rho_val = 12 * np.mean(C) - 3
    spearman_list.append(rho_val)

    dC_du = np.gradient(C, du, axis=0)
    dC_dv = np.gradient(C, du, axis=1)
    integral = np.sum(dC_du * dC_dv) * (du ** 2)
    tau_val = 1 - 4 * integral
    kendall_list.append(tau_val)

# -------------------------------
# 6. Plotting results.
# -------------------------------

# (a) Plot the marginal curves for m₁ (M(u,1)) through iterations.
plt.figure(figsize=(8, 6))
for k, M in enumerate(surfaces):
    m1, _ = compute_marginals(M)
    plt.plot(grid, m1, label=f'Iter {k}', alpha=0.8)
plt.xlabel('u')
plt.ylabel('m₁(u)=M(u,1)')
plt.title('Marginal m₁ through Iterations')
plt.legend(fontsize=8)
plt.grid(True)
plt.show()

# (b) Plot the marginal curves for m₂ (M(1,v)) through iterations.
plt.figure(figsize=(8, 6))
for k, M in enumerate(surfaces):
    _, m2 = compute_marginals(M)
    plt.plot(grid, m2, label=f'Iter {k}', alpha=0.8)
plt.xlabel('v')
plt.ylabel('m₂(v)=M(1,v)')
plt.title('Marginal m₂ through Iterations')
plt.legend(fontsize=8)
plt.grid(True)
plt.show()

# (c) Copula contour plots through iterations.
ncols = 5
nrows = int(np.ceil(n_iter / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(16, 6))
for idx in range(n_iter):
    C = get_copula(surfaces[idx], grid)
    ax = axs[idx // ncols, idx % ncols]
    cp = ax.contourf(grid, grid, C, levels=20, cmap='viridis')
    ax.contour(grid, grid, C, colors='k', linewidths=0.5)
    ax.set_title(f'Iter {idx}')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
fig.suptitle('Copula Contour Plots Through Iterations', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# (d) Joint plot of Spearman's rho and Kendall's tau evolution.
iterations = np.arange(n_iter)
plt.figure(figsize=(8, 6))
plt.plot(iterations, spearman_list, marker='o', label="Spearman's ρ")
plt.plot(iterations, kendall_list, marker='s', label="Kendall's τ")
plt.xlabel('Iteration')
plt.ylabel('Correlation Measure')
plt.title("Evolution of Spearman's ρ and Kendall's τ")
plt.legend()
plt.grid(True)
plt.show()

# (e) 3D surface plot for the starting distribution F (Iteration 0).
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
U, V = np.meshgrid(grid, grid)
ax.plot_surface(U, V, surfaces[0], cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title('3D Plot: Iteration 0 (F - Empirical Copula)')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('F(u,v)')
plt.show()

# (f) 3D surface plot for the final distribution Lⁿ (Iteration n_iter-1).
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(U, V, surfaces[-1], cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title(f'3D Plot: Iteration {n_iter - 1} (Lⁿ - Lorenz Transform)')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('L(u,v)')
plt.show()

# (g) For the final iteration marginals, plot the difference between them and the "gold section" CDF.
# We assume the target (gold section) CDF is: ϕ(x)= x^φ, with φ=(√5-1)/2.
phi_const = (np.sqrt(5) + 1) / 2  # approximately 0.618
gold_target = grid ** phi_const  # for x in [0,1]

# Get final iteration marginals.
m1_final, m2_final = compute_marginals(surfaces[-1])

# Plot difference for marginal m₁.
plt.figure(figsize=(8, 6))
plt.plot(grid, m1_final - gold_target, label='m₁ - gold target', color='b')
plt.xlabel('u')
plt.ylabel('Difference')
plt.title('Difference: Final m₁(u) - x^(φ)')
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True)
plt.show()

# Plot difference for marginal m₂.
plt.figure(figsize=(8, 6))
plt.plot(grid, m2_final - gold_target, label='m₂ - gold target', color='r')
plt.xlabel('v')
plt.ylabel('Difference')
plt.title('Difference: Final m₂(v) - x^(φ)')
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True)
plt.show()
