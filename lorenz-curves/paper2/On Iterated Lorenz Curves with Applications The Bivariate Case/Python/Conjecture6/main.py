import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr, kendalltau
from scipy.interpolate import interp1d, RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import math

# === USER CHOICE: which starting example? ===
EXAMPLE = 1   # set to 1 or 2

# PARAMETERS
N_ITER   = 20       # number of iterations
GRID_N   = 201      # resolution for x-grid
COP_GRID = 101      # resolution for copula u-grid
MC_SAMP  = 30000    # Monte Carlo draws for dependence measures

def dynamic_ncol(num_labels, max_per_col=20):
    return 1 if num_labels <= max_per_col else math.ceil(num_labels / max_per_col)

# === GRID SETUP ===
x = np.linspace(0,1,GRID_N)
h = x[1] - x[0]
X1, X2 = np.meshgrid(x, x, indexing='ij')
u = np.linspace(0,1,COP_GRID)
U1, U2 = np.meshgrid(u, u, indexing='ij')
p = np.linspace(0,1,GRID_N)

# === INITIAL DISTRIBUTION ===
if EXAMPLE == 1:
    # Original highly oscillatory example
    def m0(u): return 0.5 + 0.4 * np.sin(10 * np.pi * u)
    sigma = 0.02
    Phi, phi = norm.cdf, norm.pdf
    den = Phi((1 - m0(X1)) / sigma) - Phi(-m0(X1) / sigma)
    f2g1 = phi((X2 - m0(X1)) / sigma) / (sigma * den)
    f1_0 = np.ones_like(x)
elif EXAMPLE == 2:
    # Pathological example where A_n(s) is very nonmonotone
    f1_0 = 6 * x * (1 - x)  # Beta(2,2)
    # copula-like perturbation
    f2g1 = 1 + 0.5 * np.sin(8 * np.pi * X1) * np.cos(8 * np.pi * X2)
else:
    raise ValueError("EXAMPLE must be 1 or 2")

# Compute original marginal and CDF
f2_0 = (f1_0[:,None] * f2g1).sum(axis=0) * h
F1_0 = np.cumsum(f1_0) * h
F2_0 = np.cumsum(f2_0) * h

# === PLOT 0: Original Marginal PDFs & CDFs ===
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(x, f1_0, 'C0-', lw=2, label='f1₀(x)')
axes[0].plot(x, f2_0, 'C1-', lw=2, label='f2₀(x)')
axes[0].set(title='Original Marginal PDFs', xlabel='x', ylabel='Density')
axes[0].grid(); axes[0].legend()
axes[1].plot(x, F1_0, 'C0-', lw=2, label='F1₀(x)')
axes[1].plot(x, F2_0, 'C1-', lw=2, label='F2₀(x)')
axes[1].plot(x, x, 'k--', lw=1, label='y=x')
axes[1].set(title='Original Marginal CDFs', xlabel='x', ylabel='CDF')
axes[1].grid(); axes[1].legend()
plt.tight_layout(); plt.show()

# === ITERATION & STORAGE ===
joint = f1_0[:,None] * f2g1
F_list, L1_list, L2_list = [], [], []
rhos, taus, ratios = [], [], []
f1_list, f2_list, m_list, phi_list, phi_list1 = [], [], [], [], []

for n in range(N_ITER+1):
    # joint -> F_n
    integr = X1 * X2 * joint
    mu = integr.sum() * h * h
    F_n = np.cumsum(np.cumsum(integr, axis=1) * h, axis=0) * h / mu
    F_list.append(F_n)
    L1 = F_n[:, -1].copy()
    L2 = F_n[-1, :].copy()
    L1_list.append(L1)
    L2_list.append(L2)
    # marginals f1_n, f2_n
    f1_n = joint.sum(axis=1) * h
    f2_n = joint.sum(axis=0) * h
    f1_list.append(f1_n)
    f2_list.append(f2_n)
    # conditional mean m_n
    m_n = (joint * X2).sum(axis=1) * h / f1_n
    m_list.append(m_n)
    # phi_n(t) on grid p
    Q1 = interp1d(L1, x, bounds_error=False, fill_value=(0,1))
    xq = Q1(p)
    m_interp = interp1d(x, m_n, bounds_error=False, fill_value="extrapolate")
    phi_list.append(xq * m_interp(xq))
    phi_list1.append(m_interp(p))
    # dependence measures
    pmf = joint / joint.sum()
    idx = np.random.choice(GRID_N*GRID_N, MC_SAMP, p=pmf.ravel())
    X1s, X2s = X1.ravel()[idx], X2.ravel()[idx]
    rhos.append(spearmanr(X1s, X2s).correlation)
    taus.append(kendalltau(X1s, X2s).correlation)
    ratios.append((X1s * X2s).mean() / (X1s.mean() * X2s.mean()))
    # next joint
    if n < N_ITER:
        joint = integr / mu

# build quantile maps for the diagonal
Q1_list = [interp1d(L1_list[i], x, fill_value=(0,1), bounds_error=False) for i in range(N_ITER+1)]
Q2_list = [interp1d(L2_list[i], x, fill_value=(0,1), bounds_error=False) for i in range(N_ITER+1)]

# copula surfaces and diagonals
copula_surfaces, C_diag = [], []
for n in range(N_ITER+1):
    fn = RegularGridInterpolator((x,x), F_list[n])
    Cn = np.empty_like(U1)
    for i in range(COP_GRID):
        xi = Q1_list[n](u[i])
        for j in range(COP_GRID):
            Cn[i,j] = fn([xi, Q2_list[n](u[j])])
    copula_surfaces.append(Cn)
    C_diag.append(np.diag(Cn))

# === PLOT 2: One-Iteration Marginal PDFs & CDFs ===
pdf1_1 = np.gradient(L1_list[1], x)
pdf2_1 = np.gradient(L2_list[1], x)
fig, axs = plt.subplots(1, 2, figsize=(10,4))
axs[0].plot(x, pdf1_1, 'C0-', lw=2, label='f1₁(x)')
axs[0].plot(x, pdf2_1, 'C1-', lw=2, label='f2₁(x)')
axs[0].set(title='One-Iteration Marginal PDFs', xlabel='x', ylabel='Density')
axs[0].grid(); axs[0].legend()
axs[1].plot(x, L1_list[1], 'C0-', lw=2, label='F1₁(x)')
axs[1].plot(x, L2_list[1], 'C1-', lw=2, label='F2₁(x)')
axs[1].plot(x, x, 'k--', lw=1, label='y=x')
axs[1].set(title='One-Iteration Marginal CDFs', xlabel='x', ylabel='CDF')
axs[1].grid(); axs[1].legend()
plt.tight_layout(); plt.show()

# === PLOT A1: Starting Marginals ===
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(x, L1_list[0], 'C0-', lw=2, label='F1₀(x)')
axes[0].plot(x, x, 'k--', lw=1, label='y=x')
axes[0].set(title='L₀(x,1)', xlabel='x', ylabel='CDF'); axes[0].grid()
axes[1].plot(x, L2_list[0], 'C1-', lw=2, label='F2₀(x)')
axes[1].plot(x, x, 'k--', lw=1, label='y=x')
axes[1].set(title='L₀(1,x)', xlabel='x', ylabel='CDF'); axes[1].grid()
hnd, lbl = axes[1].get_legend_handles_labels()
axes[1].legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
               ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.85,1]); plt.show()

# === PLOT A2: Conditional Mean m(u) ===
plt.figure(figsize=(6,4))
plt.plot(u, interp1d(x, m_list[0], fill_value="extrapolate")(u), 'C2-', lw=2, label='m₀(u)')
plt.title('Conditional Mean m₀(u)'); plt.xlabel('u'); plt.ylabel('m(u)'); plt.grid(True)
hnd, lbl = plt.gca().get_legend_handles_labels()
plt.legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
           ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.8,1]); plt.show()

# === PLOT A3: Initial 2D CDF Surface ===
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, F_list[0], cmap='viridis', edgecolor='none')
ax.set(title='Initial 2D CDF F₀(x₁,x₂)', xlabel='x₁', ylabel='x₂', zlabel='F₀')
plt.tight_layout(); plt.show()

# === PLOT A4: Initial Copula Surface ===
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(U1, U2, copula_surfaces[0], cmap='plasma', edgecolor='none')
ax.set(title='Initial Copula Surface C₀(u₁,u₂)', xlabel='u₁', ylabel='u₂', zlabel='C₀')
plt.tight_layout(); plt.show()

# === PLOT 3: Marginal Evolutions ===
fig, axes = plt.subplots(1,2,figsize=(10,4))
for n in range(N_ITER+1):
    axes[0].plot(x, L1_list[n], label=f'n={n}')
    axes[1].plot(x, L2_list[n], label=f'n={n}')
for ax in axes:
    ax.plot(x, x, 'k--', lw=1)
    ax.set(xlabel='x', ylabel='CDF'); ax.grid(True)
axes[0].set(title='Lₙ(x,1)'); axes[1].set(title='Lₙ(1,x)')
hnd, lbl = axes[1].get_legend_handles_labels()
axes[1].legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
               ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.85,1]); plt.show()

# === PLOT 4: Quantile Evolutions ===
fig, axes = plt.subplots(1,2,figsize=(10,4))
for n in range(N_ITER+1):
    axes[0].plot(p, Q1_list[n](p), label=f'n={n}')
    axes[1].plot(p, Q2_list[n](p), label=f'n={n}')
for ax in axes:
    ax.plot(p, p, 'k--', lw=1)
    ax.set(xlabel='p', ylabel='Quantile'); ax.grid(True)
axes[0].set(title='Q₁ₙ(p)'); axes[1].set(title='Q₂ₙ(p)')
hnd, lbl = axes[1].get_legend_handles_labels()
axes[1].legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
               ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.85,1]); plt.show()

# === PLOT 5: Dependence Coefficients ===
plt.figure(figsize=(6,4))
plt.plot(range(N_ITER+1), rhos, '-o', label="Spearman ρ")
plt.plot(range(N_ITER+1), taus, '-s', label="Kendall τ")
plt.title('Dependence Coefficients'); plt.xlabel('n'); plt.ylabel('Value'); plt.grid(True)
hnd, lbl = plt.gca().get_legend_handles_labels()
plt.legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
           ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.8,1]); plt.show()

# === PLOT 6: Moment Ratio ===
plt.figure(figsize=(6,4))
plt.plot(range(N_ITER+1), ratios, '-^', label='E[X₁X₂]/(E[X₁]E[X₂])')
plt.title('Moment Ratio'); plt.xlabel('n'); plt.ylabel('Ratio'); plt.grid(True)
hnd, lbl = plt.gca().get_legend_handles_labels()
plt.legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
           ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.8,1]); plt.show()

# === PLOT 7: Final 2D CDF Surface ===
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, F_list[-1], cmap='plasma', edgecolor='none')
ax.set(title=f'Final 2D CDF F_{N_ITER}(x₁,x₂)', xlabel='x₁', ylabel='x₂', zlabel=f'F_{N_ITER}')
plt.tight_layout(); plt.show()

# === PLOT 8: Final Copula Surface ===
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(U1, U2, copula_surfaces[-1], cmap='viridis', edgecolor='none')
ax.set(title=f'Final Copula Surface C_{N_ITER}(u₁,u₂)', xlabel='u₁', ylabel='u₂', zlabel=f'C_{N_ITER}')
plt.tight_layout(); plt.show()

# === PLOT 9: Copula Diagonal Evolution ===
plt.figure(figsize=(6,4))
for n in range(N_ITER+1):
    plt.plot(u, C_diag[n], label=f'n={n}', lw=2)
plt.plot(u, u, 'k--', lw=2, label='M(u,u)')
plt.plot(u, np.maximum(2*u-1,0), 'k-.', lw=2, label='W(u,u)')
plt.plot(u, u*u, 'k:', lw=2, label='Π(u,u)')
plt.title('Copula Diagonal Evolution'); plt.xlabel('u'); plt.ylabel('Cₙ(u,u)'); plt.grid(True)
hnd, lbl = plt.gca().get_legend_handles_labels()
plt.legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
           ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.8,1]); plt.show()

# === PLOT 10: Compound Quantile Compositions Φ₁ₙ & Φ₂ₙ ===
fig, axes = plt.subplots(1,2,figsize=(10,4))
for n in range(N_ITER+1):
    comp1 = x.copy()
    for k in range(n+1):
        comp1 = Q1_list[k](comp1)
    axes[0].plot(x, comp1, label=f'n={n}')
    comp2 = x.copy()
    for k in range(n+1):
        comp2 = Q2_list[k](comp2)
    axes[1].plot(x, comp2, label=f'n={n}')
for ax, title in zip(axes, ['Φ₁ₙ(x)', 'Φ₂ₙ(x)']):
    ax.plot(x, x, 'k--', lw=1, label='id')
    ax.set(title=title, xlabel='x', ylabel='Composite'); ax.grid(True)
hnd, lbl = axes[1].get_legend_handles_labels()
axes[1].legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
               ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.85,1]); plt.show()

# === PLOT 11: Evolution of Marginal PDFs ===
fig, axes = plt.subplots(1,2,figsize=(10,4))
for n in range(N_ITER+1):
    axes[0].plot(x, f1_list[n], label=f'n={n}')
    axes[1].plot(x, f2_list[n], label=f'n={n}')
axes[0].set(title='Evolution of fₙ,₁(x)', xlabel='x', ylabel='PDF'); axes[0].grid(True)
axes[1].set(title='Evolution of fₙ,₂(x)', xlabel='x', ylabel='PDF'); axes[1].grid(True)
hnd, lbl = axes[1].get_legend_handles_labels()
axes[1].legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
               ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.85,1]); plt.show()

# === PLOT 12 (smoothed): Evolution of Aₙ(s)=fₙ,ᵢ(Fₙ,ᵢ⁻¹(s)) for i=1,2 ===
from scipy.interpolate import interp1d

fig, axes = plt.subplots(1, 2, figsize=(10,4))
for n in range(N_ITER+1):
    # build smooth interpolants of the two marginals
    f1_interp = interp1d(x, f1_list[n], kind='cubic',
                         bounds_error=False, fill_value="extrapolate")
    f2_interp = interp1d(x, f2_list[n], kind='cubic',
                         bounds_error=False, fill_value="extrapolate")

    # evaluate at the exact quantile values Q1_list[n](p) and Q2_list[n](p)
    A1 = f1_interp(Q1_list[n](p))
    A2 = f2_interp(Q2_list[n](p))

    axes[0].plot(p, A1, label=f'n={n}')
    axes[1].plot(p, A2, label=f'n={n}')

axes[0].set(title=r'$A_{n}^{(1)}(s)=f_{n,1}(F_{n,1}^{-1}(s))$',
            xlabel='s', ylabel=r'$A_{n}^{(1)}(s)$')
axes[1].set(title=r'$A_{n}^{(2)}(s)=f_{n,2}(F_{n,2}^{-1}(s))$',
            xlabel='s', ylabel=r'$A_{n}^{(2)}(s)$')
for ax in axes:
    ax.grid(True)
hnd, lbl = axes[1].get_legend_handles_labels()
axes[1].legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
               ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.85,1])
plt.show()


# === PLOT 13: Evolution of φₙ(t) ===
fig, ax = plt.subplots(figsize=(8,4))
for n in range(N_ITER+1):
    ax.plot(p, phi_list[n], label=f'n={n}')
ax.set(title='Evolution of φₙ(t)', xlabel='t', ylabel='φₙ(t)'); ax.grid(True)
hnd, lbl = ax.get_legend_handles_labels()
ax.legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
          ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.8,1]); plt.show()

# === PLOT 13: Evolution of φₙ(t) ===
fig, ax = plt.subplots(figsize=(8,4))
for n in range(N_ITER+1):
    ax.plot(p, np.divide(phi_list[n],p), label=f'n={n}')
ax.set(title='Evolution of φₙ(t)', xlabel='t', ylabel='φₙ(t)'); ax.grid(True)
hnd, lbl = ax.get_legend_handles_labels()
ax.legend(hnd, lbl, loc='center left', bbox_to_anchor=(1.02,0.5),
          ncol=dynamic_ncol(len(lbl)))
plt.tight_layout(rect=[0,0,0.8,1]); plt.show()


