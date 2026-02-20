#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats, integrate, optimize
from scipy.interpolate import interp1d

# Compatibility for scipy versions
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ------------------------------------------------------------
# Utilities for A_n, integrals I_n and M_n
# ------------------------------------------------------------

def get_A_func(Q_func):
    """
    Build A_n(z) = Q_n(z) * (1 - Q_n(z)) from a quantile function Q_n.
    Returns a safe, vectorized callable on [0, 1].
    """

    def _A(z):
        z_arr = np.atleast_1d(z).astype(float)
        z_arr = np.clip(z_arr, 0.0, 1.0)
        try:
            q = Q_func(z_arr)
        except Exception:
            q = np.array([Q_func(float(zi)) for zi in z_arr], dtype=float)
        q = np.clip(q, 0.0, 1.0)
        val = q * (1.0 - q)
        if np.ndim(z) == 0:
            return float(val[0])
        return val

    return _A


def calculate_I_n(A_func):
    """
    I_n = ∫_0^1 A_n(z) dz  with high-accuracy quadrature.
    """

    def integrand(z):
        if 0.0 < z < 1.0:
            try:
                v = A_func(z)
                if not np.isfinite(v):
                    return 0.0
                return float(v)
            except Exception:
                return 0.0
        return 0.0

    try:
        I_val, I_err = integrate.quad(
            integrand, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=300
        )
        if not np.isfinite(I_val) or I_val <= 0.0:
            return 1e-15, 0.0
        return I_val, I_err
    except Exception:
        return None, None


def calculate_M_n(A_func):
    """
    M_n = ∫_0^1 z(1-z) A_n(z) dz
    """

    def integrand(z):
        if 0.0 < z < 1.0:
            try:
                v = A_func(z)
                if not np.isfinite(v):
                    return 0.0
                return float(z * (1.0 - z) * v)
            except Exception:
                return 0.0
        return 0.0

    try:
        M_val, M_err = integrate.quad(
            integrand, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=300
        )
        if not np.isfinite(M_val):
            return None, None
        return M_val, M_err
    except Exception:
        return None, None


# ------------------------------------------------------------
# New Math: K_* and K_** logic (Coupled & Grid-Stabilized)
# ------------------------------------------------------------

def phi_star(x):
    # Maps [0, 0.25] -> [0, 0.5]
    # x is clamped to avoid negative sqrt
    x = np.clip(x, 0.0, 0.25)
    return 0.5 * (1.0 - np.sqrt(1.0 - 4.0 * x))


def phi_star_star(x):
    # Maps [0, 0.25] -> [0.5, 1.0]
    x = np.clip(x, 0.0, 0.25)
    return 0.5 * (1.0 + np.sqrt(1.0 - 4.0 * x))


def invert_monotonic(x, y):
    """
    Invert a function defined by arrays x, y.
    Returns an interp1d object mapping range(y) -> domain(x).
    """
    # Check monotonicity
    dy = np.diff(y)

    # Handle strictly constant case (numerical artifact)
    if np.all(dy == 0):
        return lambda u: np.full_like(u, x[0])

    if np.all(dy >= 0):
        # Increasing
        keep = np.concatenate(([True], dy > 0))
        y_u, x_u = y[keep], x[keep]
        # Ensure we have at least 2 points
        if len(y_u) < 2:
            return interp1d([y[0], y[-1]], [x[0], x[-1]], kind='linear', bounds_error=False, fill_value="extrapolate")
        return interp1d(y_u, x_u, kind='linear', bounds_error=False,
                        fill_value=(x_u[0], x_u[-1]))
    elif np.all(dy <= 0):
        # Decreasing
        keep = np.concatenate(([True], dy < 0))
        y_u, x_u = y[keep], x[keep]
        if len(y_u) < 2:
            return interp1d([y[0], y[-1]], [x[0], x[-1]], kind='linear', bounds_error=False, fill_value="extrapolate")
        # Flip for interp1d (x must be increasing for scipy)
        return interp1d(y_u[::-1], x_u[::-1], kind='linear', bounds_error=False,
                        fill_value=(x_u[-1], x_u[0]))
    else:
        # Non-monotonic (numerical noise?), sort by y
        idx = np.argsort(y)
        y_s, x_s = y[idx], x[idx]
        _, u_idx = np.unique(y_s, return_index=True)
        return interp1d(y_s[u_idx], x_s[u_idx], kind='linear', bounds_error=False,
                        fill_value="extrapolate")


def reconstruct_L(K_star_func, K_star_star_func):
    """
    Reconstruct global L_n(x).
    """
    x_grid = np.linspace(0.0, 1.0, 4001)
    w_vals = x_grid * (1.0 - x_grid)  # maps [0,1] -> [0, 0.25]

    L_vals = np.zeros_like(x_grid)
    mask_left = x_grid <= 0.5
    mask_right = ~mask_left

    L_vals[mask_left] = K_star_func(w_vals[mask_left])
    L_vals[mask_right] = K_star_star_func(w_vals[mask_right])

    L_vals[0] = 0.0
    L_vals[-1] = 1.0
    L_vals = np.clip(L_vals, 0.0, 1.0)

    # Interpolator for L
    L_func = interp1d(x_grid, L_vals, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))

    # Invert to get Q
    L_vals_clean = np.maximum.accumulate(L_vals)
    keep = np.concatenate(([True], np.diff(L_vals_clean) > 0))
    if np.sum(keep) < 2:
        Q_func = lambda x: x
    else:
        x_u = L_vals_clean[keep]
        y_u = x_grid[keep]
        Q_func = interp1d(x_u, y_u, kind='linear', bounds_error=False, fill_value=(y_u[0], y_u[-1]))

    return L_func, Q_func


# ------------------------------------------------------------
# Initial distributions
# ------------------------------------------------------------

def get_float_parameter(prompt, default=None, positive=True):
    try:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return float(default)
        v = float(s)
        if positive and v <= 0.0:
            raise ValueError
        return v
    except Exception:
        print("  Invalid numeric input.")
        raise


def get_int_parameter(prompt, default, min_value=1, require_odd=False):
    try:
        s = input(prompt).strip()
        v = int(s) if s != "" else int(default)
        if v < min_value:
            raise ValueError
        if require_odd and (v % 2 == 0):
            v += 1
        return v
    except Exception:
        print("  Invalid integer input.")
        raise


def get_initial_distribution():
    print("\nChoose initial CDF F:")
    print("  1) Uniform(0,1)")
    print("  2) Beta(α, α)      [default α = 2.0]")
    print("  3) Trunc-Scaled Lognormal  [default μ=0.5, σ=0.5, [0,2]]")
    print("  4) Trunc-Scaled Pareto     [default α=2, xm=1, [1,8]]")
    print("  5) Trunc-Scaled Gamma      [default k=2, θ=1, [0,8]]")
    choice = input("Your choice [1-5] (default 2): ").strip()
    if choice == "":
        choice = "2"

    if choice == "1":
        dist = stats.uniform(loc=0.0, scale=1.0)
        return dist, "Uniform", {}
    if choice == "2":
        alpha = get_float_parameter("  α > 0 (default 2.0): ", default=2.0)
        dist = stats.beta(alpha, alpha, loc=0.0, scale=1.0)
        return dist, f"Beta({alpha:.3g},{alpha:.3g})", {"alpha": alpha}
    if choice == "3":
        mu = get_float_parameter("  Lognormal μ (default 0.5): ", default=0.5, positive=False)
        sigma = get_float_parameter("  Lognormal σ > 0 (default 0.5): ", default=0.5)
        a = get_float_parameter("  Truncation lower a >= 0 (default 0.0): ", default=0.0, positive=False)
        b = get_float_parameter("  Truncation upper b > a (default 2.0): ", default=2.0, positive=False)
        base = stats.lognorm(s=sigma, scale=np.exp(mu))
        return make_trunc_scaled(base, a, b), f"TruncLognorm[{a},{b}]", {"mu": mu, "sigma": sigma, "a": a, "b": b}
    if choice == "4":
        alpha = get_float_parameter("  Pareto α > 0 (default 2.0): ", default=2.0)
        xm = get_float_parameter("  Pareto xm > 0 (scale; default 1.0): ", default=1.0)
        a = get_float_parameter("  Truncation lower a >= xm (default 1.0): ", default=xm, positive=False)
        b = get_float_parameter("  Truncation upper b > a (default 8.0): ", default=8.0, positive=False)
        base = stats.pareto(alpha, scale=xm)
        return make_trunc_scaled(base, a, b), f"TruncPareto[{a},{b}]", {"alpha": alpha, "xm": xm, "a": a, "b": b}
    if choice == "5":
        k = get_float_parameter("  Gamma k > 0 (shape, default 2.0): ", default=2.0)
        theta = get_float_parameter("  Gamma θ > 0 (scale, default 1.0): ", default=1.0)
        a = get_float_parameter("  Truncation lower a >= 0 (default 0.0): ", default=0.0, positive=False)
        b = get_float_parameter("  Truncation upper b > a (default 8.0): ", default=8.0, positive=False)
        base = stats.gamma(k, scale=theta)
        return make_trunc_scaled(base, a, b), f"TruncGamma[{a},{b}]", {"k": k, "theta": theta, "a": a, "b": b}
    raise ValueError("Invalid choice.")


def make_trunc_scaled(base_dist, a, b):
    Fa, Fb = base_dist.cdf(a), base_dist.cdf(b)

    def cdf_unit(x):
        x = np.asarray(x, dtype=float)
        t = a + np.clip(x, 0.0, 1.0) * (b - a)
        return np.clip((base_dist.cdf(t) - Fa) / (Fb - Fa), 0.0, 1.0)

    def ppf_unit(u):
        u = np.asarray(u, dtype=float)
        s = np.clip(u, 0.0, 1.0)
        t = base_dist.ppf(Fa + s * (Fb - Fa))
        x = (t - a) / (b - a)
        return np.clip(x, 0.0, 1.0)

    class Wrapper:
        def cdf(self, x): return cdf_unit(x)

        def ppf(self, u): return ppf_unit(u)

    return Wrapper()


# ------------------------------------------------------------
# Root finders
# ------------------------------------------------------------

def find_l_points(A_func, I_val, grid_pts=4001):
    def A_minus_I(x):
        try:
            v = float(A_func(x)) - float(I_val)
            if not np.isfinite(v): return np.nan
            return v
        except:
            return np.nan

    z = np.linspace(0.0, 1.0, int(grid_pts))
    y = np.array([A_minus_I(zi) for zi in z], dtype=float)
    if np.any(~np.isfinite(y)): y[~np.isfinite(y)] = 0.0
    brackets = []
    for i in range(len(z) - 1):
        if y[i] * y[i + 1] < 0.0: brackets.append((z[i], z[i + 1]))
    roots = []
    for (a, b) in brackets:
        try:
            r = optimize.brentq(lambda x: A_minus_I(x), a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
            if 0.0 <= r <= 1.0: roots.append(r)
        except:
            pass
    if not roots: return np.nan, np.nan
    roots = np.unique(np.round(np.array(roots, dtype=float), 14))
    if roots.size == 1: return float(roots[0]), float(roots[0])
    return float(roots[0]), float(roots[-1])


def find_diag_crossings_generic(func, grid_pts=4001):
    def f(x):
        try:
            v = float(func(float(x)))
        except:
            v_arr = func(np.array([x], dtype=float))
            v = float(np.asarray(v_arr)[0])
        return v - float(x)

    z = np.linspace(0.0, 1.0, int(grid_pts))
    y = np.array([f(zi) for zi in z], dtype=float)
    if np.any(~np.isfinite(y)): y[~np.isfinite(y)] = 0.0
    brackets = []
    for i in range(len(z) - 1):
        if y[i] * y[i + 1] < 0.0: brackets.append((z[i], z[i + 1]))
    roots = []
    for (a, b) in brackets:
        try:
            r = optimize.brentq(lambda x: f(x), a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
            if 0.0 <= r <= 1.0: roots.append(r)
        except:
            pass
    if not roots: return []
    return np.unique(np.round(np.array(roots, dtype=float), 14)).tolist()


# ------------------------------------------------------------
# Iteration driver: COUPLED SPLIT LOGIC with ADAPTIVE GRID
# ------------------------------------------------------------

def run_simulation(F_dist, F_name, F_params, N_iter, GRID_PTS):
    N_iter = int(N_iter)

    # --- CRITICAL FIX: NON-UNIFORM Z-GRID ---
    # The transformation z = x(1-x) has zero derivative at x=0.5 (z=0.25).
    # A uniform grid in z is extremely sparse in x near 0.5.
    # We construct z_grid by mapping a uniform x_grid on [0, 0.5].
    # This clusters points heavily near z=0.25, resolving the "Gamma glitch".
    x_half = np.linspace(0.0, 0.5, 2001)
    z_grid = x_half * (1.0 - x_half)
    # Ensure strict monotonicity just in case of numerical noise
    z_grid = np.sort(z_grid)

    L_funcs, Q_funcs, A_funcs, I_vals, M_vals = [], [], [], [], []

    # --- Step 0 ---
    def F_inv(u):
        arr = np.atleast_1d(u).astype(float)
        q = F_dist.ppf(arr)
        return np.clip(q, 0.0, 1.0)

    # H(u) = F^-1(u)(1-F^-1(u))
    def H(u):
        q = F_inv(u)
        return q * (1.0 - q)

    u_fine = np.linspace(0, 1, 4001)
    H_vals = H(u_fine)
    CumH = cumulative_trapezoid(H_vals, u_fine, initial=0.0)
    D0 = CumH[-1]
    CumH_func = interp1d(u_fine, CumH, kind='linear', bounds_error=False, fill_value=(0.0, D0))

    def get_K0_vals(phi_mapping):
        limits = phi_mapping(z_grid)
        return CumH_func(limits) / D0

    K_star_vals = get_K0_vals(phi_star)
    K_star_star_vals = get_K0_vals(phi_star_star)

    K_star_func = interp1d(z_grid, K_star_vals, kind='linear', fill_value="extrapolate")
    K_star_star_func = interp1d(z_grid, K_star_star_vals, kind='linear', fill_value="extrapolate")

    L0, Q0 = reconstruct_L(K_star_func, K_star_star_func)
    A0 = get_A_func(Q0)
    I0, _ = calculate_I_n(A0)
    M0, _ = calculate_M_n(A0)

    L_funcs.append(L0);
    Q_funcs.append(Q0);
    A_funcs.append(A0)
    I_vals.append(I0);
    M_vals.append(0.0 if M0 is None else M0)

    curr_K_star_func = K_star_func
    curr_K_star_star_func = K_star_star_func

    # --- Iteration ---
    for n in range(1, N_iter + 1):
        # 1. Evaluate current K on dense grid
        k_s_vals = curr_K_star_func(z_grid)
        k_ss_vals = curr_K_star_star_func(z_grid)

        # 2. Identify the dynamic mid-point
        # Force strict equality at the boundary to prevent tearing
        mid = 0.5 * (k_s_vals[-1] + k_ss_vals[-1])
        mid = np.clip(mid, 1e-9, 1.0 - 1e-9)
        k_s_vals[-1] = mid
        k_ss_vals[-1] = mid

        # 3. Invert parts
        # K_star maps [0, 0.25] -> [0, mid] (Increasing)
        # K_star_star maps [0, 0.25] -> [1, mid] (Decreasing)

        # Invert K*: returns func mapping [0, mid] -> [0, 0.25] (w values)
        inv_s = invert_monotonic(z_grid, k_s_vals)

        # Invert K**: returns func mapping [mid, 1] -> [0, 0.25] (w values)
        inv_ss = invert_monotonic(z_grid, k_ss_vals)

        # 4. Construct global Q(u) and Integrand J(u) on [0, 1]
        u_grid = np.linspace(0.0, 1.0, 4001)
        J_vals = np.zeros_like(u_grid)

        # Mask for left branch (u <= mid)
        mask_left = u_grid <= mid
        if np.any(mask_left):
            u_part = u_grid[mask_left]
            w_part = inv_s(u_part)  # w in [0, 0.25]
            q_part = phi_star(w_part)  # q in [0, 0.5]
            J_vals[mask_left] = q_part * (1.0 - q_part)

        # Mask for right branch (u > mid)
        mask_right = ~mask_left
        if np.any(mask_right):
            u_part = u_grid[mask_right]
            w_part = inv_ss(u_part)  # w in [0, 0.25]
            q_part = phi_star_star(w_part)  # q in [0.5, 1.0]
            J_vals[mask_right] = q_part * (1.0 - q_part)

        # 5. Integrate to get Normalization and Cumulative Function
        cum_J = cumulative_trapezoid(J_vals, u_grid, initial=0.0)
        D_next = cum_J[-1]

        if D_next <= 1e-15:
            next_vals_s = np.zeros_like(z_grid)
            next_vals_ss = np.ones_like(z_grid) * mid
        else:
            C_func = interp1d(u_grid, cum_J, kind='linear', bounds_error=False, fill_value=(0.0, D_next))

            # 6. Update K* and K**
            next_vals_s = C_func(phi_star(z_grid)) / D_next
            next_vals_ss = C_func(phi_star_star(z_grid)) / D_next

            # Force boundary match again for numerical cleanliness
            common_boundary = 0.5 * (next_vals_s[-1] + next_vals_ss[-1])
            next_vals_s[-1] = common_boundary
            next_vals_ss[-1] = common_boundary

        # Create function objects
        next_K_star_func = interp1d(z_grid, next_vals_s, kind='linear', fill_value="extrapolate")
        next_K_star_star_func = interp1d(z_grid, next_vals_ss, kind='linear', fill_value="extrapolate")

        # 7. Reconstruct global for plotting/logging
        L_next, Q_next = reconstruct_L(next_K_star_func, next_K_star_star_func)
        A_next = get_A_func(Q_next)
        I_next, _ = calculate_I_n(A_next)
        M_next, _ = calculate_M_n(A_next)

        L_funcs.append(L_next);
        Q_funcs.append(Q_next);
        A_funcs.append(A_next)
        I_vals.append(I_next);
        M_vals.append(0.0 if M_next is None else M_next)

        # Prepare for next iter
        curr_K_star_func = next_K_star_func
        curr_K_star_star_func = next_K_star_star_func

    return L_funcs, A_funcs, Q_funcs, np.array(I_vals), np.array(M_vals)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def annotate_indices(ax, x, y, indices=None, fmt="{:.6f}"):
    if len(x) == 0: return
    x = np.asarray(x);
    y = np.asarray(y)
    if indices is None:
        k = max(1, len(x) // 10)
        idxs = list(range(0, len(x), k))
        if (len(x) - 1) not in idxs: idxs.append(len(x) - 1)
    else:
        idxs = indices
    for i in idxs:
        if i < 0 or i >= len(x): continue
        ax.annotate(fmt.format(y[i]), (x[i], y[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)


def plot_sequence(title, x, y, ylabel, show_titles, save_plots, plot_format, filename_base, FIG, annotate_kwargs=None):
    plt.figure(figsize=FIG)
    if show_titles and title: plt.title(title)
    plt.plot(x, y, marker='o', lw=1.5)
    plt.xlabel("n");
    plt.ylabel(ylabel);
    plt.grid(True, alpha=0.3)
    if annotate_kwargs is None: annotate_kwargs = {}
    annotate_indices(plt.gca(), x, y, **annotate_kwargs)
    if save_plots: save_current_figure(filename_base, plot_format)


def plot_crossings(title, n_vals, x_vals, ylabel, show_titles, save_plots, plot_format, filename_base, FIG,
                   annotate_kwargs=None):
    plt.figure(figsize=FIG)
    if show_titles and title: plt.title(title)
    plt.scatter(n_vals, x_vals, s=25)
    plt.xlabel("n");
    plt.ylabel(ylabel);
    plt.grid(True, alpha=0.3)
    if annotate_kwargs is None: annotate_kwargs = {}
    annotate_indices(plt.gca(), n_vals, x_vals, **annotate_kwargs)
    if save_plots: save_current_figure(filename_base, plot_format)


def save_current_figure(filename_base, fmt):
    output_dir = "plot_output"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{filename_base}.{fmt}")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"  Saved: {path}")


def plot_results(L_funcs, A_funcs, Q_funcs, I_vals, M_vals, F_name, show_titles, save_plots, plot_format, size_pct,
                 GRID_PTS):
    N = len(I_vals)
    if N == 0: return
    x_plot = np.linspace(0.0, 1.0, 401)
    base = (6.0, 4.5)
    s = 1.0 + (size_pct / 100.0)
    FIG = (base[0] * s, base[1] * s)
    cmap = cm.viridis
    norm_denom = (N - 1) if N > 1 else 1.0

    # 1. Ln
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $L_n(x)$")
    for n, L in enumerate(L_funcs):
        y = np.array([L(x) for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6)
    plt.xlim(0, 1);
    plt.ylim(0, 1);
    plt.grid(True, alpha=0.3)
    plt.xlabel("x");
    plt.ylabel("$L_n(x)$");
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_1_Ln_{F_name}", plot_format)

    # Evolution of 1 - L_n(1-x)
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $1-L_n(1-x)$")
    for n, L in enumerate(L_funcs):
        y = np.array([1.0 - L(1.0 - x) for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x"); plt.ylabel(r"$1-L_n(1-x)$")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_1b_1mLn_flip_{F_name}", plot_format)

    # Evolution of T_n(x) (T_n = L_n^{-1})
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $T_n(x)$")
    for n, Q in enumerate(Q_funcs):
        y = np.array([Q(x) for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x"); plt.ylabel(r"$T_n(x)$")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_1c_Tn_{F_name}", plot_format)

    # Evolution of 1 - T_n(1-x)
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $1-T_n(1-x)$")
    for n, Q in enumerate(Q_funcs):
        y = np.array([1.0 - Q(1.0 - x) for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x"); plt.ylabel(r"$1-T_n(1-x)$")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_1d_1mTn_flip_{F_name}", plot_format)

    # 1a. Evolution of $L_n(x)+L_n(1-x)-1$
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $L_n(x)+L_n(1-x)-1$")
    for n, L in enumerate(L_funcs):
        y = np.array([L(x) + L(1.0 - x) - 1.0 for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6, linestyle='--')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel(r"$L_n(x)+L_n(1-x)-1$")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_1a_Ln_symdev_{F_name}", plot_format)

    # 1b. Evolution of $T_n(x)+T_n(1-x)-1$   (T_n = L_n^{-1})
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $T_n(x)+T_n(1-x)-1$")
    for n, Q in enumerate(Q_funcs):
        y = np.array([Q(x) + Q(1.0 - x) - 1.0 for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6, linestyle='--')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel(r"$T_n(x)+T_n(1-x)-1$")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_1b_Tn_symdev_{F_name}", plot_format)

    # 1c. Evolution of $(T_n(x)-T_n(1-x))(1-T_n(x)-T_n(1-x))$
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $(T_n(x)-T_n(1-x))(1-T_n(x)-T_n(1-x))$")
    for n, Q in enumerate(Q_funcs):
        y = np.array([(Q(x) - Q(1.0 - x)) * (1.0 - Q(x) - Q(1.0 - x)) for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6, linestyle='--')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel(r"$(T_n(x)-T_n(1-x))(1-T_n(x)-T_n(1-x))$")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_1c_Tn_combo_{F_name}", plot_format)

    # Roots l1, l2 -> c1, c2
    l1_vals = np.full(N, np.nan);
    l2_vals = np.full(N, np.nan)
    for n in range(N):
        l1, l2 = find_l_points(A_funcs[n], I_vals[n], grid_pts=GRID_PTS)
        l1_vals[n] = l1;
        l2_vals[n] = l2
    odd_indices = [i for i, n in enumerate(range(N)) if n % 2 == 1]

    # Updated to cn
    plot_sequence(r"Evolution of $c_n^1$", np.arange(N), l1_vals, r"$c_n^1$", show_titles, save_plots, plot_format,
                  f"plot_1b_c1_{F_name}", FIG, {"indices": odd_indices, "fmt": "{:.3f}"})
    plot_sequence(r"Evolution of $c_n^2$", np.arange(N), l2_vals, r"$c_n^2$", show_titles, save_plots, plot_format,
                  f"plot_1c_c2_{F_name}", FIG, {"indices": odd_indices, "fmt": "{:.3f}"})

    # Diag crossings (ell_n -> c_n)
    Ln_cross_n, Ln_cross_x = [], []
    for n in range(N):
        roots = find_diag_crossings_generic(L_funcs[n], grid_pts=GRID_PTS)
        roots_int = [r for r in roots if 1e-6 < r < 1.0 - 1e-6]
        if roots_int:
            idx = np.argmin(np.abs(np.array(roots_int) - 0.5))
            Ln_cross_n.append(float(n));
            Ln_cross_x.append(float(roots_int[idx]))
    l_diag = np.full(N, np.nan)
    for n_v, x_v in zip(Ln_cross_n, Ln_cross_x): l_diag[int(round(n_v))] = x_v
    odd_idx_L = [i for i, v in enumerate(Ln_cross_n) if int(round(v)) % 2 == 1]

    # Updated to cn
    plot_crossings(r"Evolution of $c_n$", Ln_cross_n, Ln_cross_x, r"$c_n$", show_titles, save_plots, plot_format,
                   f"plot_1f_Cn_diag_cross_{F_name}", FIG, {"indices": odd_idx_L, "fmt": "{:.3f}"})

    # In
    plot_sequence(r"Evolution of $I_n$", np.arange(N), I_vals, r"$I_n$", show_titles, save_plots, plot_format,
                  f"plot_2_In_{F_name}", FIG, {"indices": odd_indices, "fmt": "{:.3f}"})

    # An
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $A_n(z)$")
    for n in range(N):
        y = np.array([A_funcs[n](z) for z in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.xlabel("z");
    plt.ylabel(r"$A_n(z)$");
    plt.grid(True, alpha=0.3);
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_3_An_{F_name}", plot_format)

    # 3a. Evolution of $A_n(1-z)$
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $A_n(1-z)$")
    for n in range(N):
        y = np.array([A_funcs[n](1.0 - z) for z in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.xlabel("z");
    plt.ylabel(r"$A_n(1-z)$");
    plt.grid(True, alpha=0.3);
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_3a_An_flip_{F_name}", plot_format)

    # 3b. Evolution of $A_n(z)-A_n(1-z)$
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $A_n(z)-A_n(1-z)$")
    for n in range(N):
        y = np.array([A_funcs[n](z) - A_funcs[n](1.0 - z) for z in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6);
    plt.xlabel("z");
    plt.ylabel(r"$A_n(z)-A_n(1-z)$");
    plt.grid(True, alpha=0.3);
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_3b_An_diff_{F_name}", plot_format)

    # 3c. Evolution of $\int_0^z (A_n(u)-A_n(1-u))\,du$
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $\int_0^z (A_n(u)-A_n(1-u))\,du$")
    for n in range(N):
        diff_vals = np.array([A_funcs[n](z) - A_funcs[n](1.0 - z) for z in x_plot])
        int_vals = cumulative_trapezoid(diff_vals, x_plot, initial=0.0)
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, int_vals, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6);
    plt.xlabel("z");
    plt.ylabel(r"$\int_0^z (A_n(u)-A_n(1-u))\,du$");
    plt.grid(True, alpha=0.3);
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_3c_An_intdiff_{F_name}", plot_format)

    # 3d. Evolution of zero-crossing $a_n$ of $A_n(z)-A_n(1-z)$ (closest to 1/2)
    a_vals = np.full(N, np.nan)
    for n in range(N):
        z_grid = x_plot
        d = np.array([A_funcs[n](z) - A_funcs[n](1.0 - z) for z in z_grid])
        sgn = np.sign(d)
        ch = np.where(np.diff(sgn) != 0)[0]
        roots = []
        for i0 in ch:
            z1, z2 = z_grid[i0], z_grid[i0+1]
            d1, d2 = d[i0], d[i0+1]
            if np.isfinite(d1) and np.isfinite(d2) and (d2 - d1) != 0:
                r = z1 - d1 * (z2 - z1) / (d2 - d1)
                roots.append(r)
        if roots:
            roots = np.array(roots)
            a_vals[n] = roots[np.argmin(np.abs(roots - 0.5))]
    plot_sequence(r"Evolution of $a_n$", np.arange(N), a_vals, r"$a_n$", show_titles, save_plots, plot_format,
                  f"plot_3d_an_{F_name}", FIG, {"indices": odd_indices, "fmt": "{:.3f}"})

    # Mn
    plot_sequence(r"$M_n=\int_0^1 z(1-z)A_n(z)\,dz$", np.arange(N), M_vals, r"$M_n$", show_titles, save_plots,
                  plot_format, f"plot_4_Mn_{F_name}", FIG, {"indices": odd_indices, "fmt": "{:.3f}"})

    # ln - x -> cn - x
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $c_n(x)-x$")
    for n in range(N):
        if np.isfinite(l_diag[n]):
            y = l_diag[n] - x_plot
            color = cmap(1.0 - n / norm_denom)
            plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6);
    plt.grid(True, alpha=0.3);
    plt.xlabel("x");
    plt.ylabel(r"$c_n(x)-x$");
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_cn_minus_x_{F_name}", plot_format)

    # Phi_n
    def compose_Qs(k):
        def f(u):
            y = np.asarray(u, dtype=float)
            for i in range(k, -1, -1):
                y = np.clip(Q_funcs[i](y), 0.0, 1.0)
            return y

        return f

    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $\Phi_n(x)$")
    ax_phi = plt.gca()
    cmap_phi = cm.viridis_r
    for k in range(N):
        y = compose_Qs(k)(x_plot)
        color = cmap_phi(k / norm_denom)
        ax_phi.plot(x_plot, y, lw=1.0, alpha=0.9, color=color)
    ax_phi.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6);
    ax_phi.grid(True, alpha=0.3);
    ax_phi.set_xlabel("x");
    ax_phi.set_ylabel(r"$\Phi_n(x)$")
    if save_plots: save_current_figure(f"plot_5_Phi_{F_name}", plot_format)

    # Phi diff
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $\Phi_{n+1}(x)-\Phi_n(x)$")
    for n in range(N - 1):
        y1 = compose_Qs(n)(x_plot);
        y2 = compose_Qs(n + 1)(x_plot)
        color = cmap(1.0 - n / max(1, N - 2))
        plt.plot(x_plot, y2 - y1, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6);
    plt.grid(True, alpha=0.3);
    plt.xlabel("x");
    plt.ylabel("Diff");
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_Phi_diff_{F_name}", plot_format)

    # ------------------------------------------------------------------
    # G_-(x) comparison plots: evolution of T_n(x)-G_-(x) and L_n(x)-G_-(x)
    # ------------------------------------------------------------------
    phi = (np.sqrt(5.0) + 1.0) / 2.0

    def G_minus(x):
        # Piecewise definition; x can be scalar.
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        w = 4.0 * x * (1.0 - x)
        w = np.clip(w, 0.0, 1.0)
        inside = 1.0 - np.power(w, phi)
        inside = max(0.0, inside)
        root = np.sqrt(inside)
        if x <= 0.5:
            return 0.5 * (1.0 - root)
        else:
            return 0.5 * (1.0 + root)

    # Evolution of T_n(x) - G_-(x)
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $T_n(x)-G_-(x)$")
    for n, Q in enumerate(Q_funcs):
        y = np.array([Q(x) - G_minus(x) for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6, linestyle='--')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel(r"$T_n(x)-G_-(x)$")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_last_Tn_minus_Gminus_{F_name}", plot_format)

    # Evolution of L_n(x) - G_-(x)
    plt.figure(figsize=FIG)
    if show_titles: plt.title(r"Evolution of $L_n(x)-G_-(x)$")
    for n, L in enumerate(L_funcs):
        y = np.array([L(x) - G_minus(x) for x in x_plot])
        color = cmap(1.0 - n / norm_denom)
        plt.plot(x_plot, y, lw=1.2, alpha=0.9, color=color, label=f"n={n}")
    plt.axhline(0.0, color='k', lw=1.0, alpha=0.6, linestyle='--')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel(r"$L_n(x)-G_-(x)$")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    if save_plots: save_current_figure(f"plot_last_Ln_minus_Gminus_{F_name}", plot_format)

    # Phi crossings
    Phi_cross_n, Phi_cross_x = [], []
    for n in range(N):
        roots = find_diag_crossings_generic(compose_Qs(n), grid_pts=GRID_PTS)
        roots_int = [r for r in roots if 1e-6 < r < 1.0 - 1e-6]
        if roots_int:
            idx = np.argmin(np.abs(np.array(roots_int) - 0.5))
            Phi_cross_n.append(float(n));
            Phi_cross_x.append(float(roots_int[idx]))
    odd_idx_cross = [i for i, v in enumerate(Phi_cross_n) if int(round(v)) % 2 == 1]
    plot_crossings(r"Evolution of $\phi_n$", Phi_cross_n, Phi_cross_x, r"$\phi_n$", show_titles, save_plots,
                   plot_format, f"plot_5b_Phi_diag_cross_{F_name}", FIG, {"indices": odd_idx_cross, "fmt": "{:.3f}"})
    plt.show()


def main():
    try:
        F_dist, F_name, F_params = get_initial_distribution()
        ans = input("\nShow plot titles? [Y/n]: ").strip().lower();
        show_titles = (ans != 'n')
        ans = input("Save plots? [y/N]: ").strip().lower();
        save_plots = (ans == 'y')
        plot_format = "png"
        if save_plots: plot_format = input("Format [png/pdf/svg] (default png): ").strip().lower() or "png"
        GRID_PTS = get_int_parameter("Grid points (odd, >=1001) [4001]: ", 4001, 1001, True)
        size_pct = 0.0
        try:
            size_pct = float(input("Size adj % [0]: ") or "0")
        except:
            pass
        N_iter = int(input("\nIterations N [10]: ") or "10")

        print("\nRunning simulation (Coupled Split Logic with Adaptive Grid)...")
        res = run_simulation(F_dist, F_name, F_params, N_iter, GRID_PTS)
        print("Generating plots...")
        plot_results(*res, F_name, show_titles, save_plots, plot_format, size_pct, GRID_PTS)
        print("\nDone.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()