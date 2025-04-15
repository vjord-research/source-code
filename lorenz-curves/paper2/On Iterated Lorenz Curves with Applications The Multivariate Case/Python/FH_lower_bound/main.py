import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def get_A_func(Q_func):
    """Creates the A(z) = Q(z)(1-Q(z)) function, handling vectorization and boundaries."""
    def A_n(z):
        z_arr = np.atleast_1d(z)
        result = np.zeros_like(z_arr, dtype=float)
        valid_mask = (z_arr > 1e-10) & (z_arr < 1 - 1e-10)
        if np.any(valid_mask):
            try:
                q_vals = Q_func(z_arr[valid_mask])
                q_vals = np.nan_to_num(q_vals, nan=0.0)
                q_vals = np.clip(q_vals, 0.0, 1.0)
                result[valid_mask] = q_vals * (1.0 - q_vals)
            except Exception as e:
                print(f"Warning: Error computing Q_func in A_n: {e}")
                pass
        if np.isscalar(z): return result.item()
        else:
            result[z_arr <= 1e-10] = 0.0
            result[z_arr >= 1 - 1e-10] = 0.0
            return result
    return A_n

def calculate_I_n(A_func):
    """Calculates I_n = integral of A_n(u) du with high precision."""
    try:
        I_val, I_err = integrate.quad(A_func, 0, 1, epsabs=1e-12, epsrel=1e-12, limit=500)
        print(f"  Calculated I_n: {I_val:.10f} (Error: {I_err:.2e})")
        if I_val <= 1e-12:
            print("  Warning: I_n is close to zero. Stopping iteration.")
            return None, None
        if I_err > 1e-6:
             print(f"  Warning: High error estimate for I_n ({I_err:.2e}). Result might be inaccurate.")
        return I_val, I_err
    except Exception as e:
        print(f"  FATAL Error calculating I_n: {e}")
        return None, None

def calculate_M_n(A_func):
    """Calculates M_n = integral of z(1-z)A_n(z) dz with high precision."""
    integrand = lambda z: z * (1.0 - z) * A_func(z) if 0 < z < 1 else 0.0
    try:
        M_val, M_err = integrate.quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12, limit=500)
        print(f"  Calculated M_n: {M_val:.10f} (Error: {M_err:.2e})")
        if M_err > 1e-6:
             print(f"  Warning: High error estimate for M_n ({M_err:.2e}). Result might be inaccurate.")
        return M_val, M_err
    except Exception as e:
         print(f"  FATAL Error calculating M_n: {e}")
         return None, None

def create_numerical_Q(A_func, I_val, z_grid):
    """
    Numerically calculates L_{n+1} on a grid and returns an interpolation
    function for Q_{n+1} = L_{n+1}^{-1}.
    """
    g_vals = np.zeros_like(z_grid, dtype=float)
    print("   Calculating g_n grid...", end="")
    for i, z in enumerate(z_grid):
        if z > 0:
            try:
                val, err = integrate.quad(A_func, 0, z, epsabs=1e-11, epsrel=1e-11, limit=200)
                g_vals[i] = max(0.0, val) # Ensure non-negative
            except Exception as e:
                print(f"\n   Warning: Error calculating g_n({z}): {e}. Setting g_n=NaN.")
                g_vals[i] = np.nan
        else: g_vals[i] = 0.0
    print(" Done.")

    L_vals = g_vals / I_val
    if np.any(np.isnan(L_vals)):
        print("   Warning: NaN found in L_vals. Attempting linear interpolation for NaN.")
        nan_mask = np.isnan(L_vals)
        L_vals[nan_mask] = np.interp(z_grid[nan_mask], z_grid[~nan_mask], L_vals[~nan_mask])
        if np.any(np.isnan(L_vals)):
            print("   FATAL Error: Cannot proceed with NaN values in L_vals after interpolation.")
            return None, None

    L_vals = np.maximum.accumulate(L_vals) # Force non-decreasing
    L_vals = np.clip(L_vals, 0.0, 1.0)    # Ensure range [0, 1]
    unique_L, unique_idx = np.unique(L_vals, return_index=True)
    unique_z_for_Q = z_grid[unique_idx]

    # Ensure (0,0) and (1,1) coverage
    if 0.0 not in unique_L:
        final_L_pts = np.insert(unique_L, 0, 0.0)
        final_z_pts_for_Q = np.insert(unique_z_for_Q, 0, 0.0)
    else:
        final_L_pts = unique_L
        final_z_pts_for_Q = unique_z_for_Q
        final_z_pts_for_Q[final_L_pts == 0.0] = 0.0 # Force z=0 for L=0

    if 1.0 not in final_L_pts:
         idx_to_insert = np.searchsorted(final_L_pts, 1.0)
         final_L_pts = np.insert(final_L_pts, idx_to_insert, 1.0)
         final_z_pts_for_Q = np.insert(final_z_pts_for_Q, idx_to_insert, 1.0)
    else:
        final_z_pts_for_Q[final_L_pts == 1.0] = 1.0 # Force z=1 for L=1

    try:
        Q_next_func = interp1d(final_L_pts, final_z_pts_for_Q, kind='cubic', bounds_error=False, fill_value=(0.0, 1.0))
        print(f"   Successfully created Q_{{n+1}} numerical interpolator.")
        L_next_func = interp1d(z_grid, L_vals, kind='cubic', bounds_error=False, fill_value=(0.0,1.0))
        return Q_next_func, L_next_func
    except ValueError as e:
         print(f"\n   FATAL Error creating interpolation function: {e}")
         print(f"   L points ({len(final_L_pts)}): {final_L_pts}")
         print(f"   Z points ({len(final_z_pts_for_Q)}): {final_z_pts_for_Q}")
         # Add check for monotonicity in inputs
         if not np.all(np.diff(final_L_pts) >= 0):
              print("   ERROR: L points are not monotonically increasing!")
         if not np.all(np.diff(final_z_pts_for_Q) >= 0):
              print("   ERROR: Z points are not monotonically increasing!")
         return None, None
    except Exception as e:
         print(f"\n   FATAL Error during interpolation: {e}")
         return None, None

def get_initial_distribution():
    """Gets user choice for the starting distribution F."""
    print("\nChoose the starting distribution F:")
    print("1: Uniform (Beta(1,1))")
    print("2: Beta(1,2)")
    print("3: Beta(2,1)")
    print("4: Beta(2,2)")
    print("5: Lognormal (sigma=0.5, offset/scaled to approx [0,1])")
    print("6: Pareto (b=3, offset/scaled to approx [0,1])")
    print("7: Gamma (a=2, offset/scaled to approx [0,1])")

    while True:
        try:
            choice = int(input("Enter choice (1-7): "))
            if 1 <= choice <= 7:
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    params = {}
    dist = None
    name = ""

    try:
        if choice == 1:
            params['alpha'] = 1.0
            params['beta'] = 1.0
            dist = stats.beta(params['alpha'], params['beta'])
            name = "Uniform (Beta(1,1))"
        elif choice == 2:
            params['alpha'] = 1.0
            params['beta'] = 2.0
            dist = stats.beta(params['alpha'], params['beta'])
            name = "Beta(1,2)"
        elif choice == 3:
            params['alpha'] = 2.0
            params['beta'] = 1.0
            dist = stats.beta(params['alpha'], params['beta'])
            name = "Beta(2,1)"
        elif choice == 4:
            params['alpha'] = 2.0
            params['beta'] = 2.0
            dist = stats.beta(params['alpha'], params['beta'])
            name = "Beta(2,2)"

        # --- Corrected definitions for custom distributions ---
        elif choice == 5:
            sigma = 0.8
            scale_param = 0.5
            raw_dist = stats.lognorm(s=sigma, scale=scale_param)
            xmax_approx = raw_dist.ppf(0.995)
            if xmax_approx <= 0: xmax_approx = raw_dist.ppf(0.999)

            # Add 'self' argument to methods defined for the custom class
            def cdf_trunc(self, x):
                x = np.clip(x, 0.0, 1.0)
                x_orig = x * xmax_approx
                cdf_at_max = raw_dist.cdf(xmax_approx)
                if cdf_at_max <= 0: return np.zeros_like(x)
                return raw_dist.cdf(x_orig) / cdf_at_max

            def ppf_trunc(self, q): # <--- Added 'self'
                q = np.clip(q, 0.0, 1.0)
                cdf_at_max = raw_dist.cdf(xmax_approx)
                if cdf_at_max <= 0: return np.zeros_like(q)
                q_orig = q * cdf_at_max
                try:
                     x_orig = raw_dist.ppf(q_orig)
                     x_orig = np.nan_to_num(x_orig, nan=0.0, posinf=xmax_approx, neginf=0.0)
                     x_orig = np.clip(x_orig, 0.0, xmax_approx)
                except Exception:
                    x_orig = np.array([raw_dist.ppf(qi) if 0<qi<1 else (xmax_approx if qi>=1 else 0.0) for qi in np.atleast_1d(q_orig)])
                    x_orig = np.nan_to_num(x_orig, nan=0.0, posinf=xmax_approx, neginf=0.0)
                    x_orig = np.clip(x_orig, 0.0, xmax_approx)
                return x_orig / xmax_approx if xmax_approx > 0 else 0.0

            dist = type('lognorm_trunc_scaled', (stats.rv_continuous,), {
                'cdf': cdf_trunc, 'ppf': ppf_trunc, '_stats': lambda self: (None, None, None, None),
                '_argcheck': lambda self, *args: True # Basic arg check
            })(a=0.0, b=1.0, name='lognorm_trunc')
            name = f"Lognormal (Trunc/Scaled, sigma={sigma})"

        elif choice == 6:
            b = 3
            raw_dist = stats.pareto(b)
            xmax_approx = raw_dist.ppf(0.995)
            C = xmax_approx - 1.0
            if C <= 0: C = raw_dist.ppf(0.999) - 1.0

            def cdf_trunc(self, x):
                x = np.clip(x, 0.0, 1.0)
                x_old = x * C + 1.0
                x_old = np.minimum(x_old, xmax_approx)
                cdf_at_max = raw_dist.cdf(xmax_approx)
                cdf_at_min = raw_dist.cdf(1.0)
                if cdf_at_max <= cdf_at_min : return np.zeros_like(x)
                val = (raw_dist.cdf(x_old) - cdf_at_min) / (cdf_at_max - cdf_at_min)
                return np.nan_to_num(val)

            def ppf_trunc(self, q): # <--- Added 'self'
                q = np.clip(q, 0.0, 1.0)
                cdf_at_max = raw_dist.cdf(xmax_approx)
                cdf_at_min = raw_dist.cdf(1.0)
                if cdf_at_max <= cdf_at_min : return np.zeros_like(q)
                q_unscaled = q * (cdf_at_max - cdf_at_min) + cdf_at_min
                x_old = np.zeros_like(np.atleast_1d(q_unscaled))
                mask0 = q_unscaled < (cdf_at_min + 1e-15)
                mask1 = q_unscaled > (cdf_at_max - 1e-15)
                mask_mid = (~mask0) & (~mask1)
                x_old[mask0] = 1.0
                x_old[mask1] = xmax_approx
                if np.any(mask_mid):
                    try: x_old[mask_mid] = raw_dist.ppf(q_unscaled[mask_mid])
                    except Exception: x_old[mask_mid] = [raw_dist.ppf(qi) for qi in q_unscaled[mask_mid]]
                x_old = np.nan_to_num(x_old, nan=1.0, posinf=xmax_approx, neginf=1.0)
                x_old = np.clip(x_old, 1.0, xmax_approx)
                return (x_old - 1.0) / C if C > 0 else 0.0

            dist = type('pareto_trunc_scaled', (stats.rv_continuous,), {
                 'cdf': cdf_trunc, 'ppf': ppf_trunc, '_stats': lambda self: (None, None, None, None),
                 '_argcheck': lambda self, *args: True
            })(a=0.0, b=1.0, name='pareto_trunc')
            name = f"Pareto (Trunc/Scaled, b={b})"

        elif choice == 7:
            a = 2
            raw_dist = stats.gamma(a)
            xmax_approx = raw_dist.ppf(0.995)
            if xmax_approx <= 0: xmax_approx = raw_dist.ppf(0.999)

            def cdf_trunc(self, x):
                x = np.clip(x, 0.0, 1.0)
                x_orig = x * xmax_approx
                cdf_at_max = raw_dist.cdf(xmax_approx)
                if cdf_at_max <= 0 : return np.zeros_like(x)
                return raw_dist.cdf(x_orig) / cdf_at_max

            def ppf_trunc(self, q): # <--- Added 'self'
                q = np.clip(q, 0.0, 1.0)
                cdf_at_max = raw_dist.cdf(xmax_approx)
                if cdf_at_max <= 0 : return np.zeros_like(q)
                q_orig = q * cdf_at_max
                try:
                    x_orig = raw_dist.ppf(q_orig)
                    x_orig = np.nan_to_num(x_orig, nan=0.0, posinf=xmax_approx, neginf=0.0)
                    x_orig = np.clip(x_orig, 0.0, xmax_approx)
                except Exception:
                    x_orig = np.array([raw_dist.ppf(qi) if 0<qi<1 else (xmax_approx if qi>=1 else 0.0) for qi in np.atleast_1d(q_orig)])
                    x_orig = np.nan_to_num(x_orig, nan=0.0, posinf=xmax_approx, neginf=0.0)
                    x_orig = np.clip(x_orig, 0.0, xmax_approx)
                return x_orig / xmax_approx if xmax_approx > 0 else 0.0

            dist = type('gamma_trunc_scaled', (stats.rv_continuous,), {
                'cdf': cdf_trunc, 'ppf': ppf_trunc, '_stats': lambda self: (None, None, None, None),
                 '_argcheck': lambda self, *args: True
            })(a=0.0, b=1.0, name='gamma_trunc')
            name = f"Gamma (Trunc/Scaled, a={a})"

        # Verification Test
        test_q = np.array([0.0, 0.5, 1.0])
        # Call ppf with the instance 'dist'
        test_x = dist.ppf(test_q)
        print(f"Initial Distribution: {name}")
        print(f"  PPF test: Q({test_q}) = {test_x}")
        if np.any(np.isnan(test_x)) or np.any(np.isinf(test_x)):
            print("\nError: Initial distribution PPF yields non-finite values.")
            return None, None, None
        print("-" * 20)

    except Exception as e:
        print(f"\nError creating initial distribution: {e}")
        traceback.print_exc() # Print full traceback
        return None, None, None

    return dist, name, params


def run_simulation(F_dist, F_name, F_params, N_iterations):
    """Runs the simulation and collects data, handling numerical iterations."""
    if F_dist is None: return [], [], np.array([]), np.array([])

    print(f"\nStarting simulation with F = {F_name}, N = {N_iterations}\n")

    # Store results
    L_n_funcs = [] # Stores CDF functions (analytical or numerical)
    A_n_funcs = [] # Stores A_n functions
    I_n_vals = []
    M_n_vals = []

    # Numerical grid for interpolation
    NUM_GRID_POINTS = 301
    z_grid = np.linspace(0, 1, NUM_GRID_POINTS)

    # --- Initial State (n=0 uses F = L_{-1}) ---
    current_params = F_params.copy()
    current_Q = F_dist.ppf      # This is Q_{-1} initially
    current_L = F_dist.cdf      # This is L_{-1} initially

    for n in range(N_iterations):
        print(f"--- Iteration n = {n} ---")

        # --- Stage 0: Determine if current L_n (represented by current_Q/L/params) is Beta(a,a) ---
        is_beta_alpha_alpha = False
        if 'alpha' in current_params and 'beta' in current_params and current_params['alpha'] == current_params['beta']:
            is_beta_alpha_alpha = True
            current_alpha = current_params['alpha'] # This alpha is for L_n
            print(f" L_{n} is identified as Beta({current_alpha:.1f},{current_alpha:.1f}). Using analytical path for I_n, M_n.")
            current_Q = stats.beta(current_alpha, current_alpha).ppf # Ensure correct Qn
            current_L = stats.beta(current_alpha, current_alpha).cdf # Ensure correct Ln
        else:
            is_beta_alpha_alpha = False
            dist_name = F_name if n==0 else f"L_{n} (Numerical)"
            print(f" {dist_name}. Using numerical path for I_n, M_n, and next step.")
            # current_Q/L should be set from previous iter or init

        # --- Stage 1: Calculate A_n, I_n, M_n based on Q_n=current_Q ---
        A_current = get_A_func(current_Q) # This is A_n
        A_n_funcs.append(A_current)
        L_n_funcs.append(current_L) # Store L_n for plotting

        I_n_val, _ = calculate_I_n(A_current)
        if I_n_val is None: break # Stop on error

        M_n_val, _ = calculate_M_n(A_current)
        if M_n_val is None: break # Stop on error

        I_n_vals.append(I_n_val)
        M_n_vals.append(M_n_val)

        # --- Stage 2: Calculate L_{n+1} and Q_{n+1} for the *next* iteration ---
        if is_beta_alpha_alpha:
            next_alpha = current_alpha + 1.0
            current_params = {'alpha': next_alpha, 'beta': next_alpha}
            current_Q = stats.beta(next_alpha, next_alpha).ppf
            current_L = stats.beta(next_alpha, next_alpha).cdf
            print(f"  Set up for n={n+1}: L_{n+1} will be Beta({next_alpha:.1f},{next_alpha:.1f})")
        else:
            print(f"  Numerically calculating L_{n+1} and Q_{n+1}...")
            Q_next, L_next = create_numerical_Q(A_current, I_n_val, z_grid)
            if Q_next is None or L_next is None:
                print(f"  Stopping iteration at n={n} due to Q_{n+1}/L_{n+1} numerical creation error.")
                break
            current_Q = Q_next
            current_L = L_next
            current_params = {} # Clear analytical params, now it's numerical
            print(f"  Set up for n={n+1}: Using numerical L_{n+1}, Q_{n+1}")

    # Return collected data, ensuring consistent lengths
    min_len = min(len(L_n_funcs), len(A_n_funcs), len(I_n_vals), len(M_n_vals))
    return L_n_funcs[:min_len], A_n_funcs[:min_len], np.array(I_n_vals[:min_len]), np.array(M_n_vals[:min_len])

# --- Plotting ---
def plot_results(L_funcs, A_funcs, I_n_vals, M_n_vals, F_name):
    """Generates the 4 plots based on simulation results."""
    N_computed = len(I_n_vals)
    if N_computed == 0:
        print("\nNo simulation data available to plot.")
        return

    print("\nGenerating plots...")
    iterations = np.arange(N_computed)
    x_plot = np.linspace(0.0, 1.0, 401) # Increased points slightly

    # Plot 1: L_n(x)
    plt.figure(figsize=(8, 6))
    plt.title(f"Plot 1: CDFs $L_n(x)$ (F={F_name})", fontsize=14)
    plot_success_count = 0
    for n in range(N_computed):
         if L_funcs[n] is not None and callable(L_funcs[n]):
             try:
                # Evaluate CDF, handle potential issues near 0/1 if function is sensitive
                y_plot = np.array([L_funcs[n](x_p) for x_p in x_plot])
                y_plot = np.nan_to_num(y_plot, nan=0.0) # Replace NaN if any occur
                y_plot = np.clip(y_plot, 0.0, 1.0)
                plt.plot(x_plot, y_plot, label=f'$L_{n}(x)$', alpha=0.8)
                plot_success_count += 1
             except Exception as e:
                 print(f" Could not plot L_{n}(x): {e}")
         else:
              print(f" L_{n} function not available for plotting (likely numerical result).")
    if plot_success_count > 0:
        plt.xlabel('$x$', fontsize=12)
        plt.ylabel('$L_n(x)$', fontsize=12)
        plt.legend(loc='best', fontsize='small') # Improve legend placement
        plt.grid(True)
        plt.ylim(-0.05, 1.05)
        plt.show()
    else:
        plt.close() # Close empty plot figure
        print(" Plot 1 skipped: No valid L_n functions available.")


    # Plot 2: I_n
    plt.figure(figsize=(8, 6))
    # Using raw string for title
    plt.title(rf"Plot 2: $I_n = \int A_n(u) du$ (F={F_name})", fontsize=14)
    plt.plot(iterations, I_n_vals, marker='o', linestyle='-')
    plt.axhline(1/4, color='r', linestyle='--', label='$y=1/4$', alpha=0.7)
    plt.axhline(1/6, color='g', linestyle='--', label='$y=1/6$', alpha=0.7)
    plt.xlabel('$n$', fontsize=12)
    plt.ylabel('$I_n$', fontsize=12)
    if N_computed > 0: plt.xticks(iterations if N_computed <=15 else iterations[::(N_computed//10 + 1)]) # Avoid overcrowded x-axis
    plt.legend()
    plt.grid(True)
    if N_computed > 0 : plt.ylim(bottom=min(0, min(I_n_vals)*0.9 if len(I_n_vals)>0 else 0), top = max(0.27, max(I_n_vals)*1.05 if len(I_n_vals)>0 else 0.27) ) # Adjust y-axis
    else: plt.ylim(0, 0.3)
    # Annotate points with precise values
    for i, txt in enumerate(I_n_vals):
        plt.annotate(f"{txt:.6f}", (iterations[i], I_n_vals[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)
    plt.tight_layout()
    plt.show()

    # Plot 3: A_n(z)
    plt.figure(figsize=(8, 6))
    plt.title(f"Plot 3: $A_n(z) = Q_n(z)(1-Q_n(z))$ (F={F_name})", fontsize=14)
    plot_success_count = 0
    max_a_val = 0
    for n in range(N_computed):
        if A_funcs[n] is not None and callable(A_funcs[n]):
            try:
                y_plot = A_funcs[n](x_plot)
                 # Ensure y_plot is array and handle potential numerical noise near 0/1
                if np.isscalar(y_plot): y_plot = np.full_like(x_plot, y_plot)
                y_plot = np.nan_to_num(y_plot, nan=0.0)
                y_plot = np.clip(y_plot, 0.0, 0.26) # Clip near theoretical max 0.25
                if len(y_plot) > 0 : max_a_val = max(max_a_val, np.max(y_plot)) # check length
                else: max_a_val = 0.25 # Default if empty plot data
                plt.plot(x_plot, y_plot, label=f'$A_{n}(z)$', alpha=0.8)
                plot_success_count += 1
            except Exception as e:
                print(f" Could not plot A_{n}(z): {e}")
        else:
             print(f" A_{n} function not available for plotting.")
    if plot_success_count > 0:
        plt.xlabel('$z$', fontsize=12)
        plt.ylabel('$A_n(z)$', fontsize=12)
        plt.ylim(bottom=-0.01, top=max(0.26, max_a_val * 1.05)) # Ensure y=0.25 is visible
        plt.axhline(1/4, color='r', linestyle='--', label='$y=1/4$ (Max)', alpha=0.7)
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.show()
    else:
        plt.close()
        print(" Plot 3 skipped: No valid A_n functions available.")


    # Plot 4: M_n
    plt.figure(figsize=(8, 6))
     # Using raw string for title
    plt.title(rf"Plot 4: $M_n = \int z(1-z) A_n(z) dz$ (F={F_name})", fontsize=14)
    plt.plot(iterations, M_n_vals, marker='s', linestyle=':')
    plt.xlabel('$n$', fontsize=12)
    plt.ylabel('$M_n$', fontsize=12)
    if N_computed > 0: plt.xticks(iterations if N_computed <=15 else iterations[::(N_computed//10 + 1)])
    plt.grid(True)
    if N_computed > 0: plt.ylim(bottom=min(0, min(M_n_vals)*0.9 if len(M_n_vals)>0 else 0), top=max(0.06, max(M_n_vals)*1.05 if len(M_n_vals)>0 else 0.06) )# Adjust y-axis
    else: plt.ylim(0, 0.1)
     # Annotate points with precise values
    for i, txt in enumerate(M_n_vals):
        plt.annotate(f"{txt:.6f}", (iterations[i], M_n_vals[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    try:
         F_dist, F_name, F_params = get_initial_distribution()

         if F_dist is None:
              print("Exiting due to distribution initialization error.")
         else:
             while True:
                try:
                    N_iter = int(input("Enter number of iterations (n=0 to N-1, e.g., 10): "))
                    if N_iter >= 1:
                        break
                    else:
                        print("Please enter at least 1 iteration.")
                except ValueError:
                    print("Invalid input.")

             L_funcs, A_funcs, I_n_vals, M_n_vals = run_simulation(F_dist, F_name, F_params, N_iter)

             plot_results(L_funcs, A_funcs, I_n_vals, M_n_vals, F_name)

    except KeyboardInterrupt:
         print("\nSimulation interrupted by user.")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()