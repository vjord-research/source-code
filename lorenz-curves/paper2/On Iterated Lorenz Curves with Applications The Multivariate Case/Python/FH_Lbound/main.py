import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from scipy.interpolate import interp1d
import warnings
import traceback
import os

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
        if np.isscalar(z):
            return result.item()
        else:
            result[z_arr <= 1e-10] = 0.0
            result[z_arr >= 1 - 1e-10] = 0.0
            return result

    return A_n


def calculate_I_n(A_func):
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
                g_vals[i] = max(0.0, val)  # Ensure non-negative
            except Exception as e:
                print(f"\n   Warning: Error calculating g_n({z}): {e}. Setting g_n=NaN.")
                g_vals[i] = np.nan
        else:
            g_vals[i] = 0.0
    print(" Done.")

    L_vals = g_vals / I_val
    if np.any(np.isnan(L_vals)):
        print("   Warning: NaN found in L_vals. Attempting linear interpolation for NaN.")
        nan_mask = np.isnan(L_vals)
        L_vals[nan_mask] = np.interp(z_grid[nan_mask], z_grid[~nan_mask], L_vals[~nan_mask])
        if np.any(np.isnan(L_vals)):
            print("   FATAL Error: Cannot proceed with NaN values in L_vals after interpolation.")
            return None, None

    L_vals = np.maximum.accumulate(L_vals)  # Force non-decreasing
    L_vals = np.clip(L_vals, 0.0, 1.0)  # Ensure range [0, 1]
    unique_L, unique_idx = np.unique(L_vals, return_index=True)
    unique_z_for_Q = z_grid[unique_idx]

    # Ensure (0,0) and (1,1) coverage
    if 0.0 not in unique_L:
        final_L_pts = np.insert(unique_L, 0, 0.0)
        final_z_pts_for_Q = np.insert(unique_z_for_Q, 0, 0.0)
    else:
        final_L_pts = unique_L
        final_z_pts_for_Q = unique_z_for_Q
        final_z_pts_for_Q[final_L_pts == 0.0] = 0.0

    if 1.0 not in final_L_pts:
        idx_to_insert = np.searchsorted(final_L_pts, 1.0)
        final_L_pts = np.insert(final_L_pts, idx_to_insert, 1.0)
        final_z_pts_for_Q = np.insert(final_z_pts_for_Q, idx_to_insert, 1.0)
    else:
        final_z_pts_for_Q[final_L_pts == 1.0] = 1.0

    try:
        Q_next_func = interp1d(final_L_pts, final_z_pts_for_Q, kind='cubic', bounds_error=False, fill_value=(0.0, 1.0))
        print(f"   Successfully created Q_{{n+1}} numerical interpolator.")
        L_next_func = interp1d(z_grid, L_vals, kind='cubic', bounds_error=False, fill_value=(0.0, 1.0))
        return Q_next_func, L_next_func
    except ValueError as e:
        print(f"\n   FATAL Error creating interpolation function: {e}")
        if not np.all(np.diff(final_L_pts) >= 0): print("   ERROR: L points are not monotonically increasing!")
        if not np.all(np.diff(final_z_pts_for_Q) >= 0): print("   ERROR: Z points are not monotonically increasing!")
        return None, None
    except Exception as e:
        print(f"\n   FATAL Error during interpolation: {e}")
        return None, None


def get_float_parameter(prompt, condition=None, condition_desc="be valid"):
    """Prompts user for a float, validates it, and handles errors."""
    while True:
        try:
            value = float(input(prompt))
            if condition is None or condition(value):
                return value
            else:
                print(f"Error: Input must {condition_desc}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def get_initial_distribution():
    """Gets user choice for the distribution family and its parameters."""
    print("\nChoose the family for the starting distribution F (L_0):")
    print("1: Beta")
    print("2: Uniform (a special case of Beta)")
    print("3: Lognormal (will be truncated and scaled to [0,1])")
    print("4: Pareto (will be shifted, truncated, and scaled to [0,1])")
    print("5: Gamma (will be truncated and scaled to [0,1])")

    params = {}
    dist = None
    name = ""

    while True:
        try:
            choice = int(input("Enter choice (1-5): "))
            if 1 <= choice <= 5:
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    try:
        if choice == 1:  # Beta
            alpha = get_float_parameter("Enter alpha (shape > 0): ", lambda x: x > 0, "be positive")
            beta = get_float_parameter("Enter beta (shape > 0): ", lambda x: x > 0, "be positive")
            params = {'alpha': alpha, 'beta': beta}
            dist = stats.beta(alpha, beta)
            name = f"Beta(α={alpha}, β={beta})"

        elif choice == 2:  # Uniform
            params = {'alpha': 1.0, 'beta': 1.0}
            dist = stats.beta(1.0, 1.0)
            name = "Uniform (Beta(1,1))"

        elif choice == 3:  # Lognormal
            sigma = get_float_parameter("Enter sigma (shape > 0): ", lambda x: x > 0, "be positive")
            scale = get_float_parameter("Enter scale (> 0): ", lambda x: x > 0, "be positive")
            raw_dist = stats.lognorm(s=sigma, scale=scale)
            xmax_approx = raw_dist.ppf(0.995)
            if xmax_approx <= 0: xmax_approx = raw_dist.ppf(0.999)

            def cdf_trunc(self, x):
                x_orig = np.clip(x, 0, 1) * xmax_approx
                cdf_at_max = raw_dist.cdf(xmax_approx)
                return raw_dist.cdf(x_orig) / cdf_at_max if cdf_at_max > 0 else np.zeros_like(x)

            def ppf_trunc(self, q):
                cdf_at_max = raw_dist.cdf(xmax_approx)
                q_orig = np.clip(q, 0, 1) * cdf_at_max
                x_orig = raw_dist.ppf(q_orig)
                return x_orig / xmax_approx if xmax_approx > 0 else np.zeros_like(q)

            dist = type('lognorm_trunc_scaled', (stats.rv_continuous,), {'cdf': cdf_trunc, 'ppf': ppf_trunc})(a=0, b=1)
            name = f"Lognorm(σ={sigma}, scale={scale}, Trunc/Scaled)"
            params = {}

        elif choice == 4:  # Pareto
            b = get_float_parameter("Enter b (shape > 0): ", lambda x: x > 0, "be positive")
            raw_dist = stats.pareto(b)
            xmax_approx = raw_dist.ppf(0.995)
            C = xmax_approx - 1.0
            if C <= 0: C = raw_dist.ppf(0.999) - 1.0

            def cdf_trunc(self, x):
                x_old = np.clip(x, 0, 1) * C + 1.0
                cdf_at_max = raw_dist.cdf(xmax_approx)
                cdf_at_min = raw_dist.cdf(1.0)
                norm = cdf_at_max - cdf_at_min
                return (raw_dist.cdf(x_old) - cdf_at_min) / norm if norm > 0 else np.zeros_like(x)

            def ppf_trunc(self, q):
                cdf_at_max = raw_dist.cdf(xmax_approx)
                cdf_at_min = raw_dist.cdf(1.0)
                norm = cdf_at_max - cdf_at_min
                q_unscaled = np.clip(q, 0, 1) * norm + cdf_at_min
                x_old = raw_dist.ppf(q_unscaled)
                return (x_old - 1.0) / C if C > 0 else np.zeros_like(q)

            dist = type('pareto_trunc_scaled', (stats.rv_continuous,), {'cdf': cdf_trunc, 'ppf': ppf_trunc})(a=0, b=1)
            name = f"Pareto(b={b}, Trunc/Scaled)"
            params = {}

        elif choice == 5:  # Gamma
            a = get_float_parameter("Enter a (shape > 0): ", lambda x: x > 0, "be positive")
            raw_dist = stats.gamma(a)
            xmax_approx = raw_dist.ppf(0.995)
            if xmax_approx <= 0: xmax_approx = raw_dist.ppf(0.999)

            def cdf_trunc(self, x):
                x_orig = np.clip(x, 0, 1) * xmax_approx
                cdf_at_max = raw_dist.cdf(xmax_approx)
                return raw_dist.cdf(x_orig) / cdf_at_max if cdf_at_max > 0 else np.zeros_like(x)

            def ppf_trunc(self, q):
                cdf_at_max = raw_dist.cdf(xmax_approx)
                q_orig = np.clip(q, 0, 1) * cdf_at_max
                x_orig = raw_dist.ppf(q_orig)
                return x_orig / xmax_approx if xmax_approx > 0 else np.zeros_like(q)

            dist = type('gamma_trunc_scaled', (stats.rv_continuous,), {'cdf': cdf_trunc, 'ppf': ppf_trunc})(a=0, b=1)
            name = f"Gamma(a={a}, Trunc/Scaled)"
            params = {}

        # Verification Test
        test_q = np.array([0.0, 0.5, 1.0])
        test_x = dist.ppf(test_q)
        print(f"\nSuccessfully created initial distribution: {name}")
        print(f"  PPF test: Q({test_q}) = {test_x}")
        if np.any(np.isnan(test_x)) or np.any(np.isinf(test_x)):
            print("\nError: Initial distribution PPF yields non-finite values.")
            return None, None, None
        print("-" * 20)
    except Exception as e:
        print(f"\nError creating initial distribution: {e}")
        traceback.print_exc()
        return None, None, None

    return dist, name, params


def run_simulation(F_dist, F_name, F_params, N_iterations):
    """Runs the simulation and collects data, handling numerical iterations."""
    if F_dist is None: return [], [], [], np.array([]), np.array([])

    print(f"\nStarting simulation with F = {F_name} (as L_0), N = {N_iterations}\n")

    L_n_funcs, Q_n_funcs, A_n_funcs = [], [], []
    I_n_vals, M_n_vals = [], []
    z_grid = np.linspace(0, 1, 301)
    current_params = F_params.copy()
    current_Q, current_L = F_dist.ppf, F_dist.cdf

    for n in range(N_iterations):
        print(f"--- Iteration n = {n} ---")

        is_beta_alpha_alpha = False
        if 'alpha' in current_params and 'beta' in current_params and current_params['alpha'] == current_params['beta']:
            is_beta_alpha_alpha = True
            current_alpha = current_params['alpha']
            print(f" L_{n} is identified as Beta({current_alpha:.1f},{current_alpha:.1f}). Using analytical path.")
            current_Q, current_L = stats.beta(current_alpha, current_alpha).ppf, stats.beta(current_alpha,
                                                                                            current_alpha).cdf
        else:
            dist_name = F_name if n == 0 else f"L_{n} (Numerical)"
            print(f" L_{n} is {dist_name}. Using numerical path.")

        L_n_funcs.append(current_L)
        Q_n_funcs.append(current_Q)
        A_current = get_A_func(current_Q)
        A_n_funcs.append(A_current)

        I_n_val, _ = calculate_I_n(A_current)
        if I_n_val is None: break
        M_n_val, _ = calculate_M_n(A_current)
        if M_n_val is None: break
        I_n_vals.append(I_n_val)
        M_n_vals.append(M_n_val)

        if is_beta_alpha_alpha:
            next_alpha = current_alpha + 1.0
            current_params = {'alpha': next_alpha, 'beta': next_alpha}
            current_Q, current_L = stats.beta(next_alpha, next_alpha).ppf, stats.beta(next_alpha, next_alpha).cdf
            print(f"  Set up for n={n + 1}: L_{n + 1} will be Beta({next_alpha:.1f},{next_alpha:.1f})")
        else:
            print(f"  Numerically calculating L_{n + 1} and Q_{n + 1}...")
            Q_next, L_next = create_numerical_Q(A_current, I_n_val, z_grid)
            if Q_next is None or L_next is None:
                print(f"  Stopping iteration at n={n} due to numerical error.")
                break
            current_Q, current_L = Q_next, L_next
            current_params = {}

    min_len = min(len(L_n_funcs), len(Q_n_funcs), len(A_n_funcs), len(I_n_vals), len(M_n_vals))
    return L_n_funcs[:min_len], A_n_funcs[:min_len], Q_n_funcs[:min_len], np.array(I_n_vals[:min_len]), np.array(
        M_n_vals[:min_len])


def plot_results(L_funcs, A_funcs, Q_funcs, I_n_vals, M_n_vals, F_name, show_titles=True,
                 save_plots=False, plot_format=None, size_adjustment_percent=0):
    """Generates the 5 plots based on simulation results."""
    N_computed = len(I_n_vals)
    if N_computed == 0:
        print("\nNo simulation data available to plot.")
        return

    print("\nGenerating plots...")
    iterations = np.arange(N_computed)
    x_plot = np.linspace(0.0, 1.0, 401)

    # --- Size Adjustment Logic ---
    benchmark_figsize = (6.0, 4.5)
    scale_factor = 1.0 + (size_adjustment_percent / 100.0)
    FIGSIZE = (benchmark_figsize[0] * scale_factor, benchmark_figsize[1] * scale_factor)

    def save_current_figure(filename_base):
        if save_plots:
            output_dir = "plot_output"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename_base}.{plot_format}")

            save_kwargs = {'bbox_inches': 'tight'}
            if plot_format in ['png', 'jpg']:
                save_kwargs['dpi'] = 300

            plt.savefig(filepath, **save_kwargs)
            print(f"Plot saved to {filepath}")

    # Plot 1: L_n(x)
    plt.figure(figsize=FIGSIZE)
    if show_titles:
        plt.title(f"Plot 1: CDFs $L_n(x)$\n(Initial F: {F_name})", fontsize=14)
    plt.plot(x_plot, x_plot, 'k--', label='y=x', alpha=0.6)
    for n in range(N_computed):
        try:
            y_plot = L_funcs[n](x_plot)
            plt.plot(x_plot, np.clip(y_plot, 0, 1), label=f'n={n}', alpha=0.8)
        except Exception as e:
            print(f" Could not plot L_{n}(x): {e}")
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$L_n(x)$', fontsize=12)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True);
    plt.ylim(-0.05, 1.05);
    plt.tight_layout()
    save_current_figure('plot_1_L_n')
    plt.show()

    # Plot 2: I_n
    plt.figure(figsize=FIGSIZE)
    if show_titles:
        plt.title(rf"Plot 2: $I_n = \int A_n(u) du$ (F={F_name})", fontsize=14)
    plt.plot(iterations, I_n_vals, marker='o', linestyle='-')
    plt.axhline(1 / 4, color='r', linestyle='--', label='$y=1/4$', alpha=0.7)
    plt.axhline(1 / 6, color='g', linestyle='--', label='$y=1/6$', alpha=0.7)
    plt.xlabel('$n$', fontsize=12);
    plt.ylabel('$I_n$', fontsize=12)
    if N_computed > 0: plt.xticks(iterations if N_computed <= 15 else iterations[::(N_computed // 10 + 1)])
    plt.legend();
    plt.grid(True)
    if N_computed > 0: plt.ylim(bottom=min(0, min(I_n_vals) * 0.9 if I_n_vals.size > 0 else 0),
                                top=max(0.27, max(I_n_vals) * 1.05 if I_n_vals.size > 0 else 0.27))
    for i, txt in enumerate(I_n_vals): plt.annotate(f"{txt:.6f}", (i, I_n_vals[i]), textcoords="offset points",
                                                    xytext=(0, 5), ha='center', fontsize=9)
    plt.tight_layout()
    save_current_figure('plot_2_I_n')
    plt.show()

    # Plot 3: A_n(z)
    plt.figure(figsize=FIGSIZE)
    if show_titles:
        plt.title(f"Plot 3: $A_n(z) = Q_n(z)(1-Q_n(z))$ (F={F_name})", fontsize=14)
    max_a_val = 0
    for n in range(N_computed):
        try:
            y_plot = np.clip(A_funcs[n](x_plot), 0, 0.26)
            if y_plot.size > 0: max_a_val = max(max_a_val, np.max(y_plot))
            plt.plot(x_plot, y_plot, label=f'$A_{n}(z)$', alpha=0.8)
        except Exception as e:
            print(f" Could not plot A_{n}(z): {e}")
    plt.xlabel('$z$', fontsize=12);
    plt.ylabel('$A_n(z)$', fontsize=12)
    plt.ylim(bottom=-0.01, top=max(0.26, max_a_val * 1.05))
    plt.axhline(1 / 4, color='r', linestyle='--', label='$y=1/4$ (Max)', alpha=0.7)
    plt.legend(loc='best', fontsize='small');
    plt.grid(True);
    plt.tight_layout()
    save_current_figure('plot_3_A_n')
    plt.show()

    # Plot 4: M_n
    plt.figure(figsize=FIGSIZE)
    if show_titles:
        plt.title(rf"Plot 4: $M_n = \int z(1-z) A_n(z) dz$ (F={F_name})", fontsize=14)
    plt.plot(iterations, M_n_vals, marker='s', linestyle=':')
    plt.xlabel('$n$', fontsize=12);
    plt.ylabel('$M_n$', fontsize=12)
    if N_computed > 0: plt.xticks(iterations if N_computed <= 15 else iterations[::(N_computed // 10 + 1)])
    plt.grid(True)
    if N_computed > 0: plt.ylim(bottom=min(0, min(M_n_vals) * 0.9 if M_n_vals.size > 0 else 0),
                                top=max(0.06, max(M_n_vals) * 1.05 if M_n_vals.size > 0 else 0.06))
    for i, txt in enumerate(M_n_vals): plt.annotate(f"{txt:.6f}", (i, M_n_vals[i]), textcoords="offset points",
                                                    xytext=(0, 5), ha='center', fontsize=9)
    plt.tight_layout()
    save_current_figure('plot_4_M_n')
    plt.show()

    # Plot 5: Composite Function Phi_n(x)
    plt.figure(figsize=FIGSIZE)
    if show_titles:
        plt.title(r'Plot 5: Composite $\Phi_n(x) = (Q_0 \circ \dots \circ Q_n)(x)$' + f'\n(F={F_name})', fontsize=14)
    for n in range(N_computed):
        try:
            y_values = np.copy(x_plot)
            for i in range(n, -1, -1): y_values = Q_funcs[i](y_values)
            plt.plot(x_plot, np.clip(y_values, 0, 1), label=f'n={n}', alpha=0.8)
        except Exception as e:
            print(f" Could not plot Phi_{n}(x): {e}")
    plt.xlabel('$x$', fontsize=12);
    plt.ylabel(r'$\Phi_n(x)$', fontsize=12)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True);
    plt.ylim(-0.05, 1.05);
    plt.tight_layout()
    save_current_figure('plot_5_Phi_n')
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        F_dist, F_name, F_params = get_initial_distribution()

        if F_dist is None:
            print("Exiting due to distribution initialization error.")
        else:
            show_titles_input = input("Show titles on plots? (yes/no) [default: yes]: ").lower().strip()
            show_titles = not show_titles_input.startswith('n')

            # --- START: Added User Input for Plot Size and Saving ---
            size_options_map = {'-30%': -30, '-20%': -20, '-10%': -10, 'benchmark': 0, '+10%': 10, '+20%': 20,
                                '+30%': 30}
            size_prompt = f"Choose plot size from benchmark ({', '.join(size_options_map.keys())})"
            while True:
                user_size_choice = input(f"{size_prompt} [default: benchmark]: ").lower()
                if not user_size_choice:
                    user_size_choice = 'benchmark'
                if user_size_choice in size_options_map:
                    size_adjustment_percent = size_options_map[user_size_choice]
                    break
                else:
                    print("Invalid choice. Please select from the provided options.")

            save_plots_str = input("Do you want to save the plots? (yes/no) [default: yes]: ").lower().strip()
            save_plots = not save_plots_str.startswith('n')
            plot_format = None
            if save_plots:
                supported_formats = plt.figure().canvas.get_supported_filetypes()
                plt.close()  # Close the dummy figure
                recommended_formats = "Recommended: jpg, png, pdf, eps, svg"
                while True:
                    user_format = input(f"Enter plot format ({recommended_formats}) [default: jpg]: ").lower()
                    if not user_format:
                        user_format = 'jpg'
                    if user_format in supported_formats:
                        plot_format = user_format
                        break
                    else:
                        print(f"Error: Format '{user_format}' is not supported by your matplotlib backend.")
                        print(f"Supported formats are: {list(supported_formats.keys())}")
            # --- END: Added User Input ---

            N_iter = int(get_float_parameter(
                "Enter number of iterations (n=0 to N-1, e.g., 10): ",
                lambda x: x >= 1 and x == int(x),
                "be a positive integer"
            ))

            L_funcs, A_funcs, Q_funcs, I_n_vals, M_n_vals = run_simulation(F_dist, F_name, F_params, N_iter)

            # --- Pass new user choices to plotting function ---
            plot_results(L_funcs, A_funcs, Q_funcs, I_n_vals, M_n_vals, F_name,
                         show_titles=show_titles, save_plots=save_plots,
                         plot_format=plot_format, size_adjustment_percent=size_adjustment_percent)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()