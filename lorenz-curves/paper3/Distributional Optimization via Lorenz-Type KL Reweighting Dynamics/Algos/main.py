import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA
from sklearn.metrics import mutual_info_score

np.random.seed(42)


def calc_mi_hist(x, y, bins=20):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def get_mi(data):
    return calc_mi_hist(data[:, 0], data[:, 1])


def lorenz_map_inverse_sampling(x):
    n = len(x)
    idx = np.argsort(x)
    inv_idx = np.argsort(idx)
    x_pos = x[idx] - np.min(x) + 1e-7

    cum_sum = np.cumsum(x_pos)
    L_vals = cum_sum / cum_sum[-1]
    p_vals = np.linspace(1 / n, 1.0, n)

    L_vals = np.insert(L_vals, 0, 0.0)
    p_vals = np.insert(p_vals, 0, 0.0)

    # Linear interpolation to show the requested artifacts
    inv_L = interp1d(L_vals, p_vals, kind='linear', bounds_error=False, fill_value=(0, 1))

    ranks = np.linspace(1 / n, 1.0, n)
    x_new = inv_L(ranks)

    return x_new[inv_idx]


def run_ilm_approx(X, iterations=50):
    Z = X.copy()
    N, D = Z.shape
    for k in range(iterations):
        H = np.random.randn(D, D)
        Q, _ = np.linalg.qr(H)
        Z = np.dot(Z, Q)
        for d in range(D):
            Z[:, d] = lorenz_map_inverse_sampling(Z[:, d])
    return Z


def lorenz_operator_grid(density, N_grid=100):
    x = np.linspace(0, 1, N_grid)
    y = np.linspace(0, 1, N_grid)
    U, V = np.meshgrid(x, y)
    prod_field = U * V * density
    mu = np.sum(prod_field)

    numerator = np.cumsum(np.cumsum(prod_field, axis=0), axis=1)
    L_surface = numerator / mu

    new_density = np.diff(np.diff(L_surface, axis=0), axis=1)
    new_density = np.pad(new_density, ((1, 0), (1, 0)), 'constant')
    new_density = np.maximum(new_density, 0)

    if np.sum(new_density) > 0:
        new_density /= np.sum(new_density)

    return new_density


def sample_from_grid(density, n_samples=2000):
    N = density.shape[0]
    p = density.flatten()
    p /= np.sum(p)
    indices = np.random.choice(N * N, size=n_samples, p=p)
    y_idx, x_idx = np.unravel_index(indices, (N, N))
    x = (x_idx + np.random.rand(n_samples)) / N
    y = (y_idx + np.random.rand(n_samples)) / N
    return np.vstack([x, y]).T


if __name__ == "__main__":
    N = 3000
    x = np.random.uniform(0, 1, N)
    y = 4 * (x - 0.5) ** 2 + np.random.normal(0, 0.05, N)
    X_raw = np.vstack([x, y]).T

    X_c = X_raw - np.mean(X_raw, axis=0)
    U, S, V = np.linalg.svd(np.cov(X_c, rowvar=False))
    X_zca = np.dot(X_c, np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + 1e-5)), U.T)))

    ica = FastICA(n_components=2, random_state=42)
    X_ica = ica.fit_transform(X_raw)

    X_ilm = run_ilm_approx(X_raw, iterations=50)

    N_grid = 150
    gx = np.linspace(0, 1, N_grid)
    gy = np.linspace(0, 1, N_grid)
    GX, GY = np.meshgrid(gx, gy)

    density = np.exp(-((GY - 4 * (GX - 0.5) ** 2) ** 2) / 0.01)
    density /= np.sum(density)

    for _ in range(20):
        density = lorenz_operator_grid(density, N_grid)

    X_grid_samples = sample_from_grid(density, N)

    print("-" * 30)
    print("TABLE 1 RESULTS:")
    print(f"Raw:   {get_mi(X_raw):.3f}")
    print(f"ZCA:   {get_mi(X_zca):.3f}")
    print(f"ICA:   {get_mi(X_ica):.3f}")
    print(f"ILM:   {get_mi(X_ilm):.3f}")
    print(f"Grid:  {get_mi(X_grid_samples):.3f}")
    print("-" * 30)

    plt.figure(figsize=(5, 5))
    plt.scatter(X_raw[:, 0], X_raw[:, 1], s=1, alpha=0.5, color='blue')
    plt.title("Initial Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig("fig_initial.pdf")

    rank_ilm = stats.rankdata(X_ilm, axis=0) / N
    plt.figure(figsize=(5, 5))
    plt.scatter(rank_ilm[:, 0], rank_ilm[:, 1], s=1, alpha=0.5, color='red')
    plt.title("Method A: Copula")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("fig_method_a.pdf")

    rank_grid = stats.rankdata(X_grid_samples, axis=0) / N
    plt.figure(figsize=(5, 5))
    plt.scatter(rank_grid[:, 0], rank_grid[:, 1], s=1, alpha=0.5, color='green')
    plt.title("Method B: Copula")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("fig_method_b.pdf")