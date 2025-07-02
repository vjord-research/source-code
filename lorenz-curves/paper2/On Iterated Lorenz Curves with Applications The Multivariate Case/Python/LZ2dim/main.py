import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import warnings
import time
import traceback
import math
import os

from scipy import integrate

# --- Global Settings ---
warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in power")
warnings.filterwarnings("ignore", category=UserWarning, module='scipy.interpolate')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Custom Distribution Classes ---
class SineWaveCDF(stats.rv_continuous):
    """
    Custom distribution CDF: F(x) = x + A * sin(k*pi*x) / (k*pi) on [0,1].
    k: integer >= 1 (oscillations), A: float in [-1,1] (amplitude).
    """

    def __init__(self, a=0, b=1, name='sinewave', k=3, A=0.5, xtol=1e-10, fine_grid_size=25000):
        super().__init__(a=a, b=b, name=name, xtol=xtol)
        if not (isinstance(k, int) and k >= 1):
            if isinstance(k, (float, np.floating)) and np.isclose(k, round(k)) and round(k) >= 1:
                k_int = round(k)
                warnings.warn(f"SineWaveCDF k={k} was float-like, using rounded integer k={k_int}.")
                self.num_oscillations_k = k_int
            else:
                raise ValueError(f"SineWaveCDF parameter 'k' must be an integer >= 1. Got {k}.")
        else:
            self.num_oscillations_k = k

        if not (-1.0 <= A <= 1.0):
            raise ValueError(f"SineWaveCDF parameter 'A' must be between -1.0 and 1.0. Got {A}.")
        if abs(A) > 0.99:
            warnings.warn(f"SineWaveCDF |A|={abs(A):.3f} is very close to 1. PPF stability might be affected.")

        self.amplitude_A = A
        self._k_pi = self.num_oscillations_k * np.pi
        self._fine_grid_size = max(fine_grid_size, 5000 * self.num_oscillations_k)
        self._x_fine_ppf = np.linspace(self.a, self.b, self._fine_grid_size)
        self._cdf_fine_ppf = self._cdf(self._x_fine_ppf)

        if np.any(np.diff(self._cdf_fine_ppf) < -1e-9):
            self._cdf_fine_ppf = np.maximum.accumulate(self._cdf_fine_ppf)
            self._cdf_fine_ppf = np.clip(self._cdf_fine_ppf, 0.0, 1.0)

        self._cdf_fine_ppf[0], self._cdf_fine_ppf[-1] = 0.0, 1.0

        unique_cdf_pts, unique_idx = np.unique(self._cdf_fine_ppf, return_index=True)

        if not np.isclose(unique_cdf_pts[0], 0.0):
            unique_cdf_pts = np.insert(unique_cdf_pts, 0, 0.0)
            unique_idx = np.insert(unique_idx, 0, 0)
        if not np.isclose(unique_cdf_pts[-1], 1.0):
            unique_cdf_pts = np.append(unique_cdf_pts, 1.0)
            unique_idx = np.append(unique_idx, self._fine_grid_size - 1)

        final_unique_cdf, final_idx_map = np.unique(unique_cdf_pts, return_index=True)
        final_x_for_ppf = self._x_fine_ppf[unique_idx[final_idx_map]]

        if len(final_unique_cdf) < 2:
            const_val_ppf = final_x_for_ppf[0] if len(final_x_for_ppf) > 0 else (self.a + self.b) / 2
            self._ppf_interpolator = lambda q_val: np.full_like(np.asarray(q_val), const_val_ppf)
            warnings.warn(
                "SineWaveCDF: CDF appears constant after processing for PPF. PPF will return a constant value.")
        else:
            self._ppf_interpolator = interpolate.interp1d(
                final_unique_cdf, final_x_for_ppf,
                kind='linear', bounds_error=False, fill_value=(self.a, self.b)
            )

    def _cdf(self, x):
        x_arr = np.clip(np.asarray(x), self.a, self.b)
        raw_cdf = 0.0
        if np.isclose(self._k_pi, 0):
            if self.amplitude_A != 0:
                raw_cdf = x_arr + self.amplitude_A * x_arr
            else:
                raw_cdf = x_arr
        else:
            raw_cdf = x_arr + self.amplitude_A * np.sin(self._k_pi * x_arr) / self._k_pi

        return np.clip(raw_cdf, 0.0, 1.0)

    def _pdf(self, x):
        x_arr = np.clip(np.asarray(x), self.a, self.b)
        pdf_values = 1.0 + self.amplitude_A * np.cos(self._k_pi * x_arr)
        return np.where((x_arr >= self.a) & (x_arr <= self.b), np.maximum(0.0, pdf_values), 0.0)

    def _ppf(self, q):
        q_arr = np.clip(np.asarray(q), 0.0, 1.0)
        return np.clip(self._ppf_interpolator(q_arr), self.a, self.b)


class BetaMixtureCDF(stats.rv_continuous):
    """Mixture of two Beta distributions on [0,1]."""

    def __init__(self, a=0, b=1, name='betamix', a1=2., b1=5., a2=5., b2=2., w=0.5, xtol=1e-10, fine_grid_size=10000):
        super().__init__(a=a, b=b, name=name, xtol=xtol)
        if not all(p > 0 for p in [a1, b1, a2, b2]):
            raise ValueError("BetaMixture params (a1,b1,a2,b2) must be >0.")
        if not (0 < w < 1):
            raise ValueError("BetaMixture weight 'w' must be in (0,1).")
        if min(a1, b1, a2, b2) < 0.1:
            warnings.warn("BetaMixture: A beta parameter <0.1 may cause numerical issues or extreme shapes.")

        self.beta1_dist = stats.beta(a=a1, b=b1)
        self.beta2_dist = stats.beta(a=a2, b=b2)
        self.weight1 = w
        self._fine_grid_size = fine_grid_size

        self._x_fine_ppf = np.linspace(self.a, self.b, self._fine_grid_size)
        self._cdf_fine_ppf = self._cdf(self._x_fine_ppf)

        self._cdf_fine_ppf = np.maximum.accumulate(self._cdf_fine_ppf)
        self._cdf_fine_ppf = np.clip(self._cdf_fine_ppf, 0., 1.)
        self._cdf_fine_ppf[0], self._cdf_fine_ppf[-1] = 0., 1.

        unique_cdf_pts, unique_idx = np.unique(self._cdf_fine_ppf, return_index=True)
        if not np.isclose(unique_cdf_pts[0], 0.):
            unique_cdf_pts = np.insert(unique_cdf_pts, 0, 0.)
            unique_idx = np.insert(unique_idx, 0, 0)
        if not np.isclose(unique_cdf_pts[-1], 1.):
            unique_cdf_pts = np.append(unique_cdf_pts, 1.)
            unique_idx = np.append(unique_idx, self._fine_grid_size - 1)

        if len(unique_cdf_pts) < 2:
            const_val_ppf = self._x_fine_ppf[unique_idx[0]] if len(unique_idx) > 0 else (self.a + self.b) / 2
            self._ppf_interpolator = lambda q_val: np.full_like(np.asarray(q_val), const_val_ppf)
            warnings.warn("BetaMixtureCDF: CDF appears constant, PPF will return a constant value.")
        else:
            try:
                self._ppf_interpolator = interpolate.interp1d(
                    unique_cdf_pts, self._x_fine_ppf[unique_idx],
                    kind='linear', bounds_error=False, fill_value=(self.a, self.b)
                )
            except ValueError as e_interp:
                warnings.warn(f"BetaMixtureCDF PPF interpolation error: {e_interp}. Defaulting to linear span.")
                self._ppf_interpolator = lambda q_val: np.asarray(q_val) * (self.b - self.a) + self.a

    def _cdf(self, x):
        x_arr = np.clip(np.asarray(x), self.a, self.b)
        return self.weight1 * self.beta1_dist.cdf(x_arr) + (1 - self.weight1) * self.beta2_dist.cdf(x_arr)

    def _pdf(self, x):
        x_arr = np.clip(np.asarray(x), self.a, self.b)
        pdf_val = self.weight1 * self.beta1_dist.pdf(x_arr) + (1 - self.weight1) * self.beta2_dist.pdf(x_arr)
        return np.where((x_arr >= self.a) & (x_arr <= self.b), np.maximum(0., pdf_val), 0.)

    def _ppf(self, q):
        q_arr = np.clip(np.asarray(q), 0., 1.)
        return np.clip(self._ppf_interpolator(q_arr), self.a, self.b)


# --- Marginal and Copula Factories ---
def get_marginal_distribution(name: str, params: dict):
    name = name.lower()
    ppf_clip_value = 1e-9

    if name == 'uniform':
        return stats.uniform(loc=0, scale=1)
    elif name == 'pareto':
        shape_b = params['b']
        if shape_b <= 0: raise ValueError("Pareto shape 'b' must be > 0.")

        class ParetoNonNegative(stats.rv_continuous):
            def _ppf(self, q, b_param):
                q_c = np.clip(q, ppf_clip_value, 1 - ppf_clip_value)
                return np.maximum(0., (1 - q_c) ** (-1 / b_param) - 1)

            def _cdf(self, x, b_param):
                x_c = np.maximum(x, 0.)
                return 1 - (1 + x_c) ** (-b_param)

            def _pdf(self, x, b_param):
                x_c = np.maximum(x, 0.)
                pdf_v = b_param * (1 + x_c) ** (-(b_param + 1))
                return np.where(x_c >= 0, pdf_v, 0)

        dist = ParetoNonNegative(a=0, name='pareto_nonneg', shapes='b_param')
        dist.b_param = shape_b
        dist.support = lambda: (0., np.inf)
        return dist
    elif name == 'lognormal':
        s = params['s']
        scale_param = params.get('scale', 1.)
        if s <= 0: raise ValueError("Lognormal shape 's' (sigma) must be > 0.")
        if scale_param <= 0: raise ValueError("Lognormal 'scale' must be > 0.")
        dist = stats.lognorm(s=s, scale=scale_param, loc=0)
        op, oc, od = dist.ppf, dist.cdf, dist.pdf
        dist.ppf = lambda q_val: np.where(q_val <= 0, 0, op(np.clip(q_val, ppf_clip_value, 1 - ppf_clip_value)))
        dist.cdf = lambda x_val: np.where(x_val < 0, 0, oc(x_val))
        dist.pdf = lambda x_val: np.where(x_val < 0, 0, od(x_val))
        dist.a = 0.
        dist.support = lambda: (0., np.inf)
        return dist
    elif name == 'gamma':
        a_shape = params['a']
        scale_param = params.get('scale', 1.)
        if a_shape <= 0 or scale_param <= 0: raise ValueError("Gamma 'a' (shape) and 'scale' must be > 0.")
        dist = stats.gamma(a=a_shape, scale=scale_param, loc=0)
        op, oc, od = dist.ppf, dist.cdf, dist.pdf
        dist.ppf = lambda q_val: np.where(q_val <= 0, 0, op(np.clip(q_val, ppf_clip_value, 1 - ppf_clip_value)))
        dist.cdf = lambda x_val: np.where(x_val < 0, 0, oc(x_val))
        dist.pdf = lambda x_val: np.where(x_val < 0, 0, od(x_val))
        dist.a = 0.
        dist.support = lambda: (0., np.inf)
        return dist
    elif name == 'beta':
        a, b = params['a'], params['b']
        if a <= 0 or b <= 0: raise ValueError("Beta params 'a','b' must be >0.")
        return stats.beta(a=a, b=b)
    elif name == 'sinewave':
        return SineWaveCDF(k=params.get('k', 3), A=params.get('A', 0.5))
    elif name == 'betamix':
        p = params
        return BetaMixtureCDF(a1=p['a1'], b1=p['b1'], a2=p['a2'], b2=p['b2'], w=p['w'])
    else:
        raise ValueError(f"Unknown marginal distribution: {name}")


def get_copula(name: str, param):
    name = name.lower()
    tiny = 1e-9

    if name == 'independent':
        return (lambda u, v: np.asarray(u) * np.asarray(v)), \
            (lambda u, v: np.ones_like(np.broadcast_to(u * v, np.broadcast(np.asarray(u), np.asarray(v)).shape)))
    elif name == 'gaussian':
        rho = param
        norm_rv = stats.norm()
        if not (-1 <= rho <= 1): raise ValueError("Gaussian copula rho must be in [-1,1].")
        if abs(rho - 1) < tiny: return get_copula('m', None)
        if abs(rho + 1) < tiny: return get_copula('w', None)
        try:
            mvn_dist = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], allow_singular=False)
        except ValueError:
            return get_copula('m' if rho > 0 else 'w', None)

        def cdf_gaussian(u, v):
            u_clip = np.clip(u, tiny, 1 - tiny)
            v_clip = np.clip(v, tiny, 1 - tiny)
            pts = np.stack([norm_rv.ppf(u_clip), norm_rv.ppf(v_clip)], axis=-1)
            return np.clip(mvn_dist.cdf(pts), 0, 1)

        def pdf_gaussian(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            mask_out_of_bounds = (u_arr <= 0) | (u_arr >= 1) | (v_arr <= 0) | (v_arr >= 1)
            u_clip = np.clip(u_arr, tiny, 1 - tiny)
            v_clip = np.clip(v_arr, tiny, 1 - tiny)
            xn, yn = norm_rv.ppf(u_clip), norm_rv.ppf(v_clip)
            if abs(1 - rho ** 2) < tiny ** 2:
                return np.where(mask_out_of_bounds, 0., np.inf)
            pdf_val = (1 / np.sqrt(1 - rho ** 2)) * np.exp(
                -(rho ** 2 * (xn ** 2 + yn ** 2) - 2 * rho * xn * yn) / (2 * (1 - rho ** 2)))
            return np.where(mask_out_of_bounds, 0., np.maximum(0, np.nan_to_num(pdf_val, nan=0, posinf=1e12)))

        return cdf_gaussian, pdf_gaussian
    elif name == 'clayton':
        theta = param
        if np.isclose(theta, 0): return get_copula('independent', None)
        if theta < -1: raise ValueError("Clayton theta must be >= -1.")
        if abs(theta - (-1)) < tiny: return get_copula('w', None)

        def cdf_clayton(u, v):
            u_clip = np.clip(u, tiny, 1)
            v_clip = np.clip(v, tiny, 1)
            val = u_clip * v_clip  # Fallback for invalid cases or theta=0 (already handled though)
            if theta > 0:  # Standard PQD range
                val_inside = u_clip ** (-theta) + v_clip ** (-theta) - 1
                val = np.maximum(val_inside, tiny) ** (-1 / theta)
            elif -1 <= theta < 0:  # NQD range
                val_inside = u_clip ** (-theta) + v_clip ** (-theta) - 1
                val = np.maximum(val_inside, 0.0) ** (-1.0 / theta)
            return np.clip(val, 0, 1)

        def pdf_clayton(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            mask_out_of_bounds = (u_arr <= 0) | (u_arr >= 1) | (v_arr <= 0) | (v_arr >= 1)
            u_clip = np.clip(u_arr, tiny, 1 - tiny)
            v_clip = np.clip(v_arr, tiny, 1 - tiny)

            base_val = u_clip ** (-theta) + v_clip ** (-theta) - 1
            default_problem_pdf = 1e12 if theta < 0 else 0.0

            term_uv_pow = (u_clip * v_clip) ** (-theta - 1)
            term_base_pow = np.maximum(base_val, tiny) ** (-2 - 1 / theta)

            pdf_calculated_val = (1 + theta) * term_uv_pow * term_base_pow
            pdf_res = np.where(base_val > tiny, pdf_calculated_val, default_problem_pdf)

            return np.where(mask_out_of_bounds, 0.0, np.maximum(0, np.nan_to_num(pdf_res, nan=0, posinf=1e12)))

        return cdf_clayton, pdf_clayton
    elif name == 'gumbel':
        th = param
        if th < 1: raise ValueError("Gumbel th >=1.")
        if abs(th - 1) < tiny: return get_copula('independent', None)

        def cdf_gu(u, v):
            uc, vc = np.clip(u, tiny, 1), np.clip(v, tiny, 1)
            lu, lv = -np.log(uc), -np.log(vc)
            val = np.exp(
                -((lu ** th + lv ** th) ** (1 / th)))
            return np.clip(val, 0, 1)

        def pdf_gu(u, v):
            ua, va = np.asarray(u), np.asarray(v)
            mo = (ua <= 0) | (ua >= 1) | (va <= 0) | (va >= 1)
            uc, vc = np.clip(ua, tiny, 1 - tiny), np.clip(va, tiny, 1 - tiny)
            Cuv = cdf_gu(uc, vc)
            lu, lv = -np.log(uc), -np.log(vc)
            slt = np.maximum(lu ** th + lv ** th, tiny)
            tie = slt ** (1 / th)
            prod_logs_pow = (lu * lv) ** (th - 1) if not np.isclose(th, 1) else 1.0
            pv = (Cuv / (uc * vc)) * (tie ** (1 - 2 * th)) * prod_logs_pow * ((th - 1) + tie)
            return np.where(mo, 0, np.maximum(0, np.nan_to_num(pv, nan=0, posinf=1e12)))

        return cdf_gu, pdf_gu
    elif name == 'frank':
        theta = param
        if abs(theta) < tiny: return get_copula('independent', None)
        if not np.isfinite(theta): raise ValueError("Frank th finite non-zero.")

        def cdf_f(u, v):
            u_c, v_c = np.clip(u, 0, 1), np.clip(v, 0, 1)
            if np.isclose(theta, 0): return u_c * v_c
            if theta > 35: return np.minimum(u_c, v_c)
            if theta < -35: return np.maximum(0, u_c + v_c - 1)

            exp_m_thu = np.expm1(-theta * u_c)
            exp_m_thv = np.expm1(-theta * v_c)
            exp_m_th = np.expm1(-theta)

            if abs(exp_m_th) < tiny ** 2:
                return np.minimum(u_c, v_c) if theta > 0 else np.maximum(0, u_c + v_c - 1)

            val_inside_log = 1 + (exp_m_thu * exp_m_thv) / exp_m_th
            val_inside_log_clipped = np.maximum(val_inside_log, tiny)
            return np.clip((-1 / theta) * np.log(val_inside_log_clipped), 0, 1)

        def pdf_f(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            mo = (u_arr <= tiny) | (u_arr >= 1 - tiny) | (v_arr <= tiny) | (v_arr >= 1 - tiny)
            uc, vc = np.clip(u_arr, tiny, 1 - tiny), np.clip(v_arr, tiny, 1 - tiny)
            if np.isclose(theta, 0): return np.where(mo, 0., 1.)

            one_minus_exp_neg_theta = -np.expm1(-theta)
            one_minus_exp_neg_theta_u = -np.expm1(-theta * uc)
            one_minus_exp_neg_theta_v = -np.expm1(-theta * vc)

            numerator_pdf = theta * one_minus_exp_neg_theta * np.exp(-theta * (uc + vc))
            denominator_base_pdf_sq = one_minus_exp_neg_theta - (one_minus_exp_neg_theta_u * one_minus_exp_neg_theta_v)

            pv = np.where(np.abs(denominator_base_pdf_sq) < tiny ** 3,
                          1e12 if numerator_pdf > 0 else 0.0,
                          numerator_pdf / (denominator_base_pdf_sq ** 2)
                          )
            return np.where(mo, 0., np.maximum(0, np.nan_to_num(pv, nan=0, posinf=1e12)))

        return cdf_f, pdf_f
    elif name == 't':
        if not isinstance(param, dict) or not {'rho', 'nu'}.issubset(param): raise ValueError(
            "t-copula needs dict {'rho','nu'}")
        rho, nu = param['rho'], param['nu']
        trv = stats.t(df=nu)
        if not (-1 <= rho <= 1): raise ValueError("t rho in [-1,1].")
        if nu <= 0: raise ValueError("t nu>0.")
        if abs(rho - 1) < tiny: return get_copula('m', None)
        if abs(rho + 1) < tiny: return get_copula('w', None)
        try:
            mvt = stats.multivariate_t(loc=[0, 0], shape=[[1, rho], [rho, 1]], df=nu, allow_singular=False)
        except ValueError:
            return get_copula('m' if rho > 0 else 'w', None)

        def cdf_t(u, v):
            uc, vc = np.clip(u, tiny, 1 - tiny), np.clip(v, tiny, 1 - tiny)
            pts = np.stack([trv.ppf(uc), trv.ppf(vc)], axis=-1)
            return np.clip(mvt.cdf(pts), 0, 1)

        def pdf_t(u, v):
            ua, va = np.asarray(u), np.asarray(v)
            mo = (ua <= 0) | (ua >= 1) | (va <= 0) | (va >= 1)
            uc, vc = np.clip(ua, tiny, 1 - tiny), np.clip(va, tiny, 1 - tiny)
            xt, yt = trv.ppf(uc), trv.ppf(vc)
            pbiv_t = mvt.pdf(np.stack([xt, yt], axis=-1))
            den_p = trv.pdf(xt) * trv.pdf(yt)
            pv = np.where(den_p > tiny ** 2, pbiv_t / den_p, 1e12)
            return np.where(mo, 0, np.maximum(0, np.nan_to_num(pv, nan=0, posinf=1e12)))

        return cdf_t, pdf_t

    elif name == 'amh':
        theta = param
        if not (-1 <= theta <= 1):
            raise ValueError("AMH copula theta must be in [-1,1]. Theta=0 is Independence.")
        if np.isclose(theta, 0): return get_copula('independent', None)

        def cdf_amh(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            u_c = np.clip(u_arr, 0, 1)
            v_c = np.clip(v_arr, 0, 1)
            res = np.zeros_like(u_arr, dtype=float)

            m_u0 = np.isclose(u_c, 0.0)
            m_v0 = np.isclose(v_c, 0.0)
            m_u1 = np.isclose(u_c, 1.0)
            m_v1 = np.isclose(v_c, 1.0)

            res[m_u0 | m_v0] = 0.0
            mask_u1_only = m_u1 & (~m_v0)
            if np.any(mask_u1_only): res[mask_u1_only] = v_c[mask_u1_only]
            mask_v1_only = m_v1 & (~m_u0) & (~m_u1)
            if np.any(mask_v1_only): res[mask_v1_only] = u_c[mask_v1_only]

            m_interior = (~m_u0) & (~m_v0) & (~m_u1) & (~m_v1)
            if np.any(m_interior):
                ui, vi = u_c[m_interior], v_c[m_interior]
                denominator_arr = 1.0 - theta * (1.0 - ui) * (1.0 - vi)

                is_close_to_zero_tol = np.isclose(denominator_arr, 0.0)
                replacement_val_when_super_small = np.where(
                    is_close_to_zero_tol,
                    tiny ** 3,
                    np.sign(denominator_arr) * (tiny ** 3)
                )
                effective_denominator = np.where(
                    np.abs(denominator_arr) < (tiny ** 3),
                    replacement_val_when_super_small,
                    denominator_arr
                )
                res[m_interior] = (ui * vi) / effective_denominator
            return np.clip(res, 0, 1)

        def pdf_amh(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            mask_out_of_bounds = (u_arr <= tiny) | (u_arr >= 1 - tiny) | \
                                 (v_arr <= tiny) | (v_arr >= 1 - tiny)
            u_c = np.clip(u_arr, tiny, 1 - tiny)
            v_c = np.clip(v_arr, tiny, 1 - tiny)

            num_term1 = theta * (theta - 1.0) * (1.0 - u_c) * (1.0 - v_c)
            num_term2 = theta * (u_c + v_c - 2.0 * u_c * v_c)
            num_term3 = 1.0 - theta
            numerator_pdf = num_term1 + num_term2 + num_term3
            denominator_base_pdf = 1.0 - theta * (1.0 - u_c) * (1.0 - v_c)

            pdf_val = np.where(np.abs(denominator_base_pdf) < tiny ** 2,
                               np.where(numerator_pdf * np.sign(denominator_base_pdf ** 3) >= 0, 1e12, 0.0),
                               numerator_pdf / (denominator_base_pdf ** 3)
                               )
            return np.where(mask_out_of_bounds, 0., np.maximum(0, np.nan_to_num(pdf_val, nan=0, posinf=1e12)))

        return cdf_amh, pdf_amh

    elif name == 'frechetmix':
        alpha = param
        if not (0 <= alpha <= 1):
            raise ValueError("FrechetMix alpha must be in [0,1].")
        cdf_M_func, _ = get_copula('m', None)
        cdf_W_func, _ = get_copula('w', None)

        def cdf_frechet_mix(u, v):
            u_c = np.clip(np.asarray(u), 0, 1)
            v_c = np.clip(np.asarray(v), 0, 1)
            val = alpha * cdf_M_func(u_c, v_c) + (1 - alpha) * cdf_W_func(u_c, v_c)
            return np.clip(val, 0, 1)

        def pdf_frechet_mix(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            shape = np.broadcast(u_arr, v_arr).shape
            if np.isclose(alpha, 1.0):
                _, pdf_m_func = get_copula('m', None)
                return pdf_m_func(u_arr, v_arr)
            elif np.isclose(alpha, 0.0):
                _, pdf_w_func = get_copula('w', None)
                return pdf_w_func(u_arr, v_arr)
            elif 0 < alpha < 1:
                return np.full(shape, np.inf)
            else:
                return np.zeros(shape)

        return cdf_frechet_mix, pdf_frechet_mix

    elif name == 'fgm':
        theta = param
        if not (-1 <= theta <= 1):
            raise ValueError("FGM copula theta must be in [-1,1].")

        def cdf_fgm(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            u_c = np.clip(u_arr, 0, 1)
            v_c = np.clip(v_arr, 0, 1)
            return u_c * v_c * (1 + theta * (1 - u_c) * (1 - v_c))

        def pdf_fgm(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            mask_out_of_bounds = (u_arr <= 0) | (u_arr >= 1) | (v_arr <= 0) | (v_arr >= 1)
            u_c = np.clip(u_arr, 0, 1)
            v_c = np.clip(v_arr, 0, 1)
            pdf_val = 1 + theta * (1 - 2 * u_c) * (1 - 2 * v_c)
            return np.where(mask_out_of_bounds, 0., np.maximum(0, pdf_val))

        return cdf_fgm, pdf_fgm

    elif name == 'm':
        return (lambda u, v: np.minimum(np.asarray(u), np.asarray(v))), \
            (lambda u, v: np.full(np.broadcast(np.asarray(u), np.asarray(v)).shape, np.inf))
    elif name == 'w':
        return (lambda u, v: np.maximum(0, np.asarray(u) + np.asarray(v) - 1)), \
            (lambda u, v: np.full(np.broadcast(np.asarray(u), np.asarray(v)).shape, np.inf))
    else:
        raise ValueError(f"Unknown copula name: {name}")


# --- Core Calculation Functions ---
def get_marginal_cdfs_and_quantiles(L_grid, Ng):
    grid_x = np.linspace(0, 1, Ng + 1)
    tiny = 1e-10
    L1_raw = L_grid[:, Ng].copy()
    L2_raw = L_grid[Ng, :].copy()
    L1 = np.maximum.accumulate(L1_raw)
    L2 = np.maximum.accumulate(L2_raw)
    L1 = np.clip(L1, 0, 1)
    L2 = np.clip(L2, 0, 1)
    L1[0] = 0
    L2[0] = 0
    L1[Ng] = 1
    L2[Ng] = 1

    def _create_q_interp(cdf_values, x_grid_values):
        unique_cdf_pts, unique_idx = np.unique(cdf_values, return_index=True)
        if not np.isclose(unique_cdf_pts[0], 0.0):
            unique_cdf_pts = np.insert(unique_cdf_pts, 0, 0.0)
            unique_idx = np.insert(unique_idx, 0, 0)
        if not np.isclose(unique_cdf_pts[-1], 1.0):
            unique_cdf_pts = np.append(unique_cdf_pts, 1.0)
            unique_idx = np.append(unique_idx, Ng)
        final_unique_cdf, final_unique_idx_map = np.unique(unique_cdf_pts, return_index=True)
        final_unique_x = x_grid_values[unique_idx[final_unique_idx_map]]
        if len(final_unique_cdf) < 2:
            const_x_val = final_unique_x[0] if len(final_unique_x) > 0 else 0.5
            return lambda p_val: np.full_like(np.asarray(p_val), const_x_val)
        qf_interpolator = interpolate.interp1d(final_unique_cdf, final_unique_x,
                                               kind='linear', bounds_error=False, fill_value=(0., 1.))
        return lambda p_val: np.clip(qf_interpolator(np.asarray(p_val)), 0., 1.)

    Q1_func = _create_q_interp(L1, grid_x)
    Q2_func = _create_q_interp(L2, grid_x)
    return L1, L2, Q1_func, Q2_func


def calculate_dependence_and_moments(L_grid, Ng):
    gX_bnd = np.linspace(0, 1, Ng + 1)
    h = 1 / Ng
    tiny = 1e-10
    rhoS_val, tauK_val, ratio_metric_val = np.nan, np.nan, np.nan
    E1_val, E2_val, E12_val_wn = 0.5, 0.5, np.nan
    try:
        pdf_L_cells = (L_grid[1:, 1:] - L_grid[:-1, 1:] - L_grid[1:, :-1] + L_grid[:-1, :-1]) / (h ** 2)
        pdf_L_cells = np.maximum(pdf_L_cells, 0)
        if not np.all(np.isfinite(pdf_L_cells)):
            max_finite_pdf = np.max(pdf_L_cells[np.isfinite(pdf_L_cells)]) if np.any(np.isfinite(pdf_L_cells)) else 1.0
            pdf_L_cells = np.nan_to_num(pdf_L_cells, nan=0, posinf=max_finite_pdf, neginf=0)
            pdf_L_cells = np.maximum(pdf_L_cells, 0)

        gX_mid = np.linspace(h / 2, 1 - h / 2, Ng)
        u1_mid_mesh, u2_mid_mesh = np.meshgrid(gX_mid, gX_mid, indexing='ij')
        cell_probs = pdf_L_cells * (h ** 2)
        total_prob = np.sum(cell_probs)
        if abs(total_prob - 1) > 1e-2 and total_prob > tiny:
            cell_probs /= total_prob
        elif total_prob <= tiny:
            return rhoS_val, tauK_val, ratio_metric_val, E1_val, E2_val, E12_val_wn

        E1_val = np.sum(u1_mid_mesh * cell_probs)
        E2_val = np.sum(u2_mid_mesh * cell_probs)
        E1_sq_val = np.sum(u1_mid_mesh ** 2 * cell_probs)
        E2_sq_val = np.sum(u2_mid_mesh ** 2 * cell_probs)
        E12_val_wn = np.sum(u1_mid_mesh * u2_mid_mesh * cell_probs)
        Var1_val = max(0, E1_sq_val - E1_val ** 2)
        Var2_val = max(0, E2_sq_val - E2_val ** 2)
        if Var1_val > tiny and Var2_val > tiny:
            rhoS_val = np.clip((E12_val_wn - E1_val * E2_val) / np.sqrt(Var1_val * Var2_val), -1, 1)
        else:
            rhoS_val = 0.
        L_interp_for_tau = interpolate.RectBivariateSpline(gX_bnd, gX_bnd, L_grid, kx=1, ky=1, s=0)
        L_at_midpoints = L_interp_for_tau.ev(u1_mid_mesh, u2_mid_mesh)
        tauK_val = np.clip(4 * np.sum(L_at_midpoints * cell_probs) - 1, -1, 1)
        if abs(E1_val) > tiny and abs(E2_val) > tiny: ratio_metric_val = E12_val_wn / (E1_val * E2_val)
    except Exception as e:
        warnings.warn(f"Error in calculate_dependence_and_moments for L_n: {e}")
    return rhoS_val, tauK_val, ratio_metric_val, E1_val, E2_val, E12_val_wn


def get_copula_from_L_distribution(L_grid, Ng, Q1_qfunc, Q2_qfunc):
    gX_cop_pts = np.linspace(0, 1, Ng + 1)
    Cn_grid_out = np.zeros_like(L_grid)
    tiny = 1e-10
    try:
        L_cdf_interpolator = interpolate.RectBivariateSpline(gX_cop_pts, gX_cop_pts, L_grid, kx=1, ky=1, s=0)
        for i_row, v1_cop_marg in enumerate(gX_cop_pts):
            for j_col, v2_cop_marg in enumerate(gX_cop_pts):
                u1_orig_scale = np.clip(Q1_qfunc(v1_cop_marg), 0, 1)
                u2_orig_scale = np.clip(Q2_qfunc(v2_cop_marg), 0, 1)
                Cn_grid_out[i_row, j_col] = L_cdf_interpolator(u1_orig_scale, u2_orig_scale, grid=False)
        Cn_grid_out = np.maximum.accumulate(np.maximum.accumulate(np.maximum(Cn_grid_out, 0.), axis=1), axis=0)
        norm_Cn_val = Cn_grid_out[Ng, Ng]
        if norm_Cn_val > tiny and abs(norm_Cn_val - 1) > 1e-3:
            Cn_grid_out /= norm_Cn_val
        elif norm_Cn_val <= tiny:
            warnings.warn(f"Extracted copula C_n(1,1)={norm_Cn_val:.2e} from L_n. Defaulting to Independence.")
            U_m, V_m = np.meshgrid(gX_cop_pts, gX_cop_pts, indexing='ij')
            Cn_grid_out = U_m * V_m

        Cn_grid_out = np.clip(Cn_grid_out, 0, 1)
        Cn_grid_out[Ng, :] = gX_cop_pts
        Cn_grid_out[:, Ng] = gX_cop_pts
        Cn_grid_out[0, :] = 0
        Cn_grid_out[:, 0] = 0
        Cn_grid_out[Ng, Ng] = 1.0
    except Exception as e_cop_extract:
        warnings.warn(f"Error extracting Copula C_n from L_n: {e_cop_extract}. Defaulting to Independence.")
        U_m, V_m = np.meshgrid(gX_cop_pts, gX_cop_pts, indexing='ij')
        Cn_grid_out = U_m * V_m
    return Cn_grid_out


def compute_L0_initial_distribution(F1_qfunc, F2_qfunc, C_init_pdf, Ng, num_integration_points_L0=None):
    gX_L0_eval = np.linspace(0, 1, Ng + 1)
    L0_cdf_grid = np.zeros((Ng + 1, Ng + 1))
    tiny = 1e-10
    Ng_integration = num_integration_points_L0 if num_integration_points_L0 is not None else max(Ng, 100)
    h_integration = 1.0 / Ng_integration
    gX_mid_integration = np.linspace(h_integration / 2, 1 - h_integration / 2, Ng_integration)
    v1_cop_mesh_int, v2_cop_mesh_int = np.meshgrid(gX_mid_integration, gX_mid_integration, indexing='ij')
    u1_F_scale_mesh = F1_qfunc(v1_cop_mesh_int)
    u2_F_scale_mesh = F2_qfunc(v2_cop_mesh_int)
    C_init_pdf_vals_int_mesh = C_init_pdf(v1_cop_mesh_int, v2_cop_mesh_int)
    u1_F_scale_mesh = np.nan_to_num(u1_F_scale_mesh, nan=0.0, posinf=1e10, neginf=-1e10)
    u2_F_scale_mesh = np.nan_to_num(u2_F_scale_mesh, nan=0.0, posinf=1e10, neginf=-1e10)
    C_init_pdf_vals_int_mesh = np.maximum(0, np.nan_to_num(C_init_pdf_vals_int_mesh, nan=0, posinf=1e10))
    L0_integrand_for_ZF = u1_F_scale_mesh * u2_F_scale_mesh * C_init_pdf_vals_int_mesh
    ZF_terms = L0_integrand_for_ZF * (h_integration ** 2)
    max_clip_val = np.finfo(float).max / (Ng_integration ** 2 + 1)
    ZF_val = np.sum(np.clip(ZF_terms, -max_clip_val, max_clip_val))
    if not np.isfinite(ZF_val) or abs(ZF_val) <= tiny:
        warnings.warn(f"ZF (denominator for L0) is {ZF_val:.2e}. Ill-defined. Defaulting L0 to Independence Copula.")
        Um, Vm = np.meshgrid(gX_L0_eval, gX_L0_eval, indexing='ij')
        return Um * Vm, np.nan
    L0_num_density_terms = F1_qfunc(v1_cop_mesh_int) * F2_qfunc(v2_cop_mesh_int) * C_init_pdf_vals_int_mesh
    L0_eff_density_unnorm = L0_num_density_terms / ZF_val
    L0_cdf_from_integration = np.cumsum(np.cumsum(L0_eff_density_unnorm * (h_integration ** 2), axis=0), axis=1)
    L0_cdf_padded_integration = np.pad(L0_cdf_from_integration, ((1, 0), (1, 0)), 'constant', constant_values=0.)
    if Ng_integration != Ng:
        gX_bnd_integration = np.linspace(0, 1, Ng_integration + 1)
        L0_interpolator = interpolate.RectBivariateSpline(gX_bnd_integration, gX_bnd_integration,
                                                          L0_cdf_padded_integration, kx=1, ky=1, s=0)
        x1_eval_mesh, x2_eval_mesh = np.meshgrid(gX_L0_eval, gX_L0_eval, indexing='ij')
        L0_cdf_grid = L0_interpolator(x1_eval_mesh, x2_eval_mesh, grid=False)
    else:
        L0_cdf_grid = L0_cdf_padded_integration
    L0_cdf_grid = np.maximum.accumulate(np.maximum.accumulate(np.maximum(L0_cdf_grid, 0.), axis=1), axis=0)
    norm_L0_at_11 = L0_cdf_grid[Ng, Ng]
    if norm_L0_at_11 > tiny and abs(norm_L0_at_11 - 1) > 1e-3:
        L0_cdf_grid /= norm_L0_at_11
    elif norm_L0_at_11 <= tiny:
        warnings.warn(
            f"L0(1,1) after computation is {norm_L0_at_11:.2e}. May indicate issues. Defaulting to Independence.")
        Um, Vm = np.meshgrid(gX_L0_eval, gX_L0_eval, indexing='ij')
        L0_cdf_grid = Um * Vm
    L0_cdf_grid = np.clip(L0_cdf_grid, 0, 1)
    L0_cdf_grid[0, :] = 0
    L0_cdf_grid[:, 0] = 0
    L0_cdf_grid[Ng, Ng] = 1.0
    if np.isnan(L0_cdf_grid).any():
        warnings.warn("NaNs detected in computed L0 grid. Defaulting L0 to Independence.")
        Um, Vm = np.meshgrid(gX_L0_eval, gX_L0_eval, indexing='ij')
        L0_cdf_grid = Um * Vm
    return L0_cdf_grid, ZF_val


def compute_Ln_plus_1_from_Ln(Ln_curr_cdf_grid, Ng, Q1n_quantile_func, Q2n_quantile_func, wn_prev_moment):
    gX_Ln_eval = np.linspace(0, 1, Ng + 1)
    h_grid = 1.0 / Ng
    L_next_cdf_grid = np.zeros((Ng + 1, Ng + 1))
    tiny = 1e-10
    if not np.isfinite(wn_prev_moment) or wn_prev_moment <= tiny:
        warnings.warn(f"Previous moment w_n={wn_prev_moment:.2e} is invalid. L_n+1 will default to Independence.")
        Um, Vm = np.meshgrid(gX_Ln_eval, gX_Ln_eval, indexing='ij')
        return Um * Vm
    pdf_Ln_cells = (Ln_curr_cdf_grid[1:, 1:] - Ln_curr_cdf_grid[:-1, 1:] - Ln_curr_cdf_grid[1:, :-1] + Ln_curr_cdf_grid[
                                                                                                       :-1, :-1]) / (
                           h_grid ** 2)
    pdf_Ln_cells = np.maximum(pdf_Ln_cells, 0)
    if not np.all(np.isfinite(pdf_Ln_cells)):
        max_finite_pdf_ln = np.max(pdf_Ln_cells[np.isfinite(pdf_Ln_cells)]) if np.any(
            np.isfinite(pdf_Ln_cells)) else 1.0
        pdf_Ln_cells = np.nan_to_num(pdf_Ln_cells, nan=0, posinf=max_finite_pdf_ln, neginf=0)
        pdf_Ln_cells = np.maximum(pdf_Ln_cells, 0)
    mid_points_integration = np.linspace(h_grid / 2, 1 - h_grid / 2, Ng)
    u1_mesh_mid_Ln, u2_mesh_mid_Ln = np.meshgrid(mid_points_integration, mid_points_integration, indexing='ij')
    Nn_integrand_vals_at_mid = u1_mesh_mid_Ln * u2_mesh_mid_Ln * pdf_Ln_cells
    cumulative_Nn_integral_grid = np.cumsum(np.cumsum(Nn_integrand_vals_at_mid * (h_grid ** 2), axis=0), axis=1)
    cumulative_Nn_integral_padded_grid = np.pad(cumulative_Nn_integral_grid, ((1, 0), (1, 0)), 'constant',
                                                constant_values=0.)
    Nn_cumulative_integral_interpolator = interpolate.RectBivariateSpline(gX_Ln_eval, gX_Ln_eval,
                                                                          cumulative_Nn_integral_padded_grid, kx=1,
                                                                          ky=1, s=0)
    for i_row, x1_eval_pt in enumerate(gX_Ln_eval):
        for j_col, x2_eval_pt in enumerate(gX_Ln_eval):
            s1_limit = np.clip(Q1n_quantile_func(x1_eval_pt), 0, 1)
            s2_limit = np.clip(Q2n_quantile_func(x2_eval_pt), 0, 1)
            num_L_next = Nn_cumulative_integral_interpolator(s1_limit, s2_limit, grid=False)
            L_next_cdf_grid[i_row, j_col] = num_L_next / wn_prev_moment
    L_next_cdf_grid = np.maximum.accumulate(np.maximum.accumulate(np.maximum(L_next_cdf_grid, 0.), axis=1), axis=0)
    norm_L_next_at_11 = L_next_cdf_grid[Ng, Ng]
    if norm_L_next_at_11 > tiny and abs(norm_L_next_at_11 - 1) > 1e-3:
        warnings.warn(f"L_n+1(1,1) before final clip was {norm_L_next_at_11:.4f}. Re-normalizing.")
        L_next_cdf_grid /= norm_L_next_at_11
    elif norm_L_next_at_11 <= tiny:
        warnings.warn(f"L_n+1(1,1) is {norm_L_next_at_11:.2e}. L_n+1 defaults to Independence.")
        Um, Vm = np.meshgrid(gX_Ln_eval, gX_Ln_eval, indexing='ij')
        L_next_cdf_grid = Um * Vm
    L_next_cdf_grid = np.clip(L_next_cdf_grid, 0, 1)
    L_next_cdf_grid[0, :] = 0
    L_next_cdf_grid[:, 0] = 0
    L_next_cdf_grid[Ng, Ng] = 1.0
    if np.isnan(L_next_cdf_grid).any():
        warnings.warn("NaNs detected in computed L_n+1 grid. Defaulting to Independence.")
        Um, Vm = np.meshgrid(gX_Ln_eval, gX_Ln_eval, indexing='ij')
        L_next_cdf_grid = Um * Vm
    return L_next_cdf_grid


def calculate_conditional_expectation_m2_given_u1(L_curr_cdf_grid, l1n_marg_pdf_on_grid, Ng):
    grid_pts_u1 = np.linspace(0, 1, Ng + 1)
    h_grid = 1.0 / Ng
    m2_given_u1_on_grid = np.full(Ng + 1, np.nan)
    tiny = 1e-10
    pdf_Ln_biv_cells = (L_curr_cdf_grid[1:, 1:] - L_curr_cdf_grid[:-1, 1:] - L_curr_cdf_grid[1:, :-1] + L_curr_cdf_grid[
                                                                                                        :-1, :-1]) / (
                               h_grid ** 2)
    pdf_Ln_biv_cells = np.maximum(pdf_Ln_biv_cells, 0)
    u1_mid_pts = np.linspace(h_grid / 2, 1 - h_grid / 2, Ng)
    u2_mid_pts = np.linspace(h_grid / 2, 1 - h_grid / 2, Ng)
    if Ng > 0:
        l1n_pdf_interp = interpolate.interp1d(grid_pts_u1, l1n_marg_pdf_on_grid, kind='linear',
                                              fill_value="extrapolate", bounds_error=False)
        l1n_pdf_at_u1_mid = np.maximum(tiny, l1n_pdf_interp(u1_mid_pts))
    else:
        m2_given_u1_on_grid.fill(0.5)
        return np.clip(np.nan_to_num(m2_given_u1_on_grid, nan=0.5), 0, 1)
    m2_given_u1_at_u1_mid = np.full(Ng, np.nan)
    for i_u1_mid in range(Ng):
        l1n_at_u1_val = l1n_pdf_at_u1_mid[i_u1_mid]
        if l1n_at_u1_val <= tiny:
            m2_given_u1_at_u1_mid[i_u1_mid] = 0.5
            continue
        num_integral_terms = u2_mid_pts * pdf_Ln_biv_cells[i_u1_mid, :]
        numerator_val = np.sum(num_integral_terms * h_grid)
        m2_given_u1_at_u1_mid[i_u1_mid] = numerator_val / l1n_at_u1_val
    if Ng > 1 and len(m2_given_u1_at_u1_mid[~np.isnan(m2_given_u1_at_u1_mid)]) > 1:
        valid_u1_mid = u1_mid_pts[~np.isnan(m2_given_u1_at_u1_mid)]
        valid_m2_vals = m2_given_u1_at_u1_mid[~np.isnan(m2_given_u1_at_u1_mid)]
        if len(valid_u1_mid) >= 2:
            m2_interp_func = interpolate.interp1d(valid_u1_mid, valid_m2_vals, kind='linear', fill_value="extrapolate",
                                                  bounds_error=False)
            m2_given_u1_on_grid = m2_interp_func(grid_pts_u1)
        elif len(valid_u1_mid) == 1:
            m2_given_u1_on_grid.fill(valid_m2_vals[0])
        else:
            m2_given_u1_on_grid.fill(0.5)
    elif Ng == 1 and len(m2_given_u1_at_u1_mid) > 0 and np.isfinite(m2_given_u1_at_u1_mid[0]):
        m2_given_u1_on_grid.fill(m2_given_u1_at_u1_mid[0])
    else:
        m2_given_u1_on_grid.fill(0.5)
    return np.clip(np.nan_to_num(m2_given_u1_on_grid, nan=0.5), 0, 1)


# --- Plotting ---
def get_plotting_x_axis(dist_obj, num_points=200):
    try:
        dist_name_attr = getattr(dist_obj, 'name', None) or \
                         (hasattr(dist_obj, 'dist') and getattr(dist_obj.dist, 'name', None))
        is_01_bounded = (hasattr(dist_obj, 'a') and hasattr(dist_obj, 'b') and
                         np.isclose(dist_obj.a, 0) and np.isclose(dist_obj.b, 1))
        if dist_name_attr in ['uniform', 'beta', 'sinewave', 'betamix'] or is_01_bounded:
            return np.linspace(0, 1, num_points)
        s_min, s_max = dist_obj.support()
        if not (hasattr(dist_obj, 'ppf') and callable(dist_obj.ppf)):
            return np.linspace(0, 1, num_points)
        low_ppf = 0.0
        if not np.isclose(s_min, 0) or np.isinf(s_min):
            low_ppf = dist_obj.ppf(1e-3)
            if not np.isfinite(low_ppf): low_ppf = -5.0 if np.isinf(s_min) and s_min < 0 else 0.0
        high_ppf = dist_obj.ppf(1.0 - 1e-3)
        if not np.isfinite(high_ppf): high_ppf = (low_ppf + 10) if np.isfinite(low_ppf) else 10.0
        if high_ppf <= low_ppf + 1e-6: high_ppf = low_ppf + 1.0
        plot_start = max(0, low_ppf) if s_min >= -1e-9 else low_ppf
        return np.linspace(plot_start, high_ppf, num_points)
    except Exception as e_plot_range:
        warnings.warn(f"Plotting range determination failed ({e_plot_range}). Defaulting to [0,1].")
        return np.linspace(0, 1, num_points)


def plot_simulation_results(sim_data, F1_init, F2_init, C_init_cdf, Ng, N_req, m_names_lbl, C_init_name,
                            show_titles, save_plots, plot_format, size_adjustment_percent):
    gX_unit = np.linspace(0, 1, Ng + 1)
    N_run = len(sim_data['L_n_grids_history'])
    if N_run == 0:
        print("No data available for plotting.")
        return

    # --- Plotting Configuration ---
    p_colors = plt.cm.viridis(np.linspace(0, 1, max(1, N_run)))
    leg_fs = 'small' if N_run <= 10 else 'xx-small'
    leg_nc = 1 if N_run <= 5 else (2 if N_run <= 20 else 3)

    # --- Size Adjustment Logic ---
    benchmark_single_figsize = (5.95, 4.675)

    # Apply user-selected scaling
    scale_factor = 1.0 + (size_adjustment_percent / 100.0)
    final_single_figsize = (benchmark_single_figsize[0] * scale_factor, benchmark_single_figsize[1] * scale_factor)

    if save_plots:
        plot_dir = "simulation_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        print(f"\n--- Plotting Results (saving to '{plot_dir}/' as .{plot_format}) ---")
    else:
        print("\n--- Plotting Results (displaying only) ---")

    x_F1_plot = get_plotting_x_axis(F1_init)
    x_F2_plot = get_plotting_x_axis(F2_init)
    itr_plot_idx = np.arange(N_run)

    # Helper function for saving
    def save_current_figure(filename_base):
        if save_plots:
            filepath = os.path.join(plot_dir, f"{filename_base}.{plot_format}")
            save_kwargs = {}
            # Set high DPI for raster formats
            if plot_format in ['png', 'jpg', 'jpeg', 'tiff']:
                save_kwargs['dpi'] = 600
            plt.savefig(filepath, **save_kwargs)

    # --- Plot 1: Initial Distributions (Combined) ---
    fig_initial, axs_initial = plt.subplots(2, 2, figsize=final_single_figsize)
    if show_titles:
        fig_initial.suptitle(r"Plot 1: Initial Marginals (d.f.s and p.d.f.s)", fontsize=14)

    axs_initial[0, 0].plot(x_F1_plot, F1_init.cdf(x_F1_plot), label=f"$F_1$: {m_names_lbl.split('/')[0]}")
    axs_initial[0, 0].plot([0, 1], [0, 1], 'k:', alpha=0.7, label='y=x')
    axs_initial[0, 0].legend(fontsize='x-small')
    axs_initial[0, 0].grid(True)
    axs_initial[0, 0].set_title(r"$F_1$", fontsize='small')
    axs_initial[0, 0].set_xlabel('$x_1$', fontsize='small')
    axs_initial[0, 0].set_ylabel('$F_1(x_1)$', fontsize='small')
    axs_initial[0, 0].tick_params(axis='both', which='major', labelsize='x-small')

    axs_initial[0, 1].plot(x_F2_plot, F2_init.cdf(x_F2_plot), label=f"$F_2$: {m_names_lbl.split('/')[1]}", c='r')
    axs_initial[0, 1].plot([0, 1], [0, 1], 'k:', alpha=0.7, label='y=x')
    axs_initial[0, 1].legend(fontsize='x-small')
    axs_initial[0, 1].grid(True)
    axs_initial[0, 1].set_title(r"$F_2$", fontsize='small')
    axs_initial[0, 1].set_xlabel('$x_2$', fontsize='small')
    axs_initial[0, 1].set_ylabel('$F_2(x_2)$', fontsize='small')
    axs_initial[0, 1].tick_params(axis='both', which='major', labelsize='x-small')

    axs_initial[1, 0].plot(x_F1_plot, F1_init.pdf(x_F1_plot), label="$f_1$")
    axs_initial[1, 0].legend(fontsize='x-small')
    axs_initial[1, 0].grid(True)
    axs_initial[1, 0].set_ylim(bottom=0)
    axs_initial[1, 0].set_title(r"$f_1$", fontsize='small')
    axs_initial[1, 0].set_xlabel('$x_1$', fontsize='small')
    axs_initial[1, 0].set_ylabel('$f_1(x_1)$', fontsize='small')
    axs_initial[1, 0].tick_params(axis='both', which='major', labelsize='x-small')

    axs_initial[1, 1].plot(x_F2_plot, F2_init.pdf(x_F2_plot), label="$f_2$", c='r')
    axs_initial[1, 1].legend(fontsize='x-small')
    axs_initial[1, 1].grid(True)
    axs_initial[1, 1].set_ylim(bottom=0)
    axs_initial[1, 1].set_title(r"$f_2$", fontsize='small')
    axs_initial[1, 1].set_xlabel('$x_2$', fontsize='small')
    axs_initial[1, 1].set_ylabel('$f_2(x_2)$', fontsize='small')
    axs_initial[1, 1].tick_params(axis='both', which='major', labelsize='x-small')

    fig_initial.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_01_Initial_Distributions')
    plt.show()

    # --- Plot 2: L1n CDF Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 2: Marginal CDF Evolution $L_1^{n}$", fontsize=14)
    for i in range(N_run):
        plt.plot(gX_unit, sim_data['marg_cdfs_L1n_hist'][i], c=p_colors[i], label=f'n={i}', alpha=0.7)
    plt.plot(gX_unit, gX_unit, 'k:', alpha=0.7)
    plt.legend(fontsize=leg_fs, ncol=leg_nc)
    plt.grid(True)
    plt.xlabel('$x_1$')
    plt.ylabel('$L_1^{n}(x_1)$')
    plt.tight_layout()
    save_current_figure('plot_02_L1n_CDF')
    plt.show()

    # --- Plot 3: L2n CDF Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 3: Marginal CDF Evolution $L_2^{n}$", fontsize=14)
    for i in range(N_run):
        plt.plot(gX_unit, sim_data['marg_cdfs_L2n_hist'][i], c=p_colors[i], label=f'n={i}', alpha=0.7)
    plt.plot(gX_unit, gX_unit, 'k:', alpha=0.7)
    plt.legend(fontsize=leg_fs, ncol=leg_nc)
    plt.grid(True)
    plt.xlabel('$x_2$')
    plt.ylabel('$L_2^{n}(x_2)$')
    plt.tight_layout()
    save_current_figure('plot_03_L2n_CDF')
    plt.show()

    # --- Plot 4: l1n PDF Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 4: Marginal PDF Evolution $l_1^{n}$", fontsize=14)
    for i in range(N_run):
        plt.plot(gX_unit, sim_data['marg_pdfs_l1n_hist'][i], c=p_colors[i], label=f'n={i}', alpha=0.7)
    plt.legend(fontsize=leg_fs, ncol=leg_nc)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlabel('$x_1$')
    plt.ylabel('$l_1^{n}(x_1)$')
    plt.tight_layout()
    save_current_figure('plot_04_l1n_PDF')
    plt.show()

    # --- Plot 5: l2n PDF Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 5: Marginal PDF Evolution $l_2^{n}$", fontsize=14)
    for i in range(N_run):
        plt.plot(gX_unit, sim_data['marg_pdfs_l2n_hist'][i], c=p_colors[i], label=f'n={i}', alpha=0.7)
    plt.legend(fontsize=leg_fs, ncol=leg_nc)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlabel('$x_2$')
    plt.ylabel('$l_2^{n}(x_2)$')
    plt.tight_layout()
    save_current_figure('plot_05_l2n_PDF')
    plt.show()

    # --- Plot 6: mu_n Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 6: Evolution of $\mu_n=E[X_1^{L_n}X_2^{L_n}]$", fontsize=14)
    plt.plot(itr_plot_idx, sim_data['wn_moments_hist'], marker='o')
    plt.xlabel(r"n")
    plt.ylabel(r"$\mu_n$")
    plt.grid(True)
    plt.xticks(itr_plot_idx)
    plt.tight_layout()
    save_current_figure('plot_06_mu_n_evolution')
    plt.show()

    # --- Plot 7: m_2|1(u1) Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 7: Evolution of $m_{2|1}^{n}(x_1)=E[X_2^{L_n}|X_1^{L_n}=x_1]$", fontsize=14)
    m21_data_key = 'm21_given_u1_funcs_hist'
    for i in range(N_run):
        if i < len(sim_data[m21_data_key]):
            plt.plot(gX_unit, sim_data[m21_data_key][i], c=p_colors[i], label=f'n={i}', alpha=0.7)
    plt.ylim(0, 1)
    plt.legend(fontsize=leg_fs, ncol=leg_nc)
    plt.grid(True)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$m_{2|1}^{n}(x_1)$')
    plt.tight_layout()
    save_current_figure('plot_07_m21_evolution')
    plt.show()

    # --- Plot 8: h_n(u1) Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 8: Evolution of $h_n(x_1)=x_1 m_{2|1}^{n}(x_1)$", fontsize=14)
    for i in range(N_run):
        if i < len(sim_data[m21_data_key]):
            m21_current_n = sim_data[m21_data_key][i]
            plt.plot(gX_unit, gX_unit * m21_current_n, c=p_colors[i], label=f'n={i}', alpha=0.7)
    plt.ylim(0, 1)
    plt.legend(fontsize=leg_fs, ncol=leg_nc)
    plt.grid(True)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$h_n(x_1)$')
    plt.tight_layout()
    save_current_figure('plot_08_h_n_evolution')
    plt.show()

    # --- Plot 9: Dependence Measures ---
    UM, VM = np.meshgrid(gX_unit, gX_unit, indexing='ij')
    Cinit_grid = C_init_cdf(UM, VM)
    rhoF_init, tauF_init, _, _, _, _ = calculate_dependence_and_moments(Cinit_grid, Ng)
    iter_dep_indices = [-1] + list(itr_plot_idx)
    rho_plot_vals = [rhoF_init] + sim_data['rho_S_hist']
    tau_plot_vals = [tauF_init] + sim_data['tau_K_hist']
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 9: Dependence Measures ($\rho_S, \tau_K$)", fontsize=14)
    plt.plot(iter_dep_indices, rho_plot_vals, marker='o', label=r'Spearman $\rho_S$')
    plt.plot(iter_dep_indices, tau_plot_vals, marker='s', label=r'Kendall $\tau_K$')
    plt.text(0.5, 0.95, "(n=-1 for initial F's copula, n>=0 for $L_n$)",
             ha='center', transform=plt.gca().transAxes, fontsize='small')
    plt.xlabel("n")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.xticks(iter_dep_indices)
    plt.tight_layout()
    save_current_figure('plot_09_dependence_evolution')
    plt.show()

    # --- Plot 10: Ratio Metric ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 10: $\mathbb{E}[X_1^{L_n}X_2^{L_n}]/(\mathbb{E}[X_1^{L_n}]\mathbb{E}[X_2^{L_n}])$",
                  fontsize=14)
    plt.plot(itr_plot_idx, sim_data['ratio_metric_hist'], marker='o')
    plt.xlabel(r"n")
    plt.ylabel("Ratio")
    plt.grid(True)
    plt.xticks(itr_plot_idx)
    plt.tight_layout()
    save_current_figure('plot_10_ratio_metric_evolution')
    plt.show()

    # --- Plot 11: Copula Diagonals ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 11: Copula Diagonals $C_n(x,x)$", fontsize=14)
    plt.plot(gX_unit, np.diag(Cinit_grid), label=f'Initial C ({C_init_name})', linestyle=':')
    diag_plot_indices = [0]
    if N_run > 1: diag_plot_indices.append(min(1, N_run - 1))
    if N_req < N_run - 1 and N_req > (diag_plot_indices[-1] if diag_plot_indices else -1): diag_plot_indices.append(
        N_req)
    if N_run - 1 not in diag_plot_indices and N_run - 1 >= 0: diag_plot_indices.append(N_run - 1)
    diag_plot_indices = sorted(list(set(idx for idx in diag_plot_indices if idx >= 0 and idx < N_run)))
    for i_diag in diag_plot_indices:
        plt.plot(gX_unit, np.diag(sim_data['Cn_copula_grids_hist'][i_diag]),
                 c=p_colors[i_diag], label=f'$C_{{{i_diag}}}$ (from $L_{{{i_diag}}}$)', alpha=0.8)
    plt.plot(gX_unit, gX_unit, 'k--', lw=0.8, label='M(x,x)')
    plt.plot(gX_unit, np.maximum(0, 2 * gX_unit - 1), 'k-.', lw=0.8, label='W(x,x)')
    plt.xlabel("x")
    plt.ylabel(r"$C_n(x,x)$")
    plt.grid(True)
    plt.legend(fontsize='small')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    save_current_figure('plot_11_copula_diagonals')
    plt.show()

    # --- Plot 12: Phi_n^1 Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 12: Evolution of Composed Quantile Function $\Phi_n^1(x)$", fontsize=14)
    for n_phi_iter in range(N_run):
        phi_n1_current_vals = gX_unit.copy()
        for k_idx_for_L_inv in range(n_phi_iter, -1, -1):
            if k_idx_for_L_inv < len(sim_data['quant_funcs_Q1n_hist']):
                Q1_k_inv = sim_data['quant_funcs_Q1n_hist'][k_idx_for_L_inv]
                phi_n1_current_vals = Q1_k_inv(phi_n1_current_vals)
        plt.plot(gX_unit, phi_n1_current_vals, color=p_colors[n_phi_iter], label=f'n={n_phi_iter}', alpha=0.7)
    plt.plot(gX_unit, gX_unit, 'k:', label='Identity y=x', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel(r"$\Phi_n^1(x)$")
    plt.legend(fontsize=leg_fs, ncol=leg_nc)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    save_current_figure('plot_12_phi1_evolution')
    plt.show()

    # --- Plot 13: Phi_n^2 Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles:
        plt.title(r"Plot 13: Evolution of Composed Quantile Function $\Phi_n^2(x)$", fontsize=14)
    for n_phi_iter in range(N_run):
        phi_n2_current_vals = gX_unit.copy()
        for k_idx_for_L_inv in range(n_phi_iter, -1, -1):
            if k_idx_for_L_inv < len(sim_data['quant_funcs_Q2n_hist']):
                Q2_k_inv = sim_data['quant_funcs_Q2n_hist'][k_idx_for_L_inv]
                phi_n2_current_vals = Q2_k_inv(phi_n2_current_vals)
        plt.plot(gX_unit, phi_n2_current_vals, color=p_colors[n_phi_iter], label=f'n={n_phi_iter}', alpha=0.7)
    plt.plot(gX_unit, gX_unit, 'k:', label='Identity y=x', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel(r"$\Phi_n^2(x)$")
    plt.legend(fontsize=leg_fs, ncol=leg_nc)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    save_current_figure('plot_13_phi2_evolution')
    plt.show()

    print("--- Plotting Complete ---")


# --- Main Simulation Logic ---
def run_simulation_main_loop(num_iterations_max, marginal_name1, marginal_params1, marginal_name2, marginal_params2,
                             copula_name_initial, copula_param_initial, num_grid_points_Ng, show_titles,
                             save_plots, plot_format, size_adjustment_percent):
    total_sim_time_start = time.time()
    print(f"\n--- Simulation Commencing ---")
    print(f"Target L_n Iterations: {num_iterations_max} (L0 to L{num_iterations_max}) | Grid Ng: {num_grid_points_Ng}")
    print(f"Marginal 1: {marginal_name1} {marginal_params1} | Marginal 2: {marginal_name2} {marginal_params2}")
    print(f"Initial Copula (for F's dependence): {copula_name_initial} {copula_param_initial}\n")

    F1_dist_init = get_marginal_distribution(marginal_name1, marginal_params1)
    F2_dist_init = get_marginal_distribution(marginal_name2, marginal_params2)
    C_init_cdf_func, C_init_pdf_func = get_copula(copula_name_initial, copula_param_initial)
    marg_lbl1 = f"{marginal_name1}{marginal_params1 if marginal_params1 else ''}"
    marg_lbl2 = f"{marginal_name2}{marginal_params2 if marginal_params2 else ''}"
    mnames_plot_lbl = f"{marg_lbl1}/{marg_lbl2}"

    results_hist = {'L_n_grids_history': [], 'wn_moments_hist': [],
                    'marg_cdfs_L1n_hist': [], 'marg_cdfs_L2n_hist': [],
                    'marg_pdfs_l1n_hist': [], 'marg_pdfs_l2n_hist': [],
                    'quant_funcs_Q1n_hist': [], 'quant_funcs_Q2n_hist': [],
                    'm21_given_u1_funcs_hist': [],
                    'rho_S_hist': [], 'tau_K_hist': [], 'ratio_metric_hist': [],
                    'Cn_copula_grids_hist': []}

    print(f"Computing L0 (Represents n=0 in L_n series)...")
    L_curr_grid, wF_denom_L0 = compute_L0_initial_distribution(
        F1_dist_init.ppf, F2_dist_init.ppf, C_init_pdf_func,
        num_grid_points_Ng, num_integration_points_L0=max(num_grid_points_Ng, 150)
    )
    if np.isnan(wF_denom_L0):
        print("CRITICAL: L0 computation failed (wF invalid). Aborting.")
        return

    cell_h_pdf_calc = 1.0 / num_grid_points_Ng
    for iter_n_val in range(num_iterations_max + 1):
        iter_time_start = time.time()
        print(f"\n--- Processing L_{iter_n_val} (Iteration {iter_n_val}/{num_iterations_max}) ---")
        results_hist['L_n_grids_history'].append(L_curr_grid.copy())
        L1n_cdf_v, L2n_cdf_v, Q1n_qf_v, Q2n_qf_v = get_marginal_cdfs_and_quantiles(L_curr_grid, num_grid_points_Ng)
        results_hist['marg_cdfs_L1n_hist'].append(L1n_cdf_v)
        results_hist['marg_cdfs_L2n_hist'].append(L2n_cdf_v)
        results_hist['quant_funcs_Q1n_hist'].append(Q1n_qf_v)
        results_hist['quant_funcs_Q2n_hist'].append(Q2n_qf_v)

        l1n_pdf_v_unnorm = np.gradient(L1n_cdf_v, cell_h_pdf_calc, edge_order=2)
        sum_l1_h_trapz = np.trapezoid(l1n_pdf_v_unnorm, dx=cell_h_pdf_calc) if num_grid_points_Ng > 0 else np.sum(
            l1n_pdf_v_unnorm)
        l1n_pdf_v = np.maximum(0, l1n_pdf_v_unnorm / (sum_l1_h_trapz if sum_l1_h_trapz > 1e-10 else 1.0))
        l2n_pdf_v_unnorm = np.gradient(L2n_cdf_v, cell_h_pdf_calc, edge_order=2)
        sum_l2_h_trapz = np.trapezoid(l2n_pdf_v_unnorm, dx=cell_h_pdf_calc) if num_grid_points_Ng > 0 else np.sum(
            l2n_pdf_v_unnorm)

        l2n_pdf_v = np.maximum(0, l2n_pdf_v_unnorm / (sum_l2_h_trapz if sum_l2_h_trapz > 1e-10 else 1.0))
        results_hist['marg_pdfs_l1n_hist'].append(l1n_pdf_v)
        results_hist['marg_pdfs_l2n_hist'].append(l2n_pdf_v)
        rho_S_v, tau_K_v, ratio_v, E1_curr, E2_curr, wn_curr_v = calculate_dependence_and_moments(L_curr_grid,
                                                                                                  num_grid_points_Ng)
        results_hist['wn_moments_hist'].append(wn_curr_v)
        results_hist['rho_S_hist'].append(rho_S_v)
        results_hist['tau_K_hist'].append(tau_K_v)
        results_hist['ratio_metric_hist'].append(ratio_v)
        results_hist['m21_given_u1_funcs_hist'].append(
            calculate_conditional_expectation_m2_given_u1(L_curr_grid, l1n_pdf_v, num_grid_points_Ng))
        results_hist['Cn_copula_grids_hist'].append(
            get_copula_from_L_distribution(L_curr_grid, num_grid_points_Ng, Q1n_qf_v, Q2n_qf_v))
        print(
            f"  L_{iter_n_val} processed: E12_{iter_n_val}={wn_curr_v:.4f}, E1={E1_curr:.4f}, E2={E2_curr:.4f}, Spearman rho={rho_S_v:.4f}, Kendall tau={tau_K_v:.4f}")
        if iter_n_val < num_iterations_max:
            if not np.isfinite(wn_curr_v) or wn_curr_v <= 1e-10:
                print(f"STOP: w_{iter_n_val}={wn_curr_v} is invalid for L_{{{iter_n_val + 1}}}. Halting.")
                break
            L_next_grid = compute_Ln_plus_1_from_Ln(L_curr_grid, num_grid_points_Ng, Q1n_qf_v, Q2n_qf_v, wn_curr_v)
            if iter_n_val > 0 and np.allclose(L_next_grid, L_curr_grid, atol=1e-5):
                print(f"CONVERGENCE: L_{{{iter_n_val + 1}}} numerically close to L_{{{iter_n_val}}}.")
            L_curr_grid = L_next_grid
        elif iter_n_val == num_iterations_max:
            print(f"Maximum number of L_n iterations ({num_iterations_max}) reached.")
        print(f"  Iteration {iter_n_val} (L_{iter_n_val}) processing time: {time.time() - iter_time_start:.2f}s")
    print(f"\n--- Simulation Processing Complete ---")
    print(f"Total simulation logic time: {time.time() - total_sim_time_start:.2f}s")
    plot_simulation_results(results_hist, F1_dist_init, F2_dist_init, C_init_cdf_func, num_grid_points_Ng,
                            num_iterations_max, mnames_plot_lbl, copula_name_initial, show_titles,
                            save_plots, plot_format, size_adjustment_percent)
    print("\n--- Script Execution Finished ---")


def _get_validated_input(prompt_text, default_value, type_caster, validation_func=None, error_msg="Invalid input."):
    while True:
        try:
            user_val_str = input(f"{prompt_text} [default: {default_value}]: ")
            value_to_test = default_value if not user_val_str else type_caster(user_val_str)
            if validation_func:
                constraint_ok = False
                constraint_desc_str = "passes constraint"
                if isinstance(validation_func, tuple) and len(validation_func) == 2 and callable(validation_func[0]):
                    actual_validation_lambda = validation_func[0]
                    constraint_desc_str = validation_func[1]
                    constraint_ok = actual_validation_lambda(value_to_test)
                elif callable(validation_func):
                    actual_validation_lambda = validation_func
                    constraint_ok = actual_validation_lambda(value_to_test)
                    if hasattr(actual_validation_lambda,
                               '__doc__') and actual_validation_lambda.__doc__: constraint_desc_str = actual_validation_lambda.__doc__
                else:
                    print("Warning: Invalid validation_func structure provided.")
                    constraint_ok = True
                if not constraint_ok:
                    print(f"{error_msg} (Constraint: {constraint_desc_str}). Try again.")
                    continue
            return value_to_test
        except ValueError:
            print(
                f"Invalid type. Expected input convertible to '{type_caster.__name__ if hasattr(type_caster, '__name__') else str(type_caster)}'.")
        except Exception as e:
            print(f"Unexpected input error: {e}")
            traceback.print_exc()


def get_user_parameters():
    DEFAULT_NUM_ITERATIONS = 10
    DEFAULT_GRID_SIZE_NG = 100
    DEFAULT_MARGINAL_1_NAME, DEFAULT_MARGINAL_1_PARAMS = 'uniform', {}
    DEFAULT_MARGINAL_2_NAME, DEFAULT_MARGINAL_2_PARAMS = 'uniform', {}
    DEFAULT_COPULA_NAME = 'gaussian'
    DEFAULT_COPULA_PARAM_GAUSS = 0.5
    DEFAULT_COPULA_PARAM_T = {'rho': 0.5, 'nu': 4.}
    DEFAULT_COPULA_PARAM_CLAYTON = 2.0
    DEFAULT_COPULA_PARAM_GUMBEL = 2.0
    DEFAULT_COPULA_PARAM_FRANK = 5.0
    DEFAULT_COPULA_PARAM_AMH_THETA = -0.5
    DEFAULT_COPULA_PARAM_FRECHETMIX_ALPHA = 0.75
    DEFAULT_COPULA_PARAM_FGM_THETA = 0.5

    AVAILABLE_MARGINALS = {
        'uniform': {'params': [], 'defaults': [], 'constraints': [], 'desc': "U(0,1)"},
        'pareto': {'params': ['b'], 'defaults': [2.], 'constraints': [(lambda b_val: b_val > 0, "b>0")],
                   'desc': "Pareto (Lomax, b>0)"},
        'lognormal': {'params': ['s', 'scale'], 'defaults': [1., 1.],
                      'constraints': [(lambda s_val: s_val > 0, "s>0"), (lambda sc_val: sc_val > 0, "scale>0")],
                      'desc': "LogN(s>0,scale>0)"},
        'gamma': {'params': ['a', 'scale'], 'defaults': [2., 1.],
                  'constraints': [(lambda a_val: a_val > 0, "a>0"), (lambda sc_val: sc_val > 0, "scale>0")],
                  'desc': "Gamma(a>0,scale>0)"},
        'beta': {'params': ['a', 'b'], 'defaults': [2., 2.],
                 'constraints': [(lambda a_val: a_val > 0, "a>0"), (lambda b_val: b_val > 0, "b>0")],
                 'desc': "Beta(a>0,b>0)"},
        'sinewave': {'params': ['k', 'A'], 'defaults': [3, 0.5],
                     'constraints': [(lambda k_val: isinstance(k_val, int) and k_val >= 1, "k:int>=1"),
                                     (lambda A_val: -1 <= A_val <= 1, "A:[-1,1]")],
                     'desc': "SineWave(k_int>=1,A_amp[-1,1])"},
        'betamix': {'params': ['a1', 'b1', 'a2', 'b2', 'w'], 'defaults': [2.0, 5.0, 5.0, 2.0, 0.5],
                    'constraints': [(lambda p_val: p_val > 0, "val>0")] * 4 + [
                        (lambda p_w_val: 0 < p_w_val < 1, "0<w<1")], 'desc': "BetaMix(Beta_params>0;0<w<1)"}
    }
    AVAILABLE_COPULAS = {
        'independent': {'param_names': [], 'default_param_vals': None, 'constraints': []},
        'gaussian': {'param_names': ['rho'], 'default_param_vals': [DEFAULT_COPULA_PARAM_GAUSS],
                     'constraints': [(lambda r_val: -1 <= r_val <= 1, "rho:[-1,1]")]},
        'clayton': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_CLAYTON],
                    'constraints': [(lambda t_val: t_val >= -1 and abs(t_val) > 1e-9, "theta >= -1, non-zero")]},
        'gumbel': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_GUMBEL],
                   'constraints': [(lambda t_val: t_val >= 1, "theta>=1")]},
        'frank': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_FRANK],
                  'constraints': [(lambda t_val: abs(t_val) > 1e-9, "theta!=0")]},
        't': {'param_names': ['rho', 'nu'],
              'default_param_vals': [DEFAULT_COPULA_PARAM_T['rho'], DEFAULT_COPULA_PARAM_T['nu']],
              'constraints': [(lambda r_val: -1 <= r_val <= 1, "rho:[-1,1]"), (lambda nu_val: nu_val > 0, "nu>0")]},
        'amh': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_AMH_THETA],
                'constraints': [(lambda t_val: -1 <= t_val <= 1 and abs(t_val) > 1e-9, "theta:[-1,1], !=0")]},
        'frechetmix': {'param_names': ['alpha'], 'default_param_vals': [DEFAULT_COPULA_PARAM_FRECHETMIX_ALPHA],
                       'constraints': [(lambda a_val: 0 <= a_val <= 1, "alpha:[0,1]")]},
        'fgm': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_FGM_THETA],
                'constraints': [(lambda t_val: -1 <= t_val <= 1, "theta:[-1,1]")]}}

    print("--- Setup Simulation Parameters ---")
    num_iter = _get_validated_input("Number of L_n iterations N (L0 to LN)", DEFAULT_NUM_ITERATIONS, int,
                                    (lambda n_val: n_val >= 0, "N>=0"))
    grid_ng = _get_validated_input("Grid size Ng for discretization (e.g., 50-150, higher is slower)",
                                   DEFAULT_GRID_SIZE_NG, int,
                                   (lambda ng_val: ng_val >= 10, "Ng>=10 preferred for stability"))

    show_titles_str = _get_validated_input("Plot with titles? (yes/no)", "yes", str,
                                           lambda s: s.lower() in ['yes', 'no'])
    show_titles = show_titles_str.lower() == 'yes'

    save_plots_str = _get_validated_input("Do you want to save the plots? (yes/no)", "yes", str,
                                          lambda s: s.lower() in ['yes', 'no'])
    save_plots = save_plots_str.lower() == 'yes'
    plot_format = None

    if save_plots:
        supported_formats = plt.figure().canvas.get_supported_filetypes()
        plt.close()

        recommended_formats = "Recommended: jpg, png, pdf, eps"
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

    size_options_map = {'-30%': -30, '-20%': -20, '-10%': -10, 'benchmark': 0, '+10%': 10, '+20%': 20, '+30%': 30}
    size_prompt = f"Choose plot size from benchmark ({list(size_options_map.keys())})"

    while True:
        user_size_choice = input(f"{size_prompt} [default: benchmark]: ").lower()
        if not user_size_choice:
            user_size_choice = 'benchmark'

        if user_size_choice in size_options_map:
            size_adjustment_percent = size_options_map[user_size_choice]
            break
        else:
            print("Invalid choice. Please select from the provided options.")

    def _get_single_marginal_config(m_num_idx):
        print(f"\n--- Configuring Marginal Distribution F{m_num_idx} ---")
        print("Available choices (name: description):")
        [print(f"  {m_key}: {m_info_dict['desc']}") for m_key, m_info_dict in AVAILABLE_MARGINALS.items()]
        default_m_name = DEFAULT_MARGINAL_1_NAME if m_num_idx == 1 else DEFAULT_MARGINAL_2_NAME
        chosen_m_name = _get_validated_input(f"Choose marginal F{m_num_idx} name", default_m_name, str,
                                             lambda name_in: name_in.lower() in AVAILABLE_MARGINALS).lower()
        selected_marg_info = AVAILABLE_MARGINALS[chosen_m_name]
        marg_params_dict = {}
        if selected_marg_info['params']:
            print(f"Parameters for {chosen_m_name} ({selected_marg_info['desc']}):")
            for i, p_name in enumerate(selected_marg_info['params']):
                p_default = selected_marg_info['defaults'][i]
                p_validation_tuple = selected_marg_info['constraints'][i]
                p_prompt = f"  Enter param '{p_name}' ({p_validation_tuple[1]})"
                p_caster_func = int if chosen_m_name == 'sinewave' and p_name == 'k' else float
                marg_params_dict[p_name] = _get_validated_input(p_prompt, p_default, p_caster_func, p_validation_tuple)
        return chosen_m_name, marg_params_dict

    m1_name_input, m1_params_input = _get_single_marginal_config(1)
    m2_name_input, m2_params_input = _get_single_marginal_config(2)

    print("\n--- Configuring Initial Copula C_init (for dependence of F to generate L0) ---")
    print("Available choices: ")
    [print(f"  {c_key}", end='') for c_key in AVAILABLE_COPULAS.keys()]
    print()
    chosen_copula_name = _get_validated_input("Choose copula type for C_init", DEFAULT_COPULA_NAME, str,
                                              lambda name_in: name_in.lower() in AVAILABLE_COPULAS).lower()
    selected_cop_info = AVAILABLE_COPULAS[chosen_copula_name]
    copula_final_param = None
    if selected_cop_info['param_names']:
        print(f"Parameters for {chosen_copula_name}:")
        temp_param_list_parsed = []
        for i, p_name in enumerate(selected_cop_info['param_names']):
            p_default = selected_cop_info['default_param_vals'][i]
            p_validation_tuple = selected_cop_info['constraints'][i]
            p_prompt = f"  Enter param '{p_name}' ({p_validation_tuple[1]})"
            temp_param_list_parsed.append(_get_validated_input(p_prompt, p_default, float, p_validation_tuple))
        if len(selected_cop_info['param_names']) == 1:
            copula_final_param = temp_param_list_parsed[0]
        else:
            copula_final_param = dict(zip(selected_cop_info['param_names'], temp_param_list_parsed))
    else:
        copula_final_param = selected_cop_info['default_param_vals']

    return num_iter, grid_ng, m1_name_input, m1_params_input, m2_name_input, m2_params_input, \
        chosen_copula_name, copula_final_param, show_titles, save_plots, plot_format, size_adjustment_percent


# --- Main Execution Script ---
if __name__ == "__main__":
    try:
        (N_ITER_MAX, GRID_NG_USER,
         M1_NAME, M1_PARAMS, M2_NAME, M2_PARAMS,
         COP_NAME_INIT, COP_PARAM_INIT,
         SHOW_TITLES_USER, SAVE_PLOTS_USER, PLOT_FORMAT_USER,
         SIZE_ADJUST_USER) = get_user_parameters()

        run_simulation_main_loop(N_ITER_MAX,
                                 M1_NAME, M1_PARAMS, M2_NAME, M2_PARAMS,
                                 COP_NAME_INIT, COP_PARAM_INIT, GRID_NG_USER,
                                 SHOW_TITLES_USER, SAVE_PLOTS_USER, PLOT_FORMAT_USER,
                                 SIZE_ADJUST_USER)
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted by user (Ctrl+C) ---")
    except Exception as main_exec_err:
        print(f"\n--- CRITICAL ERROR IN MAIN SCRIPT EXECUTION ---")
        print(f"Error Type: {type(main_exec_err).__name__}")
        print(f"Error Message: {main_exec_err}")
        print("Traceback follows:")
        traceback.print_exc()
        print("--- SCRIPT TERMINATED DUE TO ERROR ---")