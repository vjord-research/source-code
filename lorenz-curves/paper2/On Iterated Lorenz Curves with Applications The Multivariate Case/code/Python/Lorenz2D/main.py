import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
            val = u_clip * v_clip
            if theta > 0:
                val_inside = u_clip ** (-theta) + v_clip ** (-theta) - 1
                val = np.maximum(val_inside, tiny) ** (-1 / theta)
            elif -1 <= theta < 0:
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
            val = np.exp(-((lu ** th + lv ** th) ** (1 / th)))
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
                replacement_val_when_super_small = np.where(is_close_to_zero_tol, tiny ** 3,
                                                            np.sign(denominator_arr) * (tiny ** 3))
                effective_denominator = np.where(np.abs(denominator_arr) < (tiny ** 3),
                                                 replacement_val_when_super_small, denominator_arr)
                res[m_interior] = (ui * vi) / effective_denominator
            return np.clip(res, 0, 1)

        def pdf_amh(u, v):
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            mask_out_of_bounds = (u_arr <= tiny) | (u_arr >= 1 - tiny) | (v_arr <= tiny) | (v_arr >= 1 - tiny)
            u_c = np.clip(u_arr, tiny, 1 - tiny)
            v_c = np.clip(v_arr, tiny, 1 - tiny)
            num_term1 = theta * (theta - 1.0) * (1.0 - u_c) * (1.0 - v_c)
            num_term2 = theta * (u_c + v_c - 2.0 * u_c * v_c)
            num_term3 = 1.0 - theta
            numerator_pdf = num_term1 + num_term2 + num_term3
            denominator_base_pdf = 1.0 - theta * (1.0 - u_c) * (1.0 - v_c)
            pdf_val = np.where(np.abs(denominator_base_pdf) < tiny ** 2,
                               np.where(numerator_pdf * np.sign(denominator_base_pdf ** 3) >= 0, 1e12, 0.0),
                               numerator_pdf / (denominator_base_pdf ** 3))
            return np.where(mask_out_of_bounds, 0., np.maximum(0, np.nan_to_num(pdf_val, nan=0, posinf=1e12)))

        return cdf_amh, pdf_amh

    elif name == 'frechetmix':
        alpha = param
        if not (0 <= alpha <= 1): raise ValueError("FrechetMix alpha must be in [0,1].")
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
        if not (-1 <= theta <= 1): raise ValueError("FGM copula theta must be in [-1,1].")

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
    L1_raw = L_grid[:, Ng].copy()
    L2_raw = L_grid[Ng, :].copy()
    L1 = np.maximum.accumulate(L1_raw)
    L2 = np.maximum.accumulate(L2_raw)
    L1 = np.clip(L1, 0, 1)
    L2 = np.clip(L2, 0, 1)
    L1[0], L2[0] = 0, 0
    L1[Ng], L2[Ng] = 1, 1

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
        qf_interpolator = interpolate.interp1d(final_unique_cdf, final_unique_x, kind='linear', bounds_error=False,
                                               fill_value=(0., 1.))
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


def _calculate_conditional_expectation(L_curr_cdf_grid, Ng, axis_of_integration):
    """
    axis_of_integration=1: E[X2 | X1=y]. axis_of_integration=0: E[X1 | X2=y].
    """
    grid_pts = np.linspace(0, 1, Ng + 1)
    h = 1.0 / Ng
    m_on_grid = np.full_like(grid_pts, 0.5)  # Fallback value
    tiny = 1e-12

    if Ng < 2: return m_on_grid

    u_mid_pts = np.linspace(h / 2, 1 - h / 2, Ng)
    pdf_biv = (L_curr_cdf_grid[1:, 1:] - L_curr_cdf_grid[:-1, 1:] - L_curr_cdf_grid[1:, :-1] + L_curr_cdf_grid[:-1,
                                                                                               :-1]) / (h ** 2)
    pdf_biv = np.maximum(0, np.nan_to_num(pdf_biv))

    # This discrete approach is more robust than interpolating and using quad
    if axis_of_integration == 1:  # Integrate over axis 1 (u2) for each u1
        # Denominator is the marginal pdf l1(u1)
        denominator_vals = integrate.trapezoid(pdf_biv, dx=h, axis=1)
        # Numerator is integral of u2 * l(u1, u2) over u2
        numerator_integrand = u_mid_pts * pdf_biv  # Broadcasting u2_mid_pts
        numerator_vals = integrate.trapezoid(numerator_integrand, dx=h, axis=1)

        m_at_mid = np.divide(numerator_vals, denominator_vals, out=np.full(Ng, 0.5), where=denominator_vals > tiny)
        interp = interpolate.interp1d(u_mid_pts, m_at_mid, kind='linear', fill_value="extrapolate", bounds_error=False)
        m_on_grid = interp(grid_pts)

    else:  # Integrate over axis 0 (u1) for each u2
        # Denominator is the marginal pdf l2(u2)
        denominator_vals = integrate.trapezoid(pdf_biv, dx=h, axis=0)
        # Numerator is integral of u1 * l(u1, u2) over u1
        numerator_integrand = (u_mid_pts[:, np.newaxis] * pdf_biv)  # Broadcasting u1_mid_pts
        numerator_vals = integrate.trapezoid(numerator_integrand, dx=h, axis=0)

        m_at_mid = np.divide(numerator_vals, denominator_vals, out=np.full(Ng, 0.5), where=denominator_vals > tiny)
        interp = interpolate.interp1d(u_mid_pts, m_at_mid, kind='linear', fill_value="extrapolate", bounds_error=False)
        m_on_grid = interp(grid_pts)

    return np.clip(np.nan_to_num(m_on_grid, nan=0.5), 0, 1)


# --- New/Revised Helper Functions ---
def find_all_crossings(y_values, x_values, target_y, interval_open=False):
    """Finds all x where function y(x) crosses a target_y value using linear interpolation."""
    crossings = set()
    diff = y_values - target_y
    for i in range(len(x_values) - 1):
        if np.isclose(diff[i], 0):
            crossings.add(x_values[i])
        elif np.sign(diff[i]) != np.sign(diff[i + 1]):
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = diff[i], diff[i + 1]
            root = x1 - y1 * (x2 - x1) / (y2 - y1)
            crossings.add(root)
    if np.isclose(diff[-1], 0):
        crossings.add(x_values[-1])

    if interval_open:
        # Strictly (0, 1)
        return sorted([c for c in list(crossings) if 1e-9 < c < 1.0 - 1e-9])
    else:  # interval (0, 1]
        return sorted([c for c in list(crossings) if 1e-9 < c <= 1.0])


def find_crossing_points(cdf_grid, grid_points):
    """Robustly finds all y in (0,1) where CDF(y) = y using linear interpolation."""
    return find_all_crossings(cdf_grid, grid_points, grid_points, interval_open=True)


def calculate_Gn_grids(Ln_curr_cdf_grid, Ng):
    gX_eval = np.linspace(0, 1, Ng + 1)
    h_grid = 1.0 / Ng
    G1n_grid = np.zeros(Ng + 1)
    G2n_grid = np.zeros(Ng + 1)
    tiny = 1e-12
    try:
        pdf_Ln_cells = (Ln_curr_cdf_grid[1:, 1:] - Ln_curr_cdf_grid[:-1, 1:] - Ln_curr_cdf_grid[1:,
                                                                               :-1] + Ln_curr_cdf_grid[:-1, :-1]) / (
                               h_grid ** 2)
        pdf_Ln_cells = np.maximum(pdf_Ln_cells, 0)
        pdf_Ln_cells = np.nan_to_num(pdf_Ln_cells, nan=0, posinf=1e12, neginf=0)
        mid_points = np.linspace(h_grid / 2, 1 - h_grid / 2, Ng)
        u1_mesh_mid, u2_mesh_mid = np.meshgrid(mid_points, mid_points, indexing='ij')
        integrand_vals_at_mid = u1_mesh_mid * u2_mesh_mid * pdf_Ln_cells
        Dn = np.sum(integrand_vals_at_mid * (h_grid ** 2))
        if Dn < tiny: return gX_eval, gX_eval
        inner_integral_over_u2 = np.sum(u2_mesh_mid * pdf_Ln_cells * h_grid, axis=1)
        numerator_terms_N1 = u1_mesh_mid[:, 0] * inner_integral_over_u2 * h_grid
        G1n_grid[1:] = np.cumsum(numerator_terms_N1) / Dn
        inner_integral_over_u1 = np.sum(u1_mesh_mid * pdf_Ln_cells * h_grid, axis=0)
        numerator_terms_N2 = u2_mesh_mid[0, :] * inner_integral_over_u1 * h_grid
        G2n_grid[1:] = np.cumsum(numerator_terms_N2) / Dn
        return np.clip(G1n_grid, 0, 1), np.clip(G2n_grid, 0, 1)
    except Exception as e_gn_calc:
        warnings.warn(f"Error in calculate_Gn_grids: {e_gn_calc}. Returning identity.")
        return gX_eval, gX_eval


# --- Plotting Helpers (MUST BE DEFINED BEFORE PLOTTING FUNCTION) ---
def get_plotting_x_axis(dist_obj, num_points=200):
    try:
        dist_name_attr = getattr(dist_obj, 'name', None) or (
                hasattr(dist_obj, 'dist') and getattr(dist_obj.dist, 'name', None))
        is_01_bounded = (hasattr(dist_obj, 'a') and hasattr(dist_obj, 'b') and np.isclose(dist_obj.a, 0) and np.isclose(
            dist_obj.b, 1))
        if dist_name_attr in ['uniform', 'beta', 'sinewave', 'betamix'] or is_01_bounded: return np.linspace(0, 1,
                                                                                                             num_points)
        s_min, s_max = dist_obj.support()
        if not (hasattr(dist_obj, 'ppf') and callable(dist_obj.ppf)): return np.linspace(0, 1, num_points)
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


def _find_best_label_x_index(list_of_y_arrays, search_range_start_pct=0.2, search_range_end_pct=0.8):
    if not list_of_y_arrays or len(list_of_y_arrays) < 2: return int(
        0.75 * len(list_of_y_arrays[0])) if list_of_y_arrays else 0
    num_points = len(list_of_y_arrays[0])
    start_index = int(search_range_start_pct * num_points)
    end_index = int(search_range_end_pct * num_points)
    best_x_index = start_index
    max_min_dist = -1
    for i in range(start_index, end_index):
        y_values_at_i = sorted([arr[i] for arr in list_of_y_arrays if i < len(arr)])
        diffs = np.diff(y_values_at_i)
        min_dist = np.min(diffs) if len(diffs) > 0 else 0
        if min_dist > max_min_dist:
            max_min_dist = min_dist
            best_x_index = i
    return best_x_index


# --- Plotting ---
def plot_simulation_results(sim_data, F1_init, F2_init, C_init_cdf, Ng, N_req, m_names_lbl, C_init_name,
                            show_titles, save_plots, plot_format, size_adjustment_percent,
                            run_additional_functionalities, num_labels_in_plots):
    gX_unit = np.linspace(0, 1, Ng + 1)
    N_run = len(sim_data['L_n_grids_history'])
    if N_run == 0:
        print("No data available for plotting.")
        return

    p_colors = plt.cm.viridis(np.linspace(0, 1, max(1, N_run)))
    leg_fs = 'small' if N_run <= 10 else 'xx-small'
    leg_nc = 1 if N_run <= 5 else (2 if N_run <= 20 else 3)
    if num_labels_in_plots > 0 and N_run > 0:
        indices_to_label = np.arange(min(num_labels_in_plots, N_run))
    else:
        indices_to_label = []

    def add_line_label(ax, n_index, x_data, y_data, color, base_x_idx, stagger_amount):
        if n_index in indices_to_label:
            stagger = (n_index % 3 - 1) * stagger_amount
            label_x_index = np.clip(base_x_idx + stagger, 0, len(x_data) - 1)
            label_x = x_data[label_x_index]
            label_y = y_data[label_x_index]
            ax.text(label_x, label_y, str(n_index), fontsize=7, color=color, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='circle,pad=0.1'))

    benchmark_single_figsize = (5.95, 4.675)
    scale_factor = 1.0 + (size_adjustment_percent / 100.0)
    final_single_figsize = (benchmark_single_figsize[0] * scale_factor, benchmark_single_figsize[1] * scale_factor)

    if save_plots:
        plot_dir = "simulation_plots"
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        print(f"\n--- Plotting Results (saving to '{plot_dir}/' as .{plot_format}) ---")
    else:
        print("\n--- Plotting Results (displaying only) ---")

    x_F1_plot = get_plotting_x_axis(F1_init)
    x_F2_plot = get_plotting_x_axis(F2_init)
    itr_plot_idx = np.arange(N_run)

    def save_current_figure(filename_base):
        if save_plots:
            filepath = os.path.join(plot_dir, f"{filename_base}.{plot_format}")
            save_kwargs = {'dpi': 600} if plot_format in ['png', 'jpg', 'jpeg', 'tiff'] else {}
            plt.savefig(filepath, **save_kwargs)

    # --- Plot 1: Initial Distributions (Combined) ---
    fig_initial, axs_initial = plt.subplots(2, 2, figsize=final_single_figsize)
    if show_titles: fig_initial.suptitle(r"Plot 1: Initial Marginals (d.f.s and p.d.f.s)", fontsize=14)
    axs_initial[0, 0].plot(x_F1_plot, F1_init.cdf(x_F1_plot), label=f"$F_1$: {m_names_lbl.split('/')[0]}")
    axs_initial[0, 0].plot([0, 1], [0, 1], 'k:', alpha=0.7, label='y=x')
    axs_initial[0, 0].legend(fontsize='x-small')
    axs_initial[0, 0].grid(True)
    axs_initial[0, 0].set_title(r"$F_1$", fontsize='small')
    axs_initial[0, 0].set_xlabel('$x_1$')
    axs_initial[0, 0].set_ylabel('$F_1(x_1)$')
    axs_initial[0, 0].tick_params(axis='both', which='major', labelsize='x-small')
    axs_initial[0, 1].plot(x_F2_plot, F2_init.cdf(x_F2_plot), label=f"$F_2$: {m_names_lbl.split('/')[1]}", c='r')
    axs_initial[0, 1].plot([0, 1], [0, 1], 'k:', alpha=0.7, label='y=x')
    axs_initial[0, 1].legend(fontsize='x-small')
    axs_initial[0, 1].grid(True)
    axs_initial[0, 1].set_title(r"$F_2$", fontsize='small')
    axs_initial[0, 1].set_xlabel('$x_2$')
    axs_initial[0, 1].set_ylabel('$F_2(x_2)$')
    axs_initial[0, 1].tick_params(axis='both', which='major', labelsize='x-small')
    axs_initial[1, 0].plot(x_F1_plot, F1_init.pdf(x_F1_plot), label="$f_1$")
    axs_initial[1, 0].legend(fontsize='x-small')
    axs_initial[1, 0].grid(True)
    axs_initial[1, 0].set_ylim(bottom=0)
    axs_initial[1, 0].set_title(r"$f_1$", fontsize='small')
    axs_initial[1, 0].set_xlabel('$x_1$')
    axs_initial[1, 0].set_ylabel('$f_1(x_1)$')
    axs_initial[1, 0].tick_params(axis='both', which='major', labelsize='x-small')
    axs_initial[1, 1].plot(x_F2_plot, F2_init.pdf(x_F2_plot), label="$f_2$", c='r')
    axs_initial[1, 1].legend(fontsize='x-small')
    axs_initial[1, 1].grid(True)
    axs_initial[1, 1].set_ylim(bottom=0)
    axs_initial[1, 1].set_title(r"$f_2$", fontsize='small')
    axs_initial[1, 1].set_xlabel('$x_2$')
    axs_initial[1, 1].set_ylabel('$f_2(x_2)$')
    axs_initial[1, 1].tick_params(axis='both', which='major', labelsize='x-small')
    fig_initial.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_01_Initial_Distributions')
    plt.show()

    stagger = int(0.05 * Ng)
    # --- Plot 2: L1n CDF Evolution ---
    fig, ax = plt.subplots(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 2: Marginal CDF Evolution $L_1^{n}$", fontsize=14)
    lines_to_label_data = [sim_data['marg_cdfs_L1n_hist'][i] for i in indices_to_label]
    best_x = _find_best_label_x_index(lines_to_label_data)
    for i in range(N_run):
        line_data = sim_data['marg_cdfs_L1n_hist'][i]
        ax.plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
        add_line_label(ax, i, gX_unit, line_data, p_colors[i], best_x, stagger)
    ax.plot(gX_unit, gX_unit, 'k:', alpha=0.7)
    ax.legend(fontsize=leg_fs, ncol=leg_nc)
    ax.grid(True)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$L_1^{n}(x_1)$')
    plt.tight_layout()
    save_current_figure('plot_02_L1n_CDF')
    plt.show()

    # --- Plot 3: L2n CDF Evolution ---
    fig, ax = plt.subplots(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 3: Marginal CDF Evolution $L_2^{n}$", fontsize=14)
    lines_to_label_data = [sim_data['marg_cdfs_L2n_hist'][i] for i in indices_to_label]
    best_x = _find_best_label_x_index(lines_to_label_data)
    for i in range(N_run):
        line_data = sim_data['marg_cdfs_L2n_hist'][i]
        ax.plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
        add_line_label(ax, i, gX_unit, line_data, p_colors[i], best_x, stagger)
    ax.plot(gX_unit, gX_unit, 'k:', alpha=0.7)
    ax.legend(fontsize=leg_fs, ncol=leg_nc)
    ax.grid(True)
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$L_2^{n}(x_2)$')
    plt.tight_layout()
    save_current_figure('plot_03_L2n_CDF')
    plt.show()

    # --- Plot 4: l1n PDF Evolution ---
    fig, ax = plt.subplots(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 4: Marginal PDF Evolution $l_1^{n}$", fontsize=14)
    lines_to_label_data = [sim_data['marg_pdfs_l1n_hist'][i] for i in indices_to_label]
    best_x = _find_best_label_x_index(lines_to_label_data)
    for i in range(N_run):
        line_data = sim_data['marg_pdfs_l1n_hist'][i]
        ax.plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
        add_line_label(ax, i, gX_unit, line_data, p_colors[i], best_x, stagger)
    ax.legend(fontsize=leg_fs, ncol=leg_nc)
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$l_1^{n}(x_1)$')
    plt.tight_layout()
    save_current_figure('plot_04_l1n_PDF')
    plt.show()

    # --- Plot 5: l2n PDF Evolution ---
    fig, ax = plt.subplots(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 5: Marginal PDF Evolution $l_2^{n}$", fontsize=14)
    lines_to_label_data = [sim_data['marg_pdfs_l2n_hist'][i] for i in indices_to_label]
    best_x = _find_best_label_x_index(lines_to_label_data)
    for i in range(N_run):
        line_data = sim_data['marg_pdfs_l2n_hist'][i]
        ax.plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
        add_line_label(ax, i, gX_unit, line_data, p_colors[i], best_x, stagger)
    ax.legend(fontsize=leg_fs, ncol=leg_nc)
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$l_2^{n}(x_2)$')
    plt.tight_layout()
    save_current_figure('plot_05_l2n_PDF')
    plt.show()

    # --- Plot 6: mu_n Evolution ---
    plt.figure(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 6: Evolution of $\mu_n=E[X_1^{L_n}X_2^{L_n}]$", fontsize=14)
    plt.plot(itr_plot_idx, sim_data['wn_moments_hist'], marker='o')
    plt.xlabel(r"n")
    plt.ylabel(r"$\mu_n$")
    plt.grid(True)
    plt.xticks(itr_plot_idx)
    plt.tight_layout()
    save_current_figure('plot_06_mu_n_evolution')
    plt.show()

    # --- Plot 33: Evolution of Cov(U1*U2, 1_{U1 \leq c_1^n}) under the linking copula of L^n ---
    try:
        cov_series = []
        for n in range(N_run):
            Ln_grid = sim_data['L_n_grids_history'][n]
            # Get marginals and quantiles for L^n
            L1n_vec, L2n_vec, Q1n_func, Q2n_func = get_marginal_cdfs_and_quantiles(Ln_grid, Ng)
            # Extract copula C_n(u1,u2) = L^n(Q1^n(u1), Q2^n(u2))
            Cn_grid = get_copula_from_L_distribution(Ln_grid, Ng, Q1n_func, Q2n_func)

            # Copula pdf approx on cells
            h = 1.0 / Ng
            cpdf_cells = (Cn_grid[1:, 1:] - Cn_grid[:-1, 1:] - Cn_grid[1:, :-1] + Cn_grid[:-1, :-1]) / (h * h)
            cpdf_cells = np.maximum(0.0, np.nan_to_num(cpdf_cells))

            # Midpoints (uniform scale)
            u_mid = np.linspace(h / 2, 1 - h / 2, Ng)
            U1_mid, U2_mid = np.meshgrid(u_mid, u_mid, indexing='ij')

            # Choose c1^n: median of fixed points if available; else 1.0
            c1_list = sim_data.get('c1n_crossings_hist', [[]] * N_run)[n] if 'c1n_crossings_hist' in sim_data else []
            if isinstance(c1_list, (list, tuple)) and len(c1_list) > 0:
                c1n = float(np.median(c1_list))
            else:
                c1n = 1.0

            # Define A and B on midpoints
            A = U1_mid * U2_mid
            B = (U1_mid <= c1n).astype(float)

            # Cell probabilities
            cell_probs = cpdf_cells * (h * h)
            total_p = cell_probs.sum()
            if total_p > 1e-12 and not np.isclose(total_p, 1.0, rtol=1e-2, atol=1e-10):
                cell_probs = cell_probs / total_p

            EA = np.sum(A * cell_probs)
            EB = np.sum(B * cell_probs)
            EAB = np.sum(A * B * cell_probs)
            cov_series.append(EAB - EA * EB)

        plt.figure(figsize=final_single_figsize)
        if show_titles:
            plt.title(
                r"Plot 33: Evolution of $\mathrm{Cov}\!\left(U_1U_2,\;\mathbf{1}_{\{U_1\leq c_1^n\}}\right)$ under the linking copula of $L^n$",
                fontsize=14)
        plt.plot(itr_plot_idx, cov_series, marker='o')
        plt.xlabel(r"n")
        plt.ylabel(r"$\mathrm{Cov}$")
        plt.grid(True)
        plt.xticks(itr_plot_idx)
        plt.tight_layout()
        save_current_figure('plot_33_cov_U1U2_indicator_evolution')
        plt.show()
    except Exception as _cov_err:
        warnings.warn(f"Plot 33 covariance evolution failed: {_cov_err}")

    # --- Plot 7 (was 24): c1n Evolution (FIXED) ---
    fig, ax = plt.subplots(figsize=final_single_figsize)
    if show_titles: ax.set_title(r"Plot 7: Evolution of fixed point(s) $c_1^n$ of $L_1^n$", fontsize=14)
    leg_handles = []
    for n, points_list in enumerate(sim_data['c1n_crossings_hist']):
        if not points_list:
            ax.plot(n, 1.0, 'x', color='green', markersize=8)
        elif len(points_list) == 1:
            ax.plot(n, points_list[0], 'o', color='C0')
        else:
            ax.vlines(n, min(points_list), max(points_list), color='gray', linestyle='--')
            ax.plot([n] * len(points_list), points_list, '.', color='black')
    leg_handles = [mlines.Line2D([], [], color='C0', marker='o', linestyle='None', label='Single Fixed Point'),
                   mlines.Line2D([], [], color='gray', marker='|', linestyle='None', markersize=10,
                                 label='Multiple Fixed Points (range)'),
                   mlines.Line2D([], [], color='green', marker='x', linestyle='None', label='No Fixed Point in (0,1)')]
    ax.legend(handles=leg_handles, fontsize='small')
    ax.set_xlabel(r"n")
    ax.set_ylabel(r"$c_1^n$")
    ax.grid(True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(itr_plot_idx)
    plt.tight_layout()
    save_current_figure('plot_07_c1n_evolution')
    plt.show()

    # --- Plot 8 (was 25): c2n Evolution (FIXED) ---
    fig, ax = plt.subplots(figsize=final_single_figsize)
    if show_titles: ax.set_title(r"Plot 8: Evolution of fixed point(s) $c_2^n$ of $L_2^n$", fontsize=14)
    for n, points_list in enumerate(sim_data['c2n_crossings_hist']):
        if not points_list:
            ax.plot(n, 1.0, 'x', color='green', markersize=8)
        elif len(points_list) == 1:
            ax.plot(n, points_list[0], 'o', color='C0')
        else:
            ax.vlines(n, min(points_list), max(points_list), color='gray', linestyle='--')
            ax.plot([n] * len(points_list), points_list, '.', color='black')
    ax.legend(handles=leg_handles, fontsize='small')
    ax.set_xlabel(r"n")
    ax.set_ylabel(r"$c_2^n$")
    ax.grid(True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(itr_plot_idx)
    plt.tight_layout()
    save_current_figure('plot_08_c2n_evolution')
    plt.show()

    # --- Plot 9 (was 7): Conditional Expectation Evolution (Corrected) ---
    fig, axs = plt.subplots(1, 2, figsize=(final_single_figsize[0] * 2, final_single_figsize[1]), sharey=True)
    if show_titles: fig.suptitle(r"Plot 9: Evolution of Conditional Expectations", fontsize=14)
    lines_to_label_m21 = [sim_data['m21_given_u1_funcs_hist'][i] for i in indices_to_label if
                          i < len(sim_data['m21_given_u1_funcs_hist'])]
    best_x_m21 = _find_best_label_x_index(lines_to_label_m21)
    for i in range(N_run):
        if i < len(sim_data['m21_given_u1_funcs_hist']):
            line_data = sim_data['m21_given_u1_funcs_hist'][i]
            axs[0].plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
            add_line_label(axs[0], i, gX_unit, line_data, p_colors[i], best_x_m21, stagger)
    axs[0].set_title(r"$m_{2|1}^{n}(x_1)=E[X_2^{L_n}|X_1^{L_n}=x_1]$")
    axs[0].legend(fontsize=leg_fs, ncol=leg_nc)
    axs[0].grid(True)
    axs[0].set_xlabel(r'$x_1$')
    axs[0].set_ylabel(r'Value')
    axs[0].set_ylim(0, 1)

    lines_to_label_m12 = [sim_data['m12_given_u2_funcs_hist'][i] for i in indices_to_label if
                          i < len(sim_data['m12_given_u2_funcs_hist'])]
    best_x_m12 = _find_best_label_x_index(lines_to_label_m12)
    for i in range(N_run):
        if i < len(sim_data['m12_given_u2_funcs_hist']):
            line_data = sim_data['m12_given_u2_funcs_hist'][i]
            axs[1].plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
            add_line_label(axs[1], i, gX_unit, line_data, p_colors[i], best_x_m12, stagger)
    axs[1].set_title(r"$m_{1|2}^{n}(x_2)=E[X_1^{L_n}|X_2^{L_n}=x_2]$")
    axs[1].legend(fontsize=leg_fs, ncol=leg_nc)
    axs[1].grid(True)
    axs[1].set_xlabel(r'$x_2$')
    plt.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_09_m_conditional_evolution')
    plt.show()

    # --- Plot 10 (was 8): h_n Evolution (Corrected) ---
    fig, axs = plt.subplots(1, 2, figsize=(final_single_figsize[0] * 2, final_single_figsize[1]), sharey=True)
    if show_titles: fig.suptitle(r"Plot 10: Evolution of $x \cdot m(x)$", fontsize=14)
    lines_to_label_h1 = [gX_unit * sim_data['m21_given_u1_funcs_hist'][i] for i in indices_to_label if
                         i < len(sim_data['m21_given_u1_funcs_hist'])]
    best_x_h1 = _find_best_label_x_index(lines_to_label_h1)
    for i in range(N_run):
        if i < len(sim_data['m21_given_u1_funcs_hist']):
            line_data = gX_unit * sim_data['m21_given_u1_funcs_hist'][i]
            axs[0].plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
            add_line_label(axs[0], i, gX_unit, line_data, p_colors[i], best_x_h1, stagger)
    axs[0].set_title(r"$h_1^n(x_1)=x_1 m_{2|1}^{n}(x_1)$")
    axs[0].legend(fontsize=leg_fs, ncol=leg_nc)
    axs[0].grid(True)
    axs[0].set_xlabel(r'$x_1$')
    axs[0].set_ylabel('Value')
    axs[0].set_ylim(0, 1)

    lines_to_label_h2 = [gX_unit * sim_data['m12_given_u2_funcs_hist'][i] for i in indices_to_label if
                         i < len(sim_data['m12_given_u2_funcs_hist'])]
    best_x_h2 = _find_best_label_x_index(lines_to_label_h2)
    for i in range(N_run):
        if i < len(sim_data['m12_given_u2_funcs_hist']):
            line_data = gX_unit * sim_data['m12_given_u2_funcs_hist'][i]
            axs[1].plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
            add_line_label(axs[1], i, gX_unit, line_data, p_colors[i], best_x_h2, stagger)
    axs[1].set_title(r"$h_2^n(x_2)=x_2 m_{1|2}^{n}(x_2)$")
    axs[1].legend(fontsize=leg_fs, ncol=leg_nc)
    axs[1].grid(True)
    axs[1].set_xlabel(r'$x_2$')
    plt.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_10_h_n_evolution')
    plt.show()

    # --- Plot 11 (was 9): Dependence Measures ---
    plt.figure(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 11: Dependence Measures ($\rho_S, \tau_K$)", fontsize=14)
    UM, VM = np.meshgrid(gX_unit, gX_unit, indexing='ij')
    Cinit_grid = C_init_cdf(UM, VM)
    rhoF_init, tauF_init, _, _, _, _ = calculate_dependence_and_moments(Cinit_grid, Ng)
    iter_dep_indices = [-1] + list(itr_plot_idx)
    rho_plot_vals = [rhoF_init] + sim_data['rho_S_hist']
    tau_plot_vals = [tauF_init] + sim_data['tau_K_hist']
    plt.plot(iter_dep_indices, rho_plot_vals, marker='o', label=r'Spearman $\rho_S$')
    plt.plot(iter_dep_indices, tau_plot_vals, marker='s', label=r'Kendall $\tau_K$')
    plt.text(0.5, 0.95, "(n=-1 for initial F's copula, n>=0 for $L_n$)", ha='center', transform=plt.gca().transAxes,
             fontsize='small')
    plt.xlabel("n")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.xticks(iter_dep_indices)
    plt.tight_layout()
    save_current_figure('plot_11_dependence_evolution')
    plt.show()

    # --- Plot 12 (was 10): Ratio Metric ---
    plt.figure(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 12: $\mathbb{E}[X_1^{L_n}X_2^{L_n}]/(\mathbb{E}[X_1^{L_n}]\mathbb{E}[X_2^{L_n}])$",
                              fontsize=14)
    plt.plot(itr_plot_idx, sim_data['ratio_metric_hist'], marker='o')
    plt.xlabel(r"n")
    plt.ylabel("Ratio")
    plt.grid(True)
    plt.xticks(itr_plot_idx)
    plt.tight_layout()
    save_current_figure('plot_12_ratio_metric_evolution')
    plt.show()

    # --- Plot 13 (was 11): Copula Diagonals ---
    plt.figure(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 13: Copula Diagonals $C_n(x,x)$", fontsize=14)
    plt.plot(gX_unit, np.diag(Cinit_grid), label=f'Initial C ({C_init_name})', linestyle=':')
    diag_plot_indices = sorted(list(set(idx for idx in [0, min(1, N_run - 1), N_req, N_run - 1] if 0 <= idx < N_run)))
    for i_diag in diag_plot_indices:
        plt.plot(gX_unit, np.diag(sim_data['Cn_copula_grids_hist'][i_diag]), c=p_colors[i_diag],
                 label=f'$C_{{{i_diag}}}$ (from $L_{{{i_diag}}}$)', alpha=0.8)
    plt.plot(gX_unit, gX_unit, 'k--', lw=0.8, label='M(x,x)')
    plt.plot(gX_unit, np.maximum(0, 2 * gX_unit - 1), 'k-.', lw=0.8, label='W(x,x)')
    plt.xlabel("x")
    plt.ylabel(r"$C_n(x,x)$")
    plt.grid(True)
    plt.legend(fontsize='small')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    save_current_figure('plot_13_copula_diagonals')
    plt.show()

    # --- Plot 14 (was 12): Phi_n^1 Evolution ---
    fig, ax = plt.subplots(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 14: Evolution of Composed Quantile Function $\Phi_n^1(x)$", fontsize=14)
    phi1_lines_to_label = []
    all_phi1_lines = []
    for n_phi_iter in range(N_run):
        phi_n1_current_vals = gX_unit.copy()
        for k_idx_for_L_inv in range(n_phi_iter, -1, -1):
            if k_idx_for_L_inv < len(sim_data['quant_funcs_Q1n_hist']):
                phi_n1_current_vals = sim_data['quant_funcs_Q1n_hist'][k_idx_for_L_inv](phi_n1_current_vals)
        all_phi1_lines.append(phi_n1_current_vals)
        if n_phi_iter in indices_to_label: phi1_lines_to_label.append(phi_n1_current_vals)
    best_x = _find_best_label_x_index(phi1_lines_to_label)
    for i, line_data in enumerate(all_phi1_lines):
        ax.plot(gX_unit, line_data, color=p_colors[i], label=f'n={i}', alpha=0.7)
        add_line_label(ax, i, gX_unit, line_data, p_colors[i], best_x, stagger)
    ax.plot(gX_unit, gX_unit, 'k:', label='Identity y=x', alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Phi_n^1(x)$")
    ax.legend(fontsize=leg_fs, ncol=leg_nc)
    ax.grid(True)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_current_figure('plot_14_phi1_evolution')
    plt.show()

    # --- Plot 15 (was 13): Phi_n^2 Evolution ---
    fig, ax = plt.subplots(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 15: Evolution of Composed Quantile Function $\Phi_n^2(x)$", fontsize=14)
    phi2_lines_to_label = []
    all_phi2_lines = []
    for n_phi_iter in range(N_run):
        phi_n2_current_vals = gX_unit.copy()
        for k_idx_for_L_inv in range(n_phi_iter, -1, -1):
            if k_idx_for_L_inv < len(sim_data['quant_funcs_Q2n_hist']):
                phi_n2_current_vals = sim_data['quant_funcs_Q2n_hist'][k_idx_for_L_inv](phi_n2_current_vals)
        all_phi2_lines.append(phi_n2_current_vals)
        if n_phi_iter in indices_to_label: phi2_lines_to_label.append(phi_n2_current_vals)
    best_x = _find_best_label_x_index(phi2_lines_to_label)
    for i, line_data in enumerate(all_phi2_lines):
        ax.plot(gX_unit, line_data, color=p_colors[i], label=f'n={i}', alpha=0.7)
        add_line_label(ax, i, gX_unit, line_data, p_colors[i], best_x, stagger)
    ax.plot(gX_unit, gX_unit, 'k:', label='Identity y=x', alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Phi_n^2(x)$")
    ax.legend(fontsize=leg_fs, ncol=leg_nc)
    ax.grid(True)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_current_figure('plot_15_phi2_evolution')
    plt.show()

    # --- Plot 16: Marginal Means ---
    plt.figure(figsize=final_single_figsize)
    if show_titles: plt.title(r"Plot 16: Evolution of Marginal Means", fontsize=14)
    plt.plot(itr_plot_idx, sim_data['E1_hist'], marker='o', label=r'$\mu_1^n = E[X_1^{L_n}]$')
    plt.plot(itr_plot_idx, sim_data['E2_hist'], marker='s', label=r'$\mu_2^n = E[X_2^{L_n}]$')
    plt.xlabel(r"n")
    plt.ylabel("Mean Value")
    plt.grid(True)
    plt.xticks(itr_plot_idx)
    plt.legend()
    plt.tight_layout()
    save_current_figure('plot_16_marginal_means_evolution')
    plt.show()

    # --- Plot 17 (was 18): Lambda Functions Evolution ---
    fig, axs = plt.subplots(1, 2, figsize=(final_single_figsize[0] * 2, final_single_figsize[1]), sharey=True)
    if show_titles: fig.suptitle(r"Plot 17: Evolution of $\lambda_n$ Functions", fontsize=14)
    lines_to_label_l1 = [sim_data['lambda1_hist'][i] for i in indices_to_label if i < len(sim_data['lambda1_hist'])]
    best_x_l1 = _find_best_label_x_index(lines_to_label_l1)
    for i in range(N_run):
        if i < len(sim_data['lambda1_hist']):
            line_data = sim_data['lambda1_hist'][i]
            axs[0].plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
            add_line_label(axs[0], i, gX_unit, line_data, p_colors[i], best_x_l1, stagger)
    axs[0].axhline(1.0, color='k', linestyle=':', alpha=0.7)
    axs[0].set_title(r"$\lambda_1^n(y) = \frac{y}{D_n} m_{2|1}^n(y)$")
    axs[0].legend(fontsize=leg_fs, ncol=leg_nc)
    axs[0].grid(True)
    axs[0].set_xlabel('$y$')
    axs[0].set_ylabel(r'Value')
    axs[0].set_ylim(bottom=0)

    lines_to_label_l2 = [sim_data['lambda2_hist'][i] for i in indices_to_label if i < len(sim_data['lambda2_hist'])]
    best_x_l2 = _find_best_label_x_index(lines_to_label_l2)
    for i in range(N_run):
        if i < len(sim_data['lambda2_hist']):
            line_data = sim_data['lambda2_hist'][i]
            axs[1].plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
            add_line_label(axs[1], i, gX_unit, line_data, p_colors[i], best_x_l2, stagger)
    axs[1].axhline(1.0, color='k', linestyle=':', alpha=0.7)
    axs[1].set_title(r"$\lambda_2^n(y) = \frac{y}{D_n} m_{1|2}^n(y)$")
    axs[1].legend(fontsize=leg_fs, ncol=leg_nc)
    axs[1].grid(True)
    axs[1].set_xlabel('$y$')
    plt.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_17_lambda_evolution')
    plt.show()

    # --- Plot 18 (was 19): Roots of lambda=1 (using Kappa notation) ---
    fig, axs = plt.subplots(1, 2, figsize=(final_single_figsize[0] * 2, final_single_figsize[1]), sharey=True)
    if show_titles: fig.suptitle(r"Plot 18: Evolution of roots for $\lambda_n(y)=1$", fontsize=14)
    kappa1_root1_data = np.array(sim_data['kappa1_roots_lambda1_hist'])
    kappa1_root2_data = np.array(sim_data['kappa2_roots_lambda1_hist'])
    kappa2_root1_data = np.array(sim_data['kappa1_roots_lambda2_hist'])
    kappa2_root2_data = np.array(sim_data['kappa2_roots_lambda2_hist'])
    axs[0].plot(itr_plot_idx, kappa1_root1_data, marker='o', label=r'$\varkappa_1^n$ for $\lambda_1$')
    axs[0].plot(itr_plot_idx, kappa1_root2_data, marker='s', label=r'$\varkappa_2^n$ for $\lambda_1$')
    axs[0].set_title(r'Roots of $\lambda_1^n(y)=1$')
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('Root Value')
    axs[0].grid(True)
    axs[0].set_xticks(itr_plot_idx)
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].legend()
    axs[1].plot(itr_plot_idx, kappa2_root1_data, marker='o', label=r'$\varkappa_1^n$ for $\lambda_2$')
    axs[1].plot(itr_plot_idx, kappa2_root2_data, marker='s', label=r'$\varkappa_2^n$ for $\lambda_2$')
    axs[1].set_title(r'Roots of $\lambda_2^n(y)=1$')
    axs[1].set_xlabel('n')
    axs[1].grid(True)
    axs[1].set_xticks(itr_plot_idx)
    axs[1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_18_kappa_roots_evolution')
    plt.show()

    # --- Plot 19 (NEW): Transformed Roots ---
    fig, axs = plt.subplots(1, 2, figsize=(final_single_figsize[0] * 2, final_single_figsize[1]), sharey=True)
    if show_titles: fig.suptitle(r"Plot 19: Evolution of $l_1^n$ and $l_2^n$", fontsize=14)
    l1_root1_data = np.array(sim_data['l1_lambda1_hist'])
    l1_root2_data = np.array(sim_data['l2_lambda1_hist'])
    l2_root1_data = np.array(sim_data['l1_lambda2_hist'])
    l2_root2_data = np.array(sim_data['l2_lambda2_hist'])
    axs[0].plot(itr_plot_idx, l1_root1_data, marker='o', label=r'$l_1^n$ for $\lambda_1(L_1^{n,-1})$')
    axs[0].plot(itr_plot_idx, l1_root2_data, marker='s', label=r'$l_2^n$ for $\lambda_1(L_1^{n,-1})$')
    axs[0].set_title(r'$l_1^n$ and $l_2^n$ for $L_1^n$')
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('Values')
    axs[0].grid(True)
    axs[0].set_xticks(itr_plot_idx)
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].legend()
    axs[1].plot(itr_plot_idx, l2_root1_data, marker='o', label=r'$l_1^n$ for $\lambda_2(L_2^{n,-1})$')
    axs[1].plot(itr_plot_idx, l2_root2_data, marker='s', label=r'$l_2^n$ for $\lambda_2(L_2^{n,-1})$')
    axs[1].set_title(r'$l_1^n$ and $l_2^n$ for $L_2^n$')
    axs[1].set_xlabel('n')
    axs[1].grid(True)
    axs[1].set_xticks(itr_plot_idx)
    axs[1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_19_l_roots_evolution')
    plt.show()

    # --- Plot 20 (was 19): L(l_n-1) Evolution ---
    fig, axs = plt.subplots(2, 2, figsize=(final_single_figsize[0] * 2, final_single_figsize[1] * 2), sharey=True)
    if show_titles: fig.suptitle(r"Plot 20: $L^n$ at $l_n$", fontsize=14)
    l1_prev_root1 = np.roll(l1_root1_data, 1)
    l1_prev_root1[0] = np.nan
    l1_prev_root2 = np.roll(l1_root2_data, 1)
    l1_prev_root2[0] = np.nan
    l2_prev_root1 = np.roll(l2_root1_data, 1)
    l2_prev_root1[0] = np.nan
    l2_prev_root2 = np.roll(l2_root2_data, 1)
    l2_prev_root2[0] = np.nan
    axs[0, 0].plot(itr_plot_idx, sim_data['L1_at_l_lambda1_hist_root1'], marker='o', label=r'$L_1^{n-1}(l_{1, n-1})$')
    axs[0, 0].plot(itr_plot_idx, l1_root1_data, marker='.', linestyle='--', label=r'$l_{1, n}$', alpha=0.7)
    axs[0, 0].set_title(r'$L_1^n(l_n^1)$')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 1].plot(itr_plot_idx, sim_data['L1_at_l_lambda1_hist_root2'], marker='s', label=r'$L_1^{n-1}(l_{2, n-1})$')
    axs[0, 1].plot(itr_plot_idx, l1_root2_data, marker='.', linestyle='--', label=r'$l_{2, n}$', alpha=0.7)
    axs[0, 1].set_title(r'$L_1^n(l_n^2)$')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[1, 0].plot(itr_plot_idx, sim_data['L2_at_l_lambda2_hist_root1'], marker='o', label=r'$L_2^{n-1}(l_{1, n-1})$')
    axs[1, 0].plot(itr_plot_idx, l2_root1_data, marker='.', linestyle='--', label=r'$l_{1, n}$', alpha=0.7)
    axs[1, 0].set_title(r'$L_2^n(l_n^1)$')
    axs[1, 0].set_xlabel('n')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 1].plot(itr_plot_idx, sim_data['L2_at_l_lambda2_hist_root2'], marker='s', label=r'$L_2^{n-1}(l_{2, n-1})$')
    axs[1, 1].plot(itr_plot_idx, l2_root2_data, marker='.', linestyle='--', label=r'$l_{2, n}$', alpha=0.7)
    axs[1, 1].set_title(r'$L_2^n(l_n^2)$')
    axs[1, 1].set_xlabel('n')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_20_L_at_l_evolution')
    plt.show()

    # --- Plot 21 (was 20): Differences L(l)-l ---
    fig, axs = plt.subplots(2, 2, figsize=(final_single_figsize[0] * 2, final_single_figsize[1] * 2), sharey=True)
    if show_titles: fig.suptitle(r"Plot 21: Evolution of the Differences $L_{n-1}(l_{n-1})-l_{n}$", fontsize=14)
    diff11 = np.array(sim_data['L1_at_l_lambda1_hist_root1']) - l1_root1_data
    diff12 = np.array(sim_data['L1_at_l_lambda1_hist_root2']) - l1_root2_data
    diff21 = np.array(sim_data['L2_at_l_lambda2_hist_root1']) - l2_root1_data
    diff22 = np.array(sim_data['L2_at_l_lambda2_hist_root2']) - l2_root2_data
    axs[0, 0].plot(itr_plot_idx, diff11, marker='o')
    axs[0, 0].axhline(0, color='k', ls=':')
    axs[0, 0].set_title(r'$L_1^{n-1}(l_{1, n-1}) - l_{1, n}$')
    axs[0, 0].set_ylabel('Difference')
    axs[0, 0].grid(True)
    axs[0, 1].plot(itr_plot_idx, diff12, marker='s')
    axs[0, 1].axhline(0, color='k', ls=':')
    axs[0, 1].set_title(r'$L_1^{n-1}(l_{2, n-1}) - l_{2, n}$')
    axs[0, 1].grid(True)
    axs[1, 0].plot(itr_plot_idx, diff21, marker='o')
    axs[1, 0].axhline(0, color='k', ls=':')
    axs[1, 0].set_title(r'$L_2^{n-1}(l_{1, n-1}) - l_{1, n}$')
    axs[1, 0].set_xlabel('n')
    axs[1, 0].set_ylabel('Difference')
    axs[1, 0].grid(True)
    axs[1, 1].plot(itr_plot_idx, diff22, marker='s')
    axs[1, 1].axhline(0, color='k', ls=':')
    axs[1, 1].set_title(r'$L_2^{n-1}(l_{2, n-1}) - l_{2, n}$')
    axs[1, 1].set_xlabel('n')
    axs[1, 1].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96] if show_titles else [0, 0, 1, 1])
    save_current_figure('plot_21_L_at_l_differences')
    plt.show()

    if run_additional_functionalities:
        print("\n--- Plotting Additional Functionalities ---")

        # --- Plot 22 (was 16): G1n CDF Evolution ---
        fig, ax = plt.subplots(figsize=final_single_figsize)
        if show_titles: plt.title(r"Plot 22: Marginal CDF Evolution $G_1^{n}$", fontsize=14)
        lines_to_label_data = [sim_data['G1n_grids_hist'][i] for i in indices_to_label]
        best_x = _find_best_label_x_index(lines_to_label_data)
        for i in range(N_run):
            line_data = sim_data['G1n_grids_hist'][i]
            ax.plot(gX_unit, line_data, c=p_colors[i], label=f'n={i}', alpha=0.7)
            add_line_label(ax, i, gX_unit, line_data, p_colors[i], best_x, stagger)
        ax.plot(gX_unit, gX_unit, 'k:', alpha=0.7, label='y=x')
        ax.legend(fontsize=leg_fs, ncol=leg_nc)
        ax.grid(True)
        ax.set_xlabel('$y$')
        ax.set_ylabel('$G_1^{n}(y)$')
        plt.tight_layout()
        save_current_figure('plot_22_G1n_CDF')
        plt.show()

def run_simulation_main_loop(num_iterations_max, marginal_name1, marginal_params1, marginal_name2, marginal_params2,
                             copula_name_initial, copula_param_initial, num_grid_points_Ng, show_titles,
                             save_plots, plot_format, size_adjustment_percent, run_additional_functionalities,
                             num_labels_in_plots):
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

    results_hist = {
        'L_n_grids_history': [], 'wn_moments_hist': [], 'marg_cdfs_L1n_hist': [], 'marg_cdfs_L2n_hist': [],
        'marg_pdfs_l1n_hist': [], 'marg_pdfs_l2n_hist': [], 'quant_funcs_Q1n_hist': [], 'quant_funcs_Q2n_hist': [],
        'm21_given_u1_funcs_hist': [], 'm12_given_u2_funcs_hist': [], 'rho_S_hist': [], 'tau_K_hist': [],
        'ratio_metric_hist': [], 'Cn_copula_grids_hist': [], 'G1n_grids_hist': [], 'G2n_grids_hist': [],
        'c1n_crossings_hist': [], 'c2n_crossings_hist': [], 'E1_hist': [], 'E2_hist': [],
        'lambda1_hist': [], 'lambda2_hist': [],
        'kappa1_roots_lambda1_hist': [], 'kappa2_roots_lambda1_hist': [],
        'kappa1_roots_lambda2_hist': [], 'kappa2_roots_lambda2_hist': [],
        'l1_lambda1_hist': [], 'l2_lambda1_hist': [],
        'l1_lambda2_hist': [], 'l2_lambda2_hist': [],
        'L1_at_l_lambda1_hist_root1': [], 'L1_at_l_lambda1_hist_root2': [],
        'L2_at_l_lambda2_hist_root1': [], 'L2_at_l_lambda2_hist_root2': [], 'Bn_fields_mid_hist': [],
        'Un_scalar_hist': [], 'Hn_fields_mid_hist': []}

    print(f"Computing L0 (Represents n=0 in L_n series)...")
    L_curr_grid, wF_denom_L0 = compute_L0_initial_distribution(F1_dist_init.ppf, F2_dist_init.ppf, C_init_pdf_func,
                                                               num_grid_points_Ng,
                                                               num_integration_points_L0=max(num_grid_points_Ng, 150))
    if np.isnan(wF_denom_L0): print("CRITICAL: L0 computation failed (wF invalid). Aborting."); return

    cell_h_pdf_calc = 1.0 / num_grid_points_Ng
    gX_unit_grid = np.linspace(0, 1, num_grid_points_Ng + 1)
    kappa2_for_lambda1_lost, kappa2_for_lambda2_lost = False, False

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
        l1n_pdf_v = np.maximum(0, l1n_pdf_v_unnorm / (
            integrate.trapezoid(l1n_pdf_v_unnorm, dx=cell_h_pdf_calc) if num_grid_points_Ng > 0 else 1.0))
        l2n_pdf_v_unnorm = np.gradient(L2n_cdf_v, cell_h_pdf_calc, edge_order=2)
        l2n_pdf_v = np.maximum(0, l2n_pdf_v_unnorm / (
            integrate.trapezoid(l2n_pdf_v_unnorm, dx=cell_h_pdf_calc) if num_grid_points_Ng > 0 else 1.0))
        results_hist['marg_pdfs_l1n_hist'].append(l1n_pdf_v)
        results_hist['marg_pdfs_l2n_hist'].append(l2n_pdf_v)

        rho_S_v, tau_K_v, ratio_v, E1_curr, E2_curr, wn_curr_v = calculate_dependence_and_moments(L_curr_grid,
                                                                                                  num_grid_points_Ng)
        results_hist['wn_moments_hist'].append(wn_curr_v)
        results_hist['rho_S_hist'].append(rho_S_v)
        results_hist['tau_K_hist'].append(tau_K_v)
        results_hist['ratio_metric_hist'].append(ratio_v)
        results_hist['E1_hist'].append(E1_curr)
        results_hist['E2_hist'].append(E2_curr)

        # ---- NEW METRICS (extended): A_n, B_n, B1n/B2n, T_n, S_n, U_n, and ratio R_n ----
        try:
            h_new = 1.0 / num_grid_points_Ng
            # Joint pdf on cell midpoints
            pdf_cells_new = (L_curr_grid[1:, 1:] - L_curr_grid[:-1, 1:] - L_curr_grid[1:, :-1] + L_curr_grid[:-1,
                                                                                                 :-1]) / (h_new ** 2)
            pdf_cells_new = np.maximum(pdf_cells_new, 0.0)
            # Midpoints grid
            mid_new = np.linspace(h_new / 2.0, 1.0 - h_new / 2.0, num_grid_points_Ng)
            u1m_new, u2m_new = np.meshgrid(mid_new, mid_new, indexing='ij')
            # L1^n, L2^n evaluated at cell centers
            L1_mid_new = 0.5 * (L1n_cdf_v[:-1] + L1n_cdf_v[1:])
            L2_mid_new = 0.5 * (L2n_cdf_v[:-1] + L2n_cdf_v[1:])
            L1_mid_2D_new = L1_mid_new[:, None]
            L2_mid_2D_new = L2_mid_new[None, :]
            # A_n field on midpoints
            A_n_mid_new = L1_mid_2D_new * L2_mid_2D_new - (u1m_new * u2m_new * pdf_cells_new)
            results_hist.setdefault('An_fields_mid_hist', []).append(A_n_mid_new)
            # B_n field on midpoints (bilinear average for L_n at cell centers)
            Ln_mid_new = 0.25 * (
                        L_curr_grid[:-1, :-1] + L_curr_grid[1:, :-1] + L_curr_grid[:-1, 1:] + L_curr_grid[1:, 1:])
            B_n_mid_new = Ln_mid_new - (u1m_new * u2m_new * pdf_cells_new)
            results_hist.setdefault('Bn_fields_mid_hist', []).append(B_n_mid_new)
            H_n_mid_new = L1_mid_2D_new * L2_mid_2D_new * pdf_cells_new
            results_hist.setdefault('Hn_fields_mid_hist', []).append(H_n_mid_new)
            # B1n(y) and B2n(y) on the boundary grid
            B1n_new = L1n_cdf_v - gX_unit_grid * l1n_pdf_v
            B2n_new = L2n_cdf_v - gX_unit_grid * l2n_pdf_v
            results_hist.setdefault('B1n_series_hist', []).append(B1n_new)
            results_hist.setdefault('B2n_series_hist', []).append(B2n_new)
            # T_n and S_n
            Tn_val_new = float(np.sum(L1_mid_2D_new * L2_mid_2D_new * u1m_new * u2m_new * pdf_cells_new) * (h_new ** 2))
            Sn_val_new = Tn_val_new - float(wn_curr_v ** 2)
            results_hist.setdefault('Tn_scalar_hist', []).append(Tn_val_new)
            results_hist.setdefault('Sn_scalar_hist', []).append(Sn_val_new)
            # U_n = T_n - ∬ [x1 x2 l_n]^2
            Jn_new = u1m_new * u2m_new * pdf_cells_new
            Un_val_new = float(Tn_val_new - np.sum(Jn_new * Jn_new) * (h_new ** 2))
            results_hist.setdefault('Un_scalar_hist', []).append(Un_val_new)
            # Ratio R_n = T_n / T_{n-1}
            if iter_n_val > 0 and len(results_hist.get('Tn_scalar_hist', [])) >= 2:
                T_prev_new = results_hist['Tn_scalar_hist'][-2]
                R_val_new = (Tn_val_new / T_prev_new) if (np.isfinite(T_prev_new) and T_prev_new != 0) else np.nan
            else:
                R_val_new = np.nan
            results_hist.setdefault('R_T_ratio_hist', []).append(R_val_new)
        except Exception as _new_metrics_err:
            import warnings as _warnings_tmp
            _warnings_tmp.warn(f"New metrics computation failed at n={iter_n_val}: {_new_metrics_err}")
            # Keep history lengths aligned with placeholders
            results_hist.setdefault('An_fields_mid_hist', []).append(np.zeros((num_grid_points_Ng, num_grid_points_Ng)))
            results_hist.setdefault('Bn_fields_mid_hist', []).append(np.zeros((num_grid_points_Ng, num_grid_points_Ng)))
            results_hist.setdefault('Hn_fields_mid_hist', []).append(np.zeros((num_grid_points_Ng, num_grid_points_Ng)))
            results_hist.setdefault('B1n_series_hist', []).append(np.zeros_like(gX_unit_grid))
            results_hist.setdefault('B2n_series_hist', []).append(np.zeros_like(gX_unit_grid))
            results_hist.setdefault('Tn_scalar_hist', []).append(np.nan)
            results_hist.setdefault('Sn_scalar_hist', []).append(np.nan)
            results_hist.setdefault('Un_scalar_hist', []).append(np.nan)
            results_hist.setdefault('R_T_ratio_hist', []).append(np.nan)
        m21_v = _calculate_conditional_expectation(L_curr_grid, num_grid_points_Ng, axis_of_integration=1)
        m12_v = _calculate_conditional_expectation(L_curr_grid, num_grid_points_Ng, axis_of_integration=0)
        results_hist['m21_given_u1_funcs_hist'].append(m21_v)
        results_hist['m12_given_u2_funcs_hist'].append(m12_v)
        results_hist['Cn_copula_grids_hist'].append(
            get_copula_from_L_distribution(L_curr_grid, num_grid_points_Ng, Q1n_qf_v, Q2n_qf_v))

        c1n_points = find_crossing_points(L1n_cdf_v, gX_unit_grid)
        c2n_points = find_crossing_points(L2n_cdf_v, gX_unit_grid)
        results_hist['c1n_crossings_hist'].append(c1n_points)
        results_hist['c2n_crossings_hist'].append(c2n_points)

        G1n_grid_curr, G2n_grid_curr = calculate_Gn_grids(L_curr_grid, num_grid_points_Ng)
        results_hist['G1n_grids_hist'].append(G1n_grid_curr)
        results_hist['G2n_grids_hist'].append(G2n_grid_curr)

        lambda1_curve = (gX_unit_grid / wn_curr_v) * m21_v if np.isfinite(
            wn_curr_v) and wn_curr_v > 1e-9 else np.zeros_like(gX_unit_grid)
        lambda2_curve = (gX_unit_grid / wn_curr_v) * m12_v if np.isfinite(
            wn_curr_v) and wn_curr_v > 1e-9 else np.zeros_like(gX_unit_grid)
        results_hist['lambda1_hist'].append(lambda1_curve)
        results_hist['lambda2_hist'].append(lambda2_curve)

        roots_kappa1 = find_all_crossings(lambda1_curve, gX_unit_grid, 1.0)
        if not roots_kappa1: roots_kappa1 = find_all_crossings(lambda1_curve, gX_unit_grid, 1.0 - 1e-6)
        if not roots_kappa1:
            kappa_roots_lambda1_final = [0.0]
        elif len(roots_kappa1) == 1:
            kappa_roots_lambda1_final = [roots_kappa1[0]]
        else:
            kappa_roots_lambda1_final = [roots_kappa1[0], roots_kappa1[-1]]

        if len(kappa_roots_lambda1_final) < 2 or kappa2_for_lambda1_lost:
            kappa2_for_lambda1_lost = True
            final_kappas1 = kappa_roots_lambda1_final[:1]
        else:
            final_kappas1 = kappa_roots_lambda1_final
        results_hist['kappa1_roots_lambda1_hist'].append(final_kappas1[0] if final_kappas1 else np.nan)
        results_hist['kappa2_roots_lambda1_hist'].append(final_kappas1[1] if len(final_kappas1) > 1 else np.nan)

        roots_kappa2 = find_all_crossings(lambda2_curve, gX_unit_grid, 1.0)
        if not roots_kappa2: roots_kappa2 = find_all_crossings(lambda2_curve, gX_unit_grid, 1.0 - 1e-6)
        if not roots_kappa2:
            kappa_roots_lambda2_final = [0.0]
        elif len(roots_kappa2) == 1:
            kappa_roots_lambda2_final = [roots_kappa2[0]]
        else:
            kappa_roots_lambda2_final = [roots_kappa2[0], roots_kappa2[-1]]

        if len(kappa_roots_lambda2_final) < 2 or kappa2_for_lambda2_lost:
            kappa2_for_lambda2_lost = True
            final_kappas2 = kappa_roots_lambda2_final[:1]
        else:
            final_kappas2 = kappa_roots_lambda2_final
        results_hist['kappa1_roots_lambda2_hist'].append(final_kappas2[0] if final_kappas2 else np.nan)
        results_hist['kappa2_roots_lambda2_hist'].append(final_kappas2[1] if len(final_kappas2) > 1 else np.nan)

        # New 'l' calculations
        L1n_interp = interpolate.interp1d(gX_unit_grid, L1n_cdf_v, kind='linear', fill_value="extrapolate")
        results_hist['l1_lambda1_hist'].append(L1n_interp(final_kappas1[0]) if final_kappas1 else np.nan)
        results_hist['l2_lambda1_hist'].append(L1n_interp(final_kappas1[1]) if len(final_kappas1) > 1 else np.nan)

        L2n_interp = interpolate.interp1d(gX_unit_grid, L2n_cdf_v, kind='linear', fill_value="extrapolate")
        results_hist['l1_lambda2_hist'].append(L2n_interp(final_kappas2[0]) if final_kappas2 else np.nan)
        results_hist['l2_lambda2_hist'].append(L2n_interp(final_kappas2[1]) if len(final_kappas2) > 1 else np.nan)

        if iter_n_val > 0:
            prev_l1_lambda1 = results_hist['l1_lambda1_hist'][iter_n_val - 1]
            prev_l2_lambda1 = results_hist['l2_lambda1_hist'][iter_n_val - 1]
            results_hist['L1_at_l_lambda1_hist_root1'].append(
                L1n_interp(prev_l1_lambda1) if not np.isnan(prev_l1_lambda1) else np.nan)
            results_hist['L1_at_l_lambda1_hist_root2'].append(
                L1n_interp(prev_l2_lambda1) if not np.isnan(prev_l2_lambda1) else np.nan)

            prev_l1_lambda2 = results_hist['l1_lambda2_hist'][iter_n_val - 1]
            prev_l2_lambda2 = results_hist['l2_lambda2_hist'][iter_n_val - 1]
            results_hist['L2_at_l_lambda2_hist_root1'].append(
                L2n_interp(prev_l1_lambda2) if not np.isnan(prev_l1_lambda2) else np.nan)
            results_hist['L2_at_l_lambda2_hist_root2'].append(
                L2n_interp(prev_l2_lambda2) if not np.isnan(prev_l2_lambda2) else np.nan)
        else:
            for k in ['L1_at_l_lambda1_hist_root1', 'L1_at_l_lambda1_hist_root2', 'L2_at_l_lambda2_hist_root1',
                      'L2_at_l_lambda2_hist_root2']: results_hist[k].append(np.nan)

        print(
            f"  L_{iter_n_val} processed: E12_{iter_n_val}={wn_curr_v:.4f}, E1={E1_curr:.4f}, E2={E2_curr:.4f}, Spearman rho={rho_S_v:.4f}, Kendall tau={tau_K_v:.4f}")
        if iter_n_val < num_iterations_max:
            if (not np.isfinite(wn_curr_v)) or (wn_curr_v <= 1e-10):
                print(f"STOP: w_{iter_n_val}={wn_curr_v} is invalid. Halting.")
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
                            save_plots, plot_format, size_adjustment_percent, run_additional_functionalities,
                            num_labels_in_plots)
    print("\n--- Script Execution Finished ---")


def _get_validated_input(prompt_text, default_value, type_caster, validation_func=None, error_msg="Invalid input."):
    while True:
        try:
            user_val_str = input(f"{prompt_text} [default: {default_value}]: ")
            value_to_test = default_value if not user_val_str else type_caster(user_val_str)
            if validation_func:
                constraint_ok, constraint_desc_str = False, "passes constraint"
                if isinstance(validation_func, tuple) and callable(validation_func[0]):
                    actual_validation_lambda, constraint_desc_str = validation_func
                    constraint_ok = actual_validation_lambda(value_to_test)
                elif callable(validation_func):
                    constraint_ok = validation_func(value_to_test)
                if not constraint_ok: print(f"{error_msg} (Constraint: {constraint_desc_str}). Try again."); continue
            return value_to_test
        except ValueError:
            print(f"Invalid type. Expected input convertible to '{type_caster.__name__}'.")
        except Exception as e:
            print(f"Unexpected input error: {e}")


def get_user_parameters():
    DEFAULT_NUM_ITERATIONS = 10
    DEFAULT_GRID_SIZE_NG = 100
    DEFAULT_LABELS_IN_PLOTS = 5
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
    AVAILABLE_MARGINALS = {'uniform': {'params': [], 'defaults': [], 'constraints': [], 'desc': "U(0,1)"},
                           'pareto': {'params': ['b'], 'defaults': [2.], 'constraints': [(lambda b: b > 0, "b>0")],
                                      'desc': "Pareto (Lomax, b>0)"},
                           'lognormal': {'params': ['s', 'scale'], 'defaults': [1., 1.],
                                         'constraints': [(lambda s: s > 0, "s>0"), (lambda sc: sc > 0, "scale>0")],
                                         'desc': "LogN(s>0,scale>0)"},
                           'gamma': {'params': ['a', 'scale'], 'defaults': [2., 1.],
                                     'constraints': [(lambda a: a > 0, "a>0"), (lambda sc: sc > 0, "scale>0")],
                                     'desc': "Gamma(a>0,scale>0)"}, 'beta': {'params': ['a', 'b'], 'defaults': [2., 2.],
                                                                             'constraints': [(lambda a: a > 0, "a>0"),
                                                                                             (lambda b: b > 0, "b>0")],
                                                                             'desc': "Beta(a>0,b>0)"},
                           'sinewave': {'params': ['k', 'A'], 'defaults': [3, 0.5],
                                        'constraints': [(lambda k: isinstance(k, int) and k >= 1, "k:int>=1"),
                                                        (lambda A: -1 <= A <= 1, "A:[-1,1]")],
                                        'desc': "SineWave(k_int>=1,A_amp[-1,1])"},
                           'betamix': {'params': ['a1', 'b1', 'a2', 'b2', 'w'], 'defaults': [2., 5., 5., 2., 0.5],
                                       'constraints': [(lambda p: p > 0, "val>0")] * 4 + [
                                           (lambda w: 0 < w < 1, "0<w<1")], 'desc': "BetaMix(Beta_params>0;0<w<1)"}}
    AVAILABLE_COPULAS = {'independent': {'param_names': [], 'default_param_vals': None, 'constraints': []},
                         'gaussian': {'param_names': ['rho'], 'default_param_vals': [DEFAULT_COPULA_PARAM_GAUSS],
                                      'constraints': [(lambda r: -1 <= r <= 1, "rho:[-1,1]")]},
                         'clayton': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_CLAYTON],
                                     'constraints': [(lambda t: t >= -1 and abs(t) > 1e-9, "theta >= -1, non-zero")]},
                         'gumbel': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_GUMBEL],
                                    'constraints': [(lambda t: t >= 1, "theta>=1")]},
                         'frank': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_FRANK],
                                   'constraints': [(lambda t: abs(t) > 1e-9, "theta!=0")]},
                         't': {'param_names': ['rho', 'nu'],
                               'default_param_vals': [DEFAULT_COPULA_PARAM_T['rho'], DEFAULT_COPULA_PARAM_T['nu']],
                               'constraints': [(lambda r: -1 <= r <= 1, "rho:[-1,1]"), (lambda nu: nu > 0, "nu>0")]},
                         'amh': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_AMH_THETA],
                                 'constraints': [(lambda t: -1 <= t <= 1 and abs(t) > 1e-9, "theta:[-1,1], !=0")]},
                         'frechetmix': {'param_names': ['alpha'],
                                        'default_param_vals': [DEFAULT_COPULA_PARAM_FRECHETMIX_ALPHA],
                                        'constraints': [(lambda a: 0 <= a <= 1, "alpha:[0,1]")]},
                         'fgm': {'param_names': ['theta'], 'default_param_vals': [DEFAULT_COPULA_PARAM_FGM_THETA],
                                 'constraints': [(lambda t: -1 <= t <= 1, "theta:[-1,1]")]}}

    print("--- Setup Simulation Parameters ---")
    num_iter = _get_validated_input("Number of L_n iterations N (L0 to LN)", DEFAULT_NUM_ITERATIONS, int,
                                    (lambda n: n >= 0, "N>=0"))
    grid_ng = _get_validated_input("Grid size Ng for discretization (e.g., 50-150)", DEFAULT_GRID_SIZE_NG, int,
                                   (lambda n: n >= 10, "Ng>=10"))
    num_labels = _get_validated_input("How many iterations to label in plots?", DEFAULT_LABELS_IN_PLOTS, int,
                                      (lambda n: n >= 0, "Labels>=0"))
    show_titles = _get_validated_input("Plot with titles? (yes/no)", "yes", str,
                                       lambda s: s.lower() in ['yes', 'no']) == 'yes'
    save_plots = _get_validated_input("Do you want to save the plots? (yes/no)", "no", str,
                                      lambda s: s.lower() in ['yes', 'no']) == 'yes'
    plot_format = None
    if save_plots:
        supported_formats = plt.figure().canvas.get_supported_filetypes()
        plt.close()
        rec_fmts = "Recommended: jpg, png, pdf, eps"
        while True:
            user_format = input(f"Enter plot format ({rec_fmts}) [default: jpg]: ").lower() or 'jpg'
            if user_format in supported_formats:
                plot_format = user_format;
                break
            else:
                print(f"Error: Format '{user_format}' is not supported. Supported: {list(supported_formats.keys())}")
    size_options = {'-30%': -30, '-20%': -20, '-10%': -10, 'benchmark': 0, '+10%': 10, '+20%': 20, '+30%': 30}
    size_prompt = f"Choose plot size from benchmark ({list(size_options.keys())})"
    while True:
        choice = input(f"{size_prompt} [default: benchmark]: ").lower() or 'benchmark'
        if choice in size_options:
            size_adj = size_options[choice];
            break
        else:
            print("Invalid choice.")
    run_additional = _get_validated_input("Run and plot additional functionalities (Plots 22 onwards)? (yes/no)", "no",
                                          str, lambda s: s.lower() in ['yes', 'no']) == 'yes'

    def _get_single_marginal_config(idx):
        print(f"\n--- Configuring Marginal Distribution F{idx} ---")
        print("Available choices (name: description):")
        [print(f"  {k}: {v['desc']}") for k, v in AVAILABLE_MARGINALS.items()]
        default = DEFAULT_MARGINAL_1_NAME if idx == 1 else DEFAULT_MARGINAL_2_NAME
        name = _get_validated_input(f"Choose marginal F{idx} name", default, str,
                                    lambda n: n.lower() in AVAILABLE_MARGINALS).lower()
        info = AVAILABLE_MARGINALS[name]
        params = {}
        if info['params']:
            print(f"Parameters for {name} ({info['desc']}):")
            for i, p_name in enumerate(info['params']):
                p_prompt = f"  Enter param '{p_name}' ({info['constraints'][i][1]})"
                p_caster = int if name == 'sinewave' and p_name == 'k' else float
                params[p_name] = _get_validated_input(p_prompt, info['defaults'][i], p_caster, info['constraints'][i])
        return name, params

    m1_name, m1_params = _get_single_marginal_config(1)
    m2_name, m2_params = _get_single_marginal_config(2)
    print("\n--- Configuring Initial Copula C_init ---")
    print("Available choices: " + ', '.join(AVAILABLE_COPULAS.keys()))
    cop_name = _get_validated_input("Choose copula type", DEFAULT_COPULA_NAME, str,
                                    lambda n: n.lower() in AVAILABLE_COPULAS).lower()
    cop_info = AVAILABLE_COPULAS[cop_name]
    cop_param = None
    if cop_info['param_names']:
        print(f"Parameters for {cop_name}:")
        temp_params = [_get_validated_input(f"  Enter param '{p_name}' ({cop_info['constraints'][i][1]})",
                                            cop_info['default_param_vals'][i], float, cop_info['constraints'][i]) for
                       i, p_name in enumerate(cop_info['param_names'])]
        if len(cop_info['param_names']) == 1:
            cop_param = temp_params[0]
        else:
            cop_param = dict(zip(cop_info['param_names'], temp_params))
    else:
        cop_param = cop_info['default_param_vals']

    return num_iter, grid_ng, m1_name, m1_params, m2_name, m2_params, cop_name, cop_param, \
        show_titles, save_plots, plot_format, size_adj, run_additional, num_labels


# --- Main Execution Script ---
if __name__ == "__main__":
    try:
        (N_ITER_MAX, GRID_NG_USER, M1_NAME, M1_PARAMS, M2_NAME, M2_PARAMS, COP_NAME_INIT, COP_PARAM_INIT,
         SHOW_TITLES_USER, SAVE_PLOTS_USER, PLOT_FORMAT_USER, SIZE_ADJUST_USER, RUN_ADDITIONAL_USER,
         NUM_LABELS_USER) = get_user_parameters()

        run_simulation_main_loop(N_ITER_MAX, M1_NAME, M1_PARAMS, M2_NAME, M2_PARAMS, COP_NAME_INIT, COP_PARAM_INIT,
                                 GRID_NG_USER,
                                 SHOW_TITLES_USER, SAVE_PLOTS_USER, PLOT_FORMAT_USER, SIZE_ADJUST_USER,
                                 RUN_ADDITIONAL_USER, NUM_LABELS_USER)
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted by user (Ctrl+C) ---")
    except Exception as main_exec_err:
        print(f"\n--- CRITICAL ERROR IN MAIN SCRIPT EXECUTION ---")
        print(f"Error Type: {type(main_exec_err).__name__}")
        print(f"Error Message: {main_exec_err}")
        print("Traceback follows:")
        traceback.print_exc()
        print("--- SCRIPT TERMINATED DUE TO ERROR ---")