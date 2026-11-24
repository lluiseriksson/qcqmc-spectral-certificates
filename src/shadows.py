"""
Authentic Born-rule quantum classical shadows (global Haar 3-design on C|0> (+) H_D),
distribution-free Median-of-Means (MoM) + Chebyshev finite-sample confidence boxes,
and Prediction-Powered Inference (PPI) surrogate control-variate debiasing.
"""
import math
import numpy as np


def sample_haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Exact Haar-random unitary in U(dim) via Mezzadri's QR algorithm."""
    Z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph[np.newaxis, :]


def simulate_linear_shadow_moments(psi_sur: np.ndarray, psi_T: np.ndarray,
                                   T_mat: np.ndarray, K_max: int,
                                   n_snapshots: int, rng: np.random.Generator):
    """
    Simulate authentic non-Gaussian Born-rule classical shadows of
      |chi> = (|0> + |Psi_T>) / sqrt(2) in C|0> (+) H_D
    and compute single-shot unbiased estimators of Re(b_k) = Re(<Psi_sur | T^k | Psi_T>)
    for k = 0, ..., K_max:
      bhat_k^(c) = 2 (D + 2) Re(<0 | U_c^dagger |b_c> <b_c | U_c | u_k>),
    where |u_k> = T^k |Psi_sur>.

    Exact 3-design properties (d = D + 1):
      1. Unbiased: E[bhat_k^(c)] = Re(<Psi_sur | T^k | Psi_T>).
      2. Deterministic bound: |bhat_k^(c)| <= (D + 2) * sqrt(a_{2k}) almost surely.
      3. Exact second moment: E[(bhat_k^(c))^2] = ((D + 2)/(D + 3)) * (3 a_{2k} + |b_k|^2) <= 4 a_{2k}
         (strict whenever a_{2k} > 0) for arbitrary complex (|Psi_sur>, |Psi_T>), and
         Var(bhat_k^(c)) <= 3 a_{2k} when b_k is real.
    """
    D = len(psi_T)
    dim = D + 1
    chi = np.zeros(dim, dtype=np.complex128)
    chi[0] = 1.0 / np.sqrt(2.0)
    chi[1:] = psi_T / np.sqrt(2.0)

    U_vecs = np.zeros((K_max + 1, dim), dtype=np.complex128)
    cur = psi_sur.astype(np.complex128).copy()
    for k in range(K_max + 1):
        U_vecs[k, 1:] = cur
        cur = T_mat @ cur

    samples = np.zeros((n_snapshots, K_max + 1), dtype=np.float64)
    scale = 2.0 * (D + 2)
    for c in range(n_snapshots):
        Uc = sample_haar_unitary(dim, rng)
        amps = Uc @ chi
        probs = np.abs(amps) ** 2
        probs /= probs.sum()
        b_idx = int(rng.choice(dim, p=probs))
        row = Uc[b_idx, :]
        ov0 = np.conj(row[0])
        ov_uk = U_vecs @ row
        samples[c, :] = scale * np.real(ov0 * ov_uk)
    return samples


def exact_shadow_variance_bound(a_even: list, complex_states: bool = True) -> np.ndarray:
    """
    Exact dimension-free variance upper bound for single-shot
      bhat_k^(c) = 2 (D + 2) Re(<0|U^H|b><b|U|u_k>)
    under global Haar (3-design) shadows on C^{D+1}.
    By exact 3-design Weingarten calculus with O_k = |u_k><0| + |0><u_k|
    (tr(O_k) = 0, tr(O_k^2) = 2 a_{2k}, tr(rho_chi O_k^2) = (a_{2k} + |b_k|^2)/2):
      E[(bhat_k^(c))^2] = ((D + 2)/(D + 3)) * (3 a_{2k} + |b_k|^2) <= 4 a_{2k}
    for all complex states (since |b_k|^2 <= a_{2k} by Cauchy-Schwarz), and
      Var(bhat_k^(c)) = E[(bhat_k^(c))^2] - (Re b_k)^2 <= 3 a_{2k}
    whenever b_k is real. The universal complex-safe bound 4 a_{2k} is returned by default.
    """
    factor = 4.0 if complex_states else 3.0
    return np.array([factor * float(a_2k) for a_2k in a_even], dtype=np.float64)


def mom_confidence_box(samples: np.ndarray, var_bounds: np.ndarray, alpha: float):
    """
    Exact, non-asymptotic, distribution-free Median-of-Means (MoM) confidence box
    for K+1 coordinates simultaneously with joint coverage >= 1 - alpha (alpha in (0, 1)):
      B = max(3, ceil(8 * ln((K+1) / alpha))) (rounded up to odd), m = floor(C / B) >= 1.
      On each block, Chebyshev gives P(|bar_b_{j,k} - mu_k| > 2 sqrt(sigma_k^2 / m)) <= 1/4.
      By Hoeffding on Bin(B, 1/4) >= B/2, P(|median - mu_k| > eps_k) <= exp(-B/8) <= alpha/(K+1).
      Union bound over K+1 coordinates gives joint coverage >= 1 - alpha.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    C, n_coords = samples.shape
    B = max(3, int(math.ceil(8.0 * math.log(n_coords / alpha))))
    if B % 2 == 0:
        B += 1
    m = C // B
    assert m >= 1, f"Not enough snapshots C={C} for B={B} MoM blocks"
    trimmed = samples[:B * m, :].reshape(B, m, n_coords)
    block_means = trimmed.mean(axis=1)  # B x n_coords
    center = np.median(block_means, axis=0)
    radii = 2.0 * np.sqrt(var_bounds / float(m))
    return center, radii, {"B": B, "m": m, "used_snapshots": B * m}


def ppi_walker_estimator(g_all: np.ndarray, f_sur_all: np.ndarray,
                         cal_idx: np.ndarray, vhat_cal: np.ndarray):
    """
    Proposition 4.2(ii): Prediction-Powered Inference (PPI) control-variate estimator
    for a walker-averaged linear overlap functional h = (1/M) sum_{a=1}^M g_a v_a:
      h_PPI = (1/M) sum_a g_a f_sur_a + (1/m_cal) sum_{r=1}^{m_cal} g_{a_r} (vhat_{a_r} - f_sur_{a_r}),
    where (a_1, ..., a_{m_cal}) are i.i.d. uniform draws (with replacement) from {1,...,M}
    paired with independent unbiased shadow estimates vhat_{a_r}.
    """
    sur_only = float(np.mean(g_all * f_sur_all))
    sub_raw = float(np.mean(g_all[cal_idx] * vhat_cal))
    ppi_est = sur_only + float(np.mean(g_all[cal_idx] * (vhat_cal - f_sur_all[cal_idx])))
    return {"ppi": ppi_est, "sub_raw": sub_raw, "sur_only": sur_only}
