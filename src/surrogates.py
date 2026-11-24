"""
Backflow Neural Quantum Surrogates (Jiang et al. Eq. S2.19-S2.22),
Wirtinger Green's defect G^{dbar}, Plucker-kernel distance eta_Pl,
and holomorphic projection restoration.
"""
import numpy as np
from scipy.linalg import expm
from .hubbard import slater_vec


def projector(Phi: np.ndarray) -> np.ndarray:
    """Orthogonal projector P_Phi = Phi (Phi^H Phi)^{-1} Phi^H on Gr(eta, N)."""
    return Phi @ np.linalg.inv(Phi.conj().T @ Phi) @ Phi.conj().T


def backflow_overlap(Phi: np.ndarray, G0: np.ndarray, A_bf: np.ndarray,
                     M_obs: np.ndarray, kappa: float) -> complex:
    """
    Minimal smooth GL(eta, C)-equivariant Backflow NQS:
      f_kappa(Phi) = det[(G0 + kappa * tr(M_obs P_Phi) * A_bf)^dagger Phi].
    """
    P = projector(Phi)
    s = np.trace(M_obs @ P)
    G = G0 + kappa * s * A_bf
    return complex(np.linalg.det(G.conj().T @ Phi))


def wirtinger_dbar(f, Phi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical Wirtinger anti-holomorphic derivative df / dPhi^*."""
    N, eta = Phi.shape
    W_bar = np.zeros((N, eta), dtype=np.complex128)
    for p in range(N):
        for k in range(eta):
            E = np.zeros((N, eta), dtype=np.complex128)
            E[p, k] = 1.0
            df_dx = (f(Phi + eps * E) - f(Phi - eps * E)) / (2.0 * eps)
            df_dy = (f(Phi + 1j * eps * E) - f(Phi - 1j * eps * E)) / (2.0 * eps)
            W_bar[p, k] = 0.5 * (df_dx + 1j * df_dy)
    return W_bar


def antiholomorphic_greens_defect(f, Phi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Exact second term of Jiang et al. Eq. (S2.22):
      G^{dbar}_{pq}(Phi) = -(1 / f(Phi)) sum_k (df / dPhi^*_{pk}) Phi^*_{qk}.
    Vanishes iff df/dPhi^*(Phi) = 0 (since Phi has full column rank).
    """
    W_bar = wirtinger_dbar(f, Phi, eps=eps)
    return -(1.0 / f(Phi)) * (W_bar @ Phi.conj().T)


def lie_D1(f, Phi: np.ndarray, B: np.ndarray, eps: float = 1e-5) -> complex:
    """4th-order central difference for D_B[f](Phi) = -i d/ds f(e^{isB} Phi)|_{s=0}."""
    fp1 = f(expm(1j * eps * B) @ Phi)
    fm1 = f(expm(-1j * eps * B) @ Phi)
    fp2 = f(expm(2j * eps * B) @ Phi)
    fm2 = f(expm(-2j * eps * B) @ Phi)
    return complex(-1j * (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * eps))


def lie_D2(f, Phi: np.ndarray, B: np.ndarray, eps: float = 1e-4) -> complex:
    """4th-order central difference for D_B^2[f](Phi) = -d^2/ds^2 f(e^{isB} Phi)|_{s=0}."""
    f0 = f(Phi)
    fp1 = f(expm(1j * eps * B) @ Phi)
    fm1 = f(expm(-1j * eps * B) @ Phi)
    fp2 = f(expm(2j * eps * B) @ Phi)
    fm2 = f(expm(-2j * eps * B) @ Phi)
    return complex(-(-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * eps**2))


def surrogate_local_energy_num(f, Phi: np.ndarray, B0: np.ndarray, B_list) -> complex:
    """Evaluate L_H[f](Phi) = D_{B0}[f](Phi) + sum_gamma c_gamma D_{B_gamma}^2[f](Phi)."""
    val = lie_D1(f, Phi, B0)
    for c, B in B_list:
        val += c * lie_D2(f, Phi, B)
    return val


def plucker_distance_and_projection(f, walkers):
    """
    Theorem 2A: compute exact Plucker-kernel distance eta_Pl = ||(I - S S^+) f_S^*||_2
    and the unique minimum-norm physical projection |Psi_f^{proj}> = W S^+ f_S^*.
    """
    W = np.column_stack([slater_vec(Ph)[0] for Ph in walkers])  # D x m
    S = W.conj().T @ W
    S_pinv = np.linalg.pinv(S, rcond=1e-12)
    f_S = np.array([f(Ph) for Ph in walkers], dtype=np.complex128)
    m = len(walkers)
    residual = (np.eye(m) - S @ S_pinv) @ f_S.conj()
    eta_pl = float(np.linalg.norm(residual))
    psi_proj = W @ S_pinv @ f_S.conj()
    v_proj = (W.conj().T @ psi_proj).conj()
    return {
        "eta_pl": eta_pl,
        "dist_to_proj": float(np.linalg.norm(f_S - v_proj)),
        "psi_proj": psi_proj,
        "f_S": f_S,
        "v_proj": v_proj,
        "S": S,
        "W": W,
    }
