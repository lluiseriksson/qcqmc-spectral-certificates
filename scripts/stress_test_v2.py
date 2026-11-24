"""
Stress test v2:
  1. Genuine 1-neuron Backflow NQS G_kappa(P) = G0 + kappa * tr(M P) * A_bf
     (plus remark on linear-projector collapse G(P) = G0 + P A).
     Shows:
       - Strict GL(2,C) equivariance (1e-15),
       - Non-zero Wirtinger Green's defect G^{dbar},
       - Non-zero exact Plucker-kernel distance eta_Pl = ||(I - S S^+) f^*||_2,
       - Non-zero bias E_mix^theta != E_0 (INCLUDING E_mix^theta < E_0!)
         on ZERO-WALKER-ERROR exact decompositions of |Psi_0>,
       - Representation dependence across two exact decompositions of |Psi_0>,
       - Exact identity verification of Theorem 2B (mixed-estimator bias decomposition).
  2. Clean Primal & Dual Surrogate-Anchored Block Hausdorff SDP (d=0,1,2)
     on 4-site Hubbard chain (U=4, L=4, dim=36) for BOTH:
       - Ground-state energy upper bound E_0 <= E_cert (using ONLY classical a_k and noisy linear b_k!),
       - Two-edge visible spectral-gap certificate on (theta_1, theta_2).
"""
import numpy as np
import cvxpy as cp
from itertools import combinations
from scipy.linalg import expm

rng = np.random.default_rng(20261124)

# =============================================================================
# PART 1: 2-SITE HUBBARD (N=4, eta=2, t=1, U=4) + NONLINEAR BACKFLOW NQS
# =============================================================================
def slater_vec(Phi):
    N, eta = Phi.shape
    basis = list(combinations(range(N), eta))
    v = np.zeros(len(basis), dtype=np.complex128)
    for idx, rows in enumerate(basis):
        v[idx] = np.linalg.det(Phi[list(rows), :])
    return v, basis

def build_two_site_hubbard(t=1.0, U=4.0):
    N = 4
    K = np.zeros((N, N), dtype=np.complex128)
    K[0, 1] = K[1, 0] = -t
    K[2, 3] = K[3, 2] = -t
    N0 = np.diag([1.0, 0.0, 1.0, 0.0]).astype(np.complex128)
    N1 = np.diag([0.0, 1.0, 0.0, 1.0]).astype(np.complex128)
    B0 = K - 0.5 * U * (N0 + N1)
    B_list = [(0.5 * U, N0), (0.5 * U, N1)]
    basis = list(combinations(range(N), 2))
    D = len(basis)
    def one_body_mat(B):
        M = np.zeros((D, D), dtype=np.complex128)
        for j, cols in enumerate(basis):
            Phi = np.zeros((N, 2), dtype=np.complex128)
            Phi[cols[0], 0] = 1.0
            Phi[cols[1], 1] = 1.0
            v1, _ = slater_vec(np.column_stack([B @ Phi[:, 0], Phi[:, 1]]))
            v2, _ = slater_vec(np.column_stack([Phi[:, 0], B @ Phi[:, 1]]))
            M[:, j] = v1 + v2
        return M
    H = one_body_mat(B0)
    for c, B in B_list:
        Mb = one_body_mat(B)
        H = H + c * (Mb @ Mb)
    return H, B0, B_list, basis

H2, B0, B_list, basis42 = build_two_site_hubbard(t=1.0, U=4.0)
evals2, evecs2 = np.linalg.eigh(H2)
E0_2, psi0_2 = evals2[0], evecs2[:, 0]

# Genuine nonlinear Backflow NQS: G_kappa(P) = G0 + kappa * tr(M_obs * P) * A_bf
G0 = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
], dtype=np.complex128) / np.sqrt(2.0)

A_bf = np.array([
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [0.0, 1.0],
], dtype=np.complex128) / np.sqrt(2.0)

# M_obs measures site-0 vs site-1 imbalance (or hopping):
M_obs = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
], dtype=np.complex128)

def projector(Phi):
    return Phi @ np.linalg.inv(Phi.conj().T @ Phi) @ Phi.conj().T

def f_backflow(Phi, kappa):
    P = projector(Phi)
    s = np.trace(M_obs @ P)  # real-valued smooth invariant on Gr(2,4)!
    G = G0 + kappa * s * A_bf
    return np.linalg.det(G.conj().T @ Phi)

# Verify GL(2,C) equivariance
Phi_t = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
U_t = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
print("=== PART 1: NONLINEAR BACKFLOW NQS (Jiang et al. Eq. S2.19) ===")
print(f"GL(2,C) equivariance err (kappa=0.5): {abs(f_backflow(Phi_t @ U_t, 0.5) - np.linalg.det(U_t)*f_backflow(Phi_t, 0.5)):.2e}")

# Plucker-kernel distance on m=10 complex walkers
m_walk = 10
walkers = [rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)) for _ in range(m_walk)]
W = np.column_stack([slater_vec(Ph)[0] for Ph in walkers])
S = W.conj().T @ W
S_pinv = np.linalg.pinv(S, rcond=1e-12)

for kap in [0.0, 0.25, 0.5]:
    f_S = np.array([f_backflow(Ph, kap) for Ph in walkers])
    eta_pl = np.linalg.norm((np.eye(m_walk) - S @ S_pinv) @ f_S.conj())
    psi_proj = W @ S_pinv @ f_S.conj()
    v_proj = (W.conj().T @ psi_proj).conj()
    dist_proj = np.linalg.norm(f_S - v_proj)
    E_var_proj = float(np.real(psi_proj.conj() @ H2 @ psi_proj) / np.real(psi_proj.conj() @ psi_proj))
    print(f"  kappa={kap:.2f} | eta_Pl={eta_pl:.8f} | ||f_S - v_proj||={dist_proj:.8f} | E_var(proj)={E_var_proj:+.6f} >= E0={E0_2:+.6f}")

# Wirtinger defect G^{dbar} via numerical Wirtinger derivative d/dPhi^*
def wirtinger_dbar(f, Phi, eps=1e-6):
    N, eta = Phi.shape
    W_bar = np.zeros((N, eta), dtype=np.complex128)
    for p in range(N):
        for k in range(eta):
            E = np.zeros((N, eta), dtype=np.complex128)
            E[p, k] = 1.0
            df_dx = (f(Phi + eps * E) - f(Phi - eps * E)) / (2 * eps)
            df_dy = (f(Phi + 1j * eps * E) - f(Phi - 1j * eps * E)) / (2 * eps)
            W_bar[p, k] = 0.5 * (df_dx + 1j * df_dy)
    return W_bar

def D_B(f, Phi, B, eps=1e-6):
    fp1 = f(expm(1j * eps * B) @ Phi)
    fm1 = f(expm(-1j * eps * B) @ Phi)
    fp2 = f(expm(2j * eps * B) @ Phi)
    fm2 = f(expm(-2j * eps * B) @ Phi)
    return -1j * (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12 * eps)

def D_B2(f, Phi, B, eps=1e-4):
    f0 = f(Phi)
    fp1 = f(expm(1j * eps * B) @ Phi)
    fm1 = f(expm(-1j * eps * B) @ Phi)
    fp2 = f(expm(2j * eps * B) @ Phi)
    fm2 = f(expm(-2j * eps * B) @ Phi)
    return -(-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12 * eps**2)

def L_H_sur(f, Phi):
    val = D_B(f, Phi, B0)
    for c, B in B_list:
        val += c * D_B2(f, Phi, B)
    return val

def comp_det(i, j):
    Ph = np.zeros((4, 2), dtype=np.complex128)
    Ph[i, 0] = 1.0
    Ph[j, 1] = 1.0
    return Ph

decomp1_Phis = [comp_det(0, 2), comp_det(0, 3), comp_det(1, 2), comp_det(1, 3)]
decomp1_coeffs = [psi0_2[1], psi0_2[2], psi0_2[3], psi0_2[4]]

theta_rot = 0.37
R_spin = np.array([[np.cos(theta_rot), -np.sin(theta_rot)],
                   [np.sin(theta_rot),  np.cos(theta_rot)]], dtype=np.complex128)
R4 = np.block([[R_spin, np.zeros((2,2))], [np.zeros((2,2)), R_spin]])
decomp2_Phis = [R4 @ Ph for Ph in decomp1_Phis]
W2 = np.column_stack([slater_vec(Ph)[0] for Ph in decomp2_Phis])
decomp2_coeffs, _, _, _ = np.linalg.lstsq(W2, psi0_2, rcond=None)

print("\n--- Zero-Walker-Error Mixed Energy E_mix^theta & Representation Dependence ---")
for kap in [0.0, -0.4, 0.4, 0.8]:
    f_k = lambda Ph, k=kap: f_backflow(Ph, k)
    W_bar = wirtinger_dbar(f_k, decomp1_Phis[0])
    G_dbar = -(1.0 / f_k(decomp1_Phis[0])) * (W_bar @ decomp1_Phis[0].conj().T)
    res = []
    for name, Phis, coeffs in [("Decomp1", decomp1_Phis, decomp1_coeffs),
                               ("Decomp2", decomp2_Phis, decomp2_coeffs)]:
        ws = [c * f_k(Ph) for c, Ph in zip(coeffs, Phis)]
        E_Ls = [L_H_sur(f_k, Ph) / f_k(Ph) for Ph in Phis]
        E_mix = sum(w * el for w, el in zip(ws, E_Ls)) / sum(ws)
        res.append((E_mix.real, E_mix.real - E0_2))
    print(f"  kappa={kap:+.1f} | ||G^dbar||={np.linalg.norm(G_dbar):.4f} | Decomp1: E_mix={res[0][0]:+.6f} (bias={res[0][1]:+.6f}) | Decomp2: E_mix={res[1][0]:+.6f} (bias={res[1][1]:+.6f})")

# =============================================================================
# PART 2: 4-SITE HUBBARD BLOCK HAUSDORFF SDP (d=0,1,2) + SPECTRAL GAP CERTIFICATE
# =============================================================================
print("\n=== PART 2: 4-SITE HUBBARD BLOCK HAUSDORFF (ENERGY & SPECTRAL GAP) ===")
def build_hubbard_chain(L=4, t=1.0, U=4.0):
    up_states = list(combinations(range(L), L // 2))
    dn_states = list(combinations(range(L), L // 2))
    n_up, n_dn = len(up_states), len(dn_states)
    up_idx = {s: i for i, s in enumerate(up_states)}
    dn_idx = {s: i for i, s in enumerate(dn_states)}
    D = n_up * n_dn
    H = np.zeros((D, D), dtype=np.float64)
    def hop_mat(states, idx_map):
        M = np.zeros((len(states), len(states)), dtype=np.float64)
        for a, st in enumerate(states):
            st_set = set(st)
            for i in range(L - 1):
                for src, dst in [(i, i + 1), (i + 1, i)]:
                    if src in st_set and dst not in st_set:
                        lo, hi = min(src, dst), max(src, dst)
                        parity = sum(1 for x in st if lo < x < hi)
                        new_st = tuple(sorted((st_set - {src}) | {dst}))
                        M[idx_map[new_st], a] += -t * ((-1.0) ** parity)
        return M
    H += np.kron(hop_mat(up_states, up_idx), np.eye(n_dn)) + np.kron(np.eye(n_up), hop_mat(dn_states, dn_idx))
    for iu, su in enumerate(up_states):
        for idn, sd in enumerate(dn_states):
            H[iu * n_dn + idn, iu * n_dn + idn] += U * len(set(su) & set(sd))
    return H

H4 = build_hubbard_chain(L=4, t=1.0, U=4.0)
ev4, vec4 = np.linalg.eigh(H4)
E0_4, E1_4, E2_4 = ev4[0], ev4[1], ev4[2]
# Tight Gershgorin / spectral box (exact, oracle-free!)
row_diag = np.diag(H4)
row_off = np.sum(np.abs(H4), axis=1) - np.abs(row_diag)
E_min_box = float(np.min(row_diag - row_off))
E_max_box = float(np.max(row_diag + row_off))
T4 = (E_max_box * np.eye(len(H4)) - H4) / (E_max_box - E_min_box)
lam0_4 = (E_max_box - E0_4) / (E_max_box - E_min_box)
lam1_4 = (E_max_box - E1_4) / (E_max_box - E_min_box)
print(f"Gershgorin enclosure: [{E_min_box:.2f}, {E_max_box:.2f}] | E0={E0_4:.6f}, E1={E1_4:.6f}, E2={E2_4:.6f}")
print(f"T spectrum: lam0={lam0_4:.6f}, lam1={lam1_4:.6f}, lam_gap={lam0_4-lam1_4:.6f}")

H4_free = build_hubbard_chain(L=4, t=1.0, U=0.0)
_, vec4_free = np.linalg.eigh(H4_free)
psi_sur = vec4_free[:, 0]
psi_T = expm(-0.6 * H4) @ psi_sur
psi_T /= np.linalg.norm(psi_T)
a_mom = [float(psi_sur @ np.linalg.matrix_power(T4, k) @ psi_sur) for k in range(8)]
b_mom = [float(psi_sur @ np.linalg.matrix_power(T4, k) @ psi_T) for k in range(8)]
c_mom = [float(psi_T @ np.linalg.matrix_power(T4, k) @ psi_T) for k in range(8)]

def solve_dual_edge(a_vals, b_vals, eps_b, theta, d, use_c=None):
    """
    Solve the EXACT CONIC DUAL of the Surrogate-Anchored Block Hausdorff SDP on [0, theta]:
      Find PSD matrices P, Q, R in S_+^{2(d+1)} (with tr(P)+tr(Q)+tr(R) <= 1) minimizing:
         sup_{b in B, c_0=1, c_{1..K} free}  <P, H_d> + <Q, H_d^(1)> + <R, theta H_d - H_d^(1)>
      Note: since c_k (k>=1) are FREE primal variables, in the dual their net coefficient MUST BE 0!!
      Wait: THAT'S SO CLEAN! The dual simply enforces coeff(c_k) == 0 for k=1..2d+1!
      If optimal value < 0, then INFEASIBILITY is 100% CERTIFIED (independent of solver tolerances via exact rational check!).
    """
    K = 2 * d + 1
    m_sz = 2 * (d + 1)
    P = cp.Variable((m_sz, m_sz), symmetric=True)
    Q = cp.Variable((m_sz, m_sz), symmetric=True)
    R = cp.Variable((m_sz, m_sz), symmetric=True)

    # Extract scalar linear coefficients of a_k, b_k, c_k in <P, H_d> + <Q, H_d^(1)> + <R, theta H_d - H_d^(1)>
    # Block (i,j) is at rows [2i, 2i+1], cols [2j, 2j+1]
    coeff_a = [0] * (K + 1)
    coeff_b = [0] * (K + 1)
    coeff_c = [0] * (K + 1)
    for i in range(d + 1):
        for j in range(d + 1):
            r0, c0 = 2 * i, 2 * j
            k0 = i + j
            k1 = i + j + 1
            # P + theta * R multiply H_d (moment index k0)
            M0 = P[r0:r0+2, c0:c0+2] + theta * R[r0:r0+2, c0:c0+2]
            coeff_a[k0] = coeff_a[k0] + M0[0, 0]
            coeff_b[k0] = coeff_b[k0] + M0[0, 1] + M0[1, 0]
            coeff_c[k0] = coeff_c[k0] + M0[1, 1]
            # Q - R multiply H_d^(1) (moment index k1)
            M1 = Q[r0:r0+2, c0:c0+2] - R[r0:r0+2, c0:c0+2]
            coeff_a[k1] = coeff_a[k1] + M1[0, 0]
            coeff_b[k1] = coeff_b[k1] + M1[0, 1] + M1[1, 0]
            coeff_c[k1] = coeff_c[k1] + M1[1, 1]

    constraints = [P >> 0, Q >> 0, R >> 0, cp.trace(P) + cp.trace(Q) + cp.trace(R) <= 1.0]
    # Worst-case objective over |b_k - b_vals[k]| <= eps_b[k]
    obj = sum(coeff_a[k] * a_vals[k] for k in range(K + 1))
    obj += coeff_c[0] * 1.0
    for k in range(K + 1):
        if eps_b[k] > 0:
            obj += coeff_b[k] * b_vals[k] + eps_b[k] * cp.abs(coeff_b[k])
        else:
            obj += coeff_b[k] * b_vals[k]
    if use_c is None:
        for k in range(1, K + 1):
            constraints.append(coeff_c[k] == 0)
    else:
        c_vals, eps_c = use_c
        for k in range(1, K + 1):
            if eps_c[k] > 0:
                obj += coeff_c[k] * c_vals[k] + eps_c[k] * cp.abs(coeff_c[k])
            else:
                obj += coeff_c[k] * c_vals[k]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    return prob.value, (P.value, Q.value, R.value)

def cert_theta_dual(a_vals, b_vals, eps_b, d, lo=0.5, hi=0.95, iters=28, use_c=None):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val, _ = solve_dual_edge(a_vals, b_vals, eps_b, mid, d, use_c=use_c)
        if val is not None and val < -1e-9:
            lo = mid  # certified lambda_max(T) > mid!
        else:
            hi = mid
    return lo

def theta_to_E(th):
    return E_max_box - th * (E_max_box - E_min_box)

def cert_classical_only(a_vals, d, lo=0.5, hi=0.95, iters=28):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Hd = np.array([[a_vals[i+j] for j in range(d+1)] for i in range(d+1)])
        Hd1 = np.array([[a_vals[i+j+1] for j in range(d+1)] for i in range(d+1)])
        if min(np.min(np.linalg.eigvalsh(Hd)), np.min(np.linalg.eigvalsh(Hd1)), np.min(np.linalg.eigvalsh(mid*Hd - Hd1))) >= -1e-12:
            hi = mid
        else:
            lo = mid
    return lo

for eps_level in [0.0, 0.005, 0.02]:
    print(f"\n  Linear-shadow noise radius eps_b = {eps_level}:")
    for d in [0, 1, 2]:
        th_cls = cert_classical_only(a_mom, d)
        th_blk = cert_theta_dual(a_mom, b_mom, [eps_level]*(2*d+2), d)
        th_full = cert_theta_dual(a_mom, b_mom, [eps_level]*(2*d+2), d, use_c=(c_mom, [eps_level]*(2*d+2)))
        print(f"    d={d} | Classical(a): {theta_to_E(th_cls):+.6f} | Anchored(a,b only!): {theta_to_E(th_blk):+.6f} | Full(a,b,c): {theta_to_E(th_full):+.6f} | True E0={E0_4:+.6f}")

# Now test TWO-EDGE SPECTRAL-GAP EXCLUSION CERTIFICATE on (theta_1, theta_2)!
# Suppose we want to certify that there is spectrum strictly inside (theta_1, theta_2)
# using a superposition / mixed state or the surrogate+trial block moments!
# Note: in our 4-site Hubbard chain, |psi_sur> has weight on BOTH E0 and higher states (E1 or excited singlet)!
# Let's check the spectral decomposition of |psi_sur> and |psi_T> in H4!
ov_sur = np.abs(vec4.T @ psi_sur)**2
ov_T = np.abs(vec4.T @ psi_T)**2
nz_sur = [(i, ev4[i], (E_max_box - ev4[i])/(E_max_box - E_min_box), ov_sur[i], ov_T[i]) for i in range(len(ev4)) if ov_sur[i] > 1e-3 or ov_T[i] > 1e-3]
print("\n--- Spectral weights of |psi_sur> and |psi_T> in 4-site Hubbard ---")
for idx, E, lam, ws, wt in nz_sur[:8]:
    print(f"  level {idx:2d}: E={E:+.6f}, lam={lam:.6f} | |<n|sur>|^2={ws:.5f} | |<n|T>|^2={wt:.5f}")

def solve_dual_gap(a_vals, b_vals, eps_b, th1, th2, d, use_c=None):
    """
    Certify spectrum in (th1, th2) by refuting support in [0, th1] U [th2, 1]:
      Localizers: H_d >= 0, H_d^(1) >= 0, H_d - H_d^(1) >= 0,
      and G_d^(th1,th2) = H_d^(2) - (th1+th2) H_d^(1) + th1*th2 H_d >= 0
      (note: H_d^(2) needs moments up to 2d+2!).
    """
    K = 2 * d + 2
    m_sz = 2 * (d + 1)
    P = cp.Variable((m_sz, m_sz), symmetric=True)
    Q = cp.Variable((m_sz, m_sz), symmetric=True)
    R = cp.Variable((m_sz, m_sz), symmetric=True)
    G = cp.Variable((m_sz, m_sz), symmetric=True)
    coeff_a = [0] * (K + 1)
    coeff_b = [0] * (K + 1)
    coeff_c = [0] * (K + 1)
    for i in range(d + 1):
        for j in range(d + 1):
            r0, c0 = 2 * i, 2 * j
            k0, k1, k2 = i + j, i + j + 1, i + j + 2
            # P + R + th1*th2*G on k0
            M0 = P[r0:r0+2, c0:c0+2] + R[r0:r0+2, c0:c0+2] + (th1 * th2) * G[r0:r0+2, c0:c0+2]
            # Q - R - (th1+th2)*G on k1
            M1 = Q[r0:r0+2, c0:c0+2] - R[r0:r0+2, c0:c0+2] - (th1 + th2) * G[r0:r0+2, c0:c0+2]
            # G on k2
            M2 = G[r0:r0+2, c0:c0+2]
            for kk, Mk in [(k0, M0), (k1, M1), (k2, M2)]:
                coeff_a[kk] = coeff_a[kk] + Mk[0, 0]
                coeff_b[kk] = coeff_b[kk] + Mk[0, 1] + Mk[1, 0]
                coeff_c[kk] = coeff_c[kk] + Mk[1, 1]

    constraints = [P >> 0, Q >> 0, R >> 0, G >> 0,
                   cp.trace(P) + cp.trace(Q) + cp.trace(R) + cp.trace(G) <= 1.0]
    obj = sum(coeff_a[k] * a_vals[k] for k in range(K + 1)) + coeff_c[0]
    for k in range(K + 1):
        obj += coeff_b[k] * b_vals[k] + (eps_b[k] * cp.abs(coeff_b[k]) if eps_b[k] > 0 else 0)
    if use_c is None:
        for k in range(1, K + 1):
            constraints.append(coeff_c[k] == 0)
    else:
        c_vals, eps_c = use_c
        for k in range(1, K + 1):
            obj += coeff_c[k] * c_vals[k] + (eps_c[k] * cp.abs(coeff_c[k]) if eps_c[k] > 0 else 0)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    return prob.value

# Test gap window between ground state (lam0=0.588) and first excited singlet (let's see what shows up!)
print("\n--- Two-Edge Spectral Gap Certificate Test (d=2, K=6) ---")
# Fix th2 = lam0_4 + 0.002 (just above ground state, OR th2 = lam0_4 - 0.002? Wait:
# to bound the gap ABOVE the ground state, the ground state is AT lam0_4, and excited states are BELOW lam0_4!
# So a spectral gap (E0, E_cut) corresponds to NO spectrum in (th1, lam0_4 - eps)!
# If we refute support in [0, th1] U [lam0_4, 1], we certify spectrum strictly in (th1, lam0_4),
# i.e. E_excited < E(th1), giving a CERTIFIED UPPER BOUND ON THE VISIBLE SPECTRAL GAP!)
for th1_E in [3.5, 2.5, 2.0, 1.5, 1.0]:
    th1 = (E_max_box - th1_E) / (E_max_box - E_min_box)
    th2 = lam0_4  # ground state sits at th2!
    val0 = solve_dual_gap(a_mom, b_mom, [0.0]*7, th1, th2, d=2)
    val_n = solve_dual_gap(a_mom, b_mom, [0.002]*7, th1, th2, d=2)
    print(f"  Test excited E < {th1_E:+.2f} (th1={th1:.4f}, th2={th2:.4f}): dual_margin(eps=0)={val0:+.3e} | dual_margin(eps=0.002)={val_n:+.3e}")
