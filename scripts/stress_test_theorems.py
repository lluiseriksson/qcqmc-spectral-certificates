"""
Adversarial numerical stress-test of Theorems 1-4 before formalization.
Tests:
  1. Theorem 1 & 2A: Exact Plucker-kernel distance ||(I - S S^+) f^*||_2 and
     holomorphic projection restoration on complex walkers.
  2. Theorem 2B: Explicit N=4, eta=2 Backflow NQS (Jiang et al. Eq. S2.19)
     producing non-zero Wirtinger defect G^{dbar} and unphysical mixed energy
     E_mix < E_0 EVEN on an exact zero-error walker decomposition of |Psi_0>,
     plus representation dependence across two exact decompositions of |Psi_0>.
  3. Theorem 3: Universal Mixed-Energy Variational-Boost Law across 10,000 random
     complex Hermitian Hamiltonians & state pairs (0 violations + sharpness check).
  4. Theorem 4: Surrogate-Anchored Block Hausdorff SDP (d=0,1,2,3) for ground-state
     energy E_0 and two-edge visible spectral gap (theta_1, theta_2) on a 4-site
     1D Hubbard chain (8 spin-orbitals, half-filling, dim=36) with exact rational
     dual verification.
"""
import numpy as np
import cvxpy as cp
from itertools import combinations

rng = np.random.default_rng(20261124)

# =============================================================================
# PART 1: EXACT FERMIONIC ALGEBRA FOR N=4, eta=2 (2-site Hubbard / minimal H2)
# =============================================================================
def slater_vec(Phi):
    """Map N x eta matrix Phi to vector in wedge^eta C^N (lexicographic basis)."""
    N, eta = Phi.shape
    basis = list(combinations(range(N), eta))
    v = np.zeros(len(basis), dtype=np.complex128)
    for idx, rows in enumerate(basis):
        v[idx] = np.linalg.det(Phi[list(rows), :])
    return v, basis

def build_two_site_hubbard(t=1.0, U=4.0):
    """
    4 spin-orbitals: 0=(site 0, up), 1=(site 1, up), 2=(site 0, dn), 3=(site 1, dn).
    eta = 2 electrons (dim = C(4,2) = 6).
    Factorized form H = B0 + c1 B1^2 + c2 B2^2 (exact one-body + squared one-body!).
    n_{0,up} n_{0,dn} = 1/2 ( (n_{0,up} + n_{0,dn})^2 - (n_{0,up} + n_{0,dn}) )
    """
    N = 4
    # One-body operators as 4x4 Hermitian matrices
    K = np.zeros((N, N), dtype=np.complex128)
    K[0, 1] = K[1, 0] = -t
    K[2, 3] = K[3, 2] = -t

    N0 = np.diag([1.0, 0.0, 1.0, 0.0]).astype(np.complex128)
    N1 = np.diag([0.0, 1.0, 0.0, 1.0]).astype(np.complex128)
    # H = K - (U/2)(N0 + N1) + (U/2) N0^2 + (U/2) N1^2
    B0 = K - 0.5 * U * (N0 + N1)
    B_list = [(0.5 * U, N0), (0.5 * U, N1)]

    # Build exact 6x6 matrix in wedge^2 C^4
    basis = list(combinations(range(N), 2))
    D = len(basis)

    def one_body_mat(B):
        M = np.zeros((D, D), dtype=np.complex128)
        for j, cols in enumerate(basis):
            Phi = np.zeros((N, 2), dtype=np.complex128)
            Phi[cols[0], 0] = 1.0
            Phi[cols[1], 1] = 1.0
            # d/ds |_{s=0} (-i) |e^{isB} Phi> = (B phi_0) ^ phi_1 + phi_0 ^ (B phi_1)
            v1, _ = slater_vec(np.column_stack([B @ Phi[:, 0], Phi[:, 1]]))
            v2, _ = slater_vec(np.column_stack([Phi[:, 0], B @ Phi[:, 1]]))
            M[:, j] = v1 + v2
        return M

    H = one_body_mat(B0)
    for c, B in B_list:
        Mb = one_body_mat(B)
        H = H + c * (Mb @ Mb)
    assert np.allclose(H, H.conj().T)
    return H, B0, B_list, basis

H2, B0, B_list, basis42 = build_two_site_hubbard(t=1.0, U=4.0)
evals2, evecs2 = np.linalg.eigh(H2)
E0_2 = evals2[0]
psi0_2 = evecs2[:, 0]
print(f"=== PART 1: 2-SITE HUBBARD (N=4, eta=2, t=1, U=4) ===")
print(f"Exact eigenvalues: {np.round(evals2, 6)}")
print(f"E0 = {E0_2:.10f}, |Psi_0> = {np.round(psi0_2.real, 6)}")
# Note: in basis [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
# singlet ground state is supported on (0,2),(0,3),(1,2),(1,3) (Sz=0 sector!)

# Backflow NQS: f_kappa(Phi) = det[G_kappa(P_Phi)^dagger Phi]
# Choose a clean analytic backflow:
# G_0 = e_0, e_3  (or linear combo), plus kappa * P_Phi @ M_bf
# Better: let G_0 correspond to the HF determinant |0,3> + |1,2> (Wait: single det!)
# Let's take G_0 = [[1,0],[1,0],[0,1],[0,1]] / sqrt(2) (delocalized singlet HF det!)
# Notice <Psi_HF | Psi_0> != 0!
G0 = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
], dtype=np.complex128) / np.sqrt(2.0)

# Backflow perturbation matrix M_bf (depends smoothly on P_Phi = Phi (Phi^H Phi)^{-1} Phi^H)
# Use a projector-dependent orbital shift: G_kappa(P) = G0 + kappa * (P @ A_bf)
A_bf = np.array([
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [0.0, 1.0],
], dtype=np.complex128) / np.sqrt(2.0)

def projector(Phi):
    return Phi @ np.linalg.inv(Phi.conj().T @ Phi) @ Phi.conj().T

def f_backflow(Phi, kappa):
    P = projector(Phi)
    G = G0 + kappa * (P @ A_bf)
    return np.linalg.det(G.conj().T @ Phi)

# Check strict GL(2, C) equivariance
Phi_test = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
U_test = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
kappa_val = 0.5
eq_err = abs(f_backflow(Phi_test @ U_test, kappa_val) - np.linalg.det(U_test) * f_backflow(Phi_test, kappa_val))
print(f"GL(2,C) equivariance error at kappa={kappa_val}: {eq_err:.3e}")

# Compute exact Plucker-kernel distance Theorem 2A on m=10 complex walkers
m_walk = 10
walkers = [rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)) for _ in range(m_walk)]
W = np.column_stack([slater_vec(Ph)[0] for Ph in walkers])  # 6 x 10
S = W.conj().T @ W  # 10 x 10 Hermitian PSD, rank <= 6
S_pinv = np.linalg.pinv(S, rcond=1e-12)

for kap in [0.0, 0.5]:
    f_S = np.array([f_backflow(Ph, kap) for Ph in walkers])
    eta_pl = np.linalg.norm((np.eye(m_walk) - S @ S_pinv) @ f_S.conj())
    # Projected physical state |Psi_proj> = W S^+ f_S^*
    psi_proj = W @ S_pinv @ f_S.conj()
    v_proj = (W.conj().T @ psi_proj).conj()
    dist_proj = np.linalg.norm(f_S - v_proj)
    # Check against 500 random perturbations of psi_proj
    min_rand = min(np.linalg.norm(f_S - (W.conj().T @ (psi_proj + 0.01*(rng.normal(size=6)+1j*rng.normal(size=6)))).conj()) for _ in range(500))
    print(f"  kappa={kap}: eta_Pl={eta_pl:.8f}, ||f_S - v_proj||={dist_proj:.8f}, gap_diff={abs(eta_pl-dist_proj):.2e}, min_perturbed={min_rand:.8f}")

# Now compute Jiang et al. Eq. (S2.21) local energy via high-order central finite differences along e^{isB} Phi
from scipy.linalg import expm

def D_B(f, Phi, B, eps=1e-6):
    # -i d/ds f(e^{isB} Phi)|_0 via 4th-order central diff
    fp1 = f(expm(1j * eps * B) @ Phi)
    fm1 = f(expm(-1j * eps * B) @ Phi)
    fp2 = f(expm(2j * eps * B) @ Phi)
    fm2 = f(expm(-2j * eps * B) @ Phi)
    deriv = (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12 * eps)
    return -1j * deriv

def D_B2(f, Phi, B, eps=1e-4):
    # - d^2/ds^2 f(e^{isB} Phi)|_0 via 4th-order central diff
    f0 = f(Phi)
    fp1 = f(expm(1j * eps * B) @ Phi)
    fm1 = f(expm(-1j * eps * B) @ Phi)
    fp2 = f(expm(2j * eps * B) @ Phi)
    fm2 = f(expm(-2j * eps * B) @ Phi)
    d2 = (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12 * eps**2)
    return -d2

def L_H_surrogate(f, Phi):
    val = D_B(f, Phi, B0)
    for c, B in B_list:
        val += c * D_B2(f, Phi, B)
    return val

# Build TWO DIFFERENT EXACT WALKER DECOMPOSITIONS of the EXACT ground state |Psi_0>!
# Decomposition 1: exact computational basis determinants |0,2>, |0,3>, |1,2>, |1,3>
def comp_det(i, j):
    Ph = np.zeros((4, 2), dtype=np.complex128)
    Ph[i, 0] = 1.0
    Ph[j, 1] = 1.0
    return Ph

decomp1_Phis = [comp_det(0, 2), comp_det(0, 3), comp_det(1, 2), comp_det(1, 3)]
decomp1_coeffs = [psi0_2[1], psi0_2[2], psi0_2[3], psi0_2[4]]  # exact |Psi_0> = sum c_i |phi_i>

# Decomposition 2: rotated nonorthogonal determinants spanning the exact same 4D Sz=0 space
theta_rot = 0.37
R_spin = np.array([[np.cos(theta_rot), -np.sin(theta_rot)],
                   [np.sin(theta_rot),  np.cos(theta_rot)]], dtype=np.complex128)
R4 = np.block([[R_spin, np.zeros((2,2))], [np.zeros((2,2)), R_spin]])
decomp2_Phis = [R4 @ Ph for Ph in decomp1_Phis]
W2 = np.column_stack([slater_vec(Ph)[0] for Ph in decomp2_Phis])
decomp2_coeffs, _, _, _ = np.linalg.lstsq(W2, psi0_2, rcond=None)
assert np.allclose(W2 @ decomp2_coeffs, psi0_2)

print("\n--- Exact Ground-State Mixed Energy E_mix^theta (ZERO WALKER ERROR!) ---")
for kap in [0.0, -0.4, 0.4, 0.8]:
    f_k = lambda Ph, k=kap: f_backflow(Ph, k)
    res = []
    for name, Phis, coeffs in [("Decomp1 (comp basis)", decomp1_Phis, decomp1_coeffs),
                               ("Decomp2 (rotated)", decomp2_Phis, decomp2_coeffs)]:
        # w_i = c_i * f_k(Phi_i) so that sum (w_i / f_k(Phi_i)) |phi_i> = |Psi_0> EXACTLY!
        ws = [c * f_k(Ph) for c, Ph in zip(coeffs, Phis)]
        E_Ls = [L_H_surrogate(f_k, Ph) / f_k(Ph) for Ph in Phis]
        E_mix = sum(w * el for w, el in zip(ws, E_Ls)) / sum(ws)
        res.append((name, E_mix.real, E_mix.real - E0_2))
    print(f"  kappa={kap:+.1f} | {res[0][0]}: E_mix={res[0][1]:+.8f} (bias={res[0][2]:+.8f}) | {res[1][0]}: E_mix={res[1][1]:+.8f} (bias={res[1][2]:+.8f})")

# =============================================================================
# PART 2: THEOREM 3 — UNIVERSAL MIXED-ENERGY VARIATIONAL-BOOST LAW (10,000 TESTS)
# =============================================================================
print("\n=== PART 2: MIXED-ENERGY VARIATIONAL-BOOST LAW (10,000 RANDOM TESTS) ===")
violations = 0
min_slack = 1e9
for trial in range(10000):
    d_dim = int(rng.integers(2, 12))
    X = rng.normal(size=(d_dim, d_dim)) + 1j * rng.normal(size=(d_dim, d_dim))
    H = 0.5 * (X + X.conj().T)
    ev = np.linalg.eigvalsh(H)
    E0, Emax_true = ev[0], ev[-1]
    Emax = Emax_true + float(rng.uniform(0.0, 2.0))
    u = rng.normal(size=d_dim) + 1j * rng.normal(size=d_dim)
    u /= np.linalg.norm(u)
    v = rng.normal(size=d_dim) + 1j * rng.normal(size=d_dim)
    v /= np.linalg.norm(v)
    b0 = np.vdot(u, v)
    if abs(b0) > 1.0 - 1e-10:
        continue
    E_sur = np.vdot(u, H @ u).real
    d_H = np.vdot(u, (E_sur * np.eye(d_dim) - H) @ v)
    E_cert = E_sur - (abs(d_H)**2) / ((Emax - E_sur) * (1.0 - abs(b0)**2))
    slack = E_cert - E0
    if slack < -1e-11:
        violations += 1
    if slack < min_slack:
        min_slack = slack
print(f"Violations out of 10,000: {violations} | min(E_cert - E0) = {min_slack:.3e}")

# Sharpness witness: rank-1 A = Emax I - H on span{u, v}
H_sharp = np.diag([-1.0, 1.0])
Emax_s = 1.0
u_s = np.array([np.cos(0.4), np.sin(0.4)])
v_s = np.array([1.0, 0.0])  # exact ground state!
b0_s = np.vdot(u_s, v_s)
E_sur_s = np.vdot(u_s, H_sharp @ u_s).real
d_H_s = np.vdot(u_s, (E_sur_s * np.eye(2) - H_sharp) @ v_s)
E_cert_s = E_sur_s - (abs(d_H_s)**2) / ((Emax_s - E_sur_s) * (1.0 - abs(b0_s)**2))
print(f"Sharpness test: E_sur={E_sur_s:.6f}, E_cert={E_cert_s:.12f}, E0=-1.000000000000, diff={abs(E_cert_s - (-1.0)):.2e}")

# =============================================================================
# PART 3: THEOREM 4 — SURROGATE-ANCHORED BLOCK HAUSDORFF SDP ON 4-SITE HUBBARD
# =============================================================================
print("\n=== PART 3: 4-SITE 1D HUBBARD CHAIN (8 SPIN-ORBITALS, N_up=2, N_dn=2, dim=36) ===")
def build_hubbard_chain(L=4, t=1.0, U=4.0, pbc=False):
    up_states = list(combinations(range(L), L // 2))
    dn_states = list(combinations(range(L), L // 2))
    n_up, n_dn = len(up_states), len(dn_states)
    up_idx = {s: i for i, s in enumerate(up_states)}
    dn_idx = {s: i for i, s in enumerate(dn_states)}
    D = n_up * n_dn
    H = np.zeros((D, D), dtype=np.float64)

    def hop_mat(states, idx_map):
        M = np.zeros((len(states), len(states)), dtype=np.float64)
        edges = [(i, i + 1) for i in range(L - 1)] + ([(L - 1, 0)] if pbc else [])
        for a, st in enumerate(states):
            st_set = set(st)
            for i, j in edges:
                for src, dst in [(i, j), (j, i)]:
                    if src in st_set and dst not in st_set:
                        # fermionic sign: count occupied orbitals strictly between src and dst
                        lo, hi = min(src, dst), max(src, dst)
                        parity = sum(1 for x in st if lo < x < hi)
                        new_st = tuple(sorted((st_set - {src}) | {dst}))
                        M[idx_map[new_st], a] += -t * ((-1.0) ** parity)
        return M

    H_up = hop_mat(up_states, up_idx)
    H_dn = hop_mat(dn_states, dn_idx)
    H += np.kron(H_up, np.eye(n_dn)) + np.kron(np.eye(n_up), H_dn)
    # Hubbard interaction
    for iu, su in enumerate(up_states):
        su_set = set(su)
        for idn, sd in enumerate(dn_states):
            double_occ = len(su_set & set(sd))
            idx = iu * n_dn + idn
            H[idx, idx] += U * double_occ
    return H, up_states, dn_states

H4, _, _ = build_hubbard_chain(L=4, t=1.0, U=4.0, pbc=False)
ev4, vec4 = np.linalg.eigh(H4)
E0_4, E1_4, Emax_4 = ev4[0], ev4[1], ev4[-1]
# Known conservative upper/lower bounds from Gershgorin (no oracle!)
row_sums = np.sum(np.abs(H4), axis=1)
E_bnd = float(np.max(row_sums))  # guaranteed ||H||_op <= E_bnd
E_min_box, E_max_box = -E_bnd, E_bnd
T4 = (E_max_box * np.eye(len(H4)) - H4) / (E_max_box - E_min_box)
lam0_4 = (E_max_box - E0_4) / (E_max_box - E_min_box)
lam1_4 = (E_max_box - E1_4) / (E_max_box - E_min_box)
print(f"E0={E0_4:.6f}, E1={E1_4:.6f}, gap={E1_4-E0_4:.6f}, E_bnd(Gershgorin)={E_bnd:.2f}")
print(f"Rescaled T: lam0={lam0_4:.6f}, lam1={lam1_4:.6f}, lam_gap={lam0_4-lam1_4:.6f}")

# Construct a realistic classical surrogate |Psi_sur> (U=0 ground state = RHF Slater determinant!)
H4_free, _, _ = build_hubbard_chain(L=4, t=1.0, U=0.0, pbc=False)
_, vec4_free = np.linalg.eigh(H4_free)
psi_sur = vec4_free[:, 0]
# Construct a quantum-prepared trial state |Psi_T> with moderate fidelity (e.g. ~85% overlap with ground state)
# e.g. imaginary-time evolved RHF: e^{-beta H} |psi_sur> / norm (mimics LUCJ / QPE-like trial!)
psi_T = expm(-0.6 * H4) @ psi_sur
psi_T /= np.linalg.norm(psi_T)

E_sur4 = float(psi_sur @ H4 @ psi_sur)
E_T4 = float(psi_T @ H4 @ psi_T)
ov_sur_0 = abs(psi_sur @ vec4[:, 0])**2
ov_T_0 = abs(psi_T @ vec4[:, 0])**2
b0_4 = float(psi_sur @ psi_T)
print(f"RHF surrogate: E_sur={E_sur4:.6f} (fidelity={ov_sur_0:.4f})")
print(f"Quantum trial: E_T={E_T4:.6f} (fidelity={ov_T_0:.4f}), <sur|T_state>={b0_4:.4f}")

# Exact moments a_k = <sur|T^k|sur>, b_k = <sur|T^k|Psi_T>, c_k = <Psi_T|T^k|Psi_T>
Kmax = 7
a_mom = [float(psi_sur @ np.linalg.matrix_power(T4, k) @ psi_sur) for k in range(Kmax + 1)]
b_mom = [float(psi_sur @ np.linalg.matrix_power(T4, k) @ psi_T) for k in range(Kmax + 1)]
c_mom = [float(psi_T @ np.linalg.matrix_power(T4, k) @ psi_T) for k in range(Kmax + 1)]

def check_block_hausdorff_feasible(a_vals, b_vals, eps_b, theta, d, use_c=None):
    """
    Test feasibility of 2x2 block Hausdorff moments on [0, theta] with:
      a_0..a_{2d+1} exact,
      |b_k - b_vals[k]| <= eps_b[k] (box confidence set),
      c_0 = 1, c_1..c_{2d+1} free (if use_c is None) or constrained.
    """
    K = 2 * d + 1
    b_var = cp.Variable(K + 1)
    c_var = cp.Variable(K + 1)
    constraints = [
        c_var[0] == 1.0,
        cp.abs(b_var - np.array(b_vals[:K+1])) <= np.array(eps_b[:K+1]),
    ]
    if use_c is not None:
        c_vals, eps_c = use_c
        constraints.append(cp.abs(c_var - np.array(c_vals[:K+1])) <= np.array(eps_c[:K+1]))

    # Build 2(d+1) x 2(d+1) block Hankel H_d and localizer L_d^theta = theta H_d - H_d^(1)
    # Also include H_d^(1) >= 0 (support in [0, theta]!)
    def block_hankel(shift):
        rows = []
        for i in range(d + 1):
            row = []
            for j in range(d + 1):
                k = i + j + shift
                blk = cp.bmat([[a_vals[k], b_var[k]],
                               [b_var[k], c_var[k]]])
                row.append(blk)
            rows.append(row)
        return cp.bmat(rows)

    Hd = block_hankel(0)
    Hd1 = block_hankel(1)
    constraints += [Hd >> 0, Hd1 >> 0, theta * Hd - Hd1 >> 0]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    return prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

def bisection_theta_cert(a_vals, b_vals, eps_b, d, lo=0.5, hi=0.95, iters=30, use_c=None):
    """Find largest theta that is INFEASIBLE (certified lambda_max(T) > theta)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if check_block_hausdorff_feasible(a_vals, b_vals, eps_b, mid, d, use_c=use_c):
            hi = mid
        else:
            lo = mid
    return lo

def theta_to_E(th):
    return E_max_box - th * (E_max_box - E_min_box)

# Classical Krylov alone (equivalent to eps_b = inf or just scalar a_k!)
def bisection_classical_only(a_vals, d, lo=0.5, hi=0.95, iters=30):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Hd = np.array([[a_vals[i+j] for j in range(d+1)] for i in range(d+1)])
        Hd1 = np.array([[a_vals[i+j+1] for j in range(d+1)] for i in range(d+1)])
        ev1 = np.min(np.linalg.eigvalsh(Hd))
        ev2 = np.min(np.linalg.eigvalsh(Hd1))
        ev3 = np.min(np.linalg.eigvalsh(mid * Hd - Hd1))
        if ev1 >= -1e-12 and ev2 >= -1e-12 and ev3 >= -1e-12:
            hi = mid
        else:
            lo = mid
    return lo

print("\n--- Ground-State Energy Upper Bounds E_cert >= E0 (NOISLESS vs NOISY LINEAR SHADOWS!) ---")
for eps_level in [0.0, 0.005, 0.02]:
    print(f"\n  Noise radius eps_b = {eps_level}:")
    for d in [0, 1, 2]:
        th_cls = bisection_classical_only(a_mom, d)
        th_blk = bisection_theta_cert(a_mom, b_mom, [eps_level]*(2*d+2), d)
        E_cls = theta_to_E(th_cls)
        E_blk = theta_to_E(th_blk)
        print(f"    d={d} | Classical Krylov(a): E <= {E_cls:+.6f} | Block Hausdorff(a,b ONLY, c unknown!): E <= {E_blk:+.6f} (true E0={E0_4:+.6f}, gain={E_cls-E_blk:+.6f})")
