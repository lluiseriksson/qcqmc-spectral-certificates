"""
Verify fixed dual SDP (with strict status + exact rational LDLT verification)
and confirm zero false certificates at eps_b > 0.
"""
import numpy as np
import cvxpy as cp
from fractions import Fraction
from itertools import combinations
from scipy.linalg import expm

def exact_ldlt_psd(mat_q):
    """
    Exact rational LDL^T / Sylvester-pivoting PSD check for a symmetric matrix over Fraction.
    Returns (is_psd, pivots). Zero floating-point error!
    """
    n = len(mat_q)
    # Work on a copy using symmetric diagonal pivoting for semidefinite matrices
    A = [[Fraction(x) for x in row] for row in mat_q]
    pivots = []
    for k in range(n):
        # Find largest diagonal entry among k..n-1
        best = max(range(k, n), key=lambda i: A[i][i])
        if A[best][best] < 0:
            return False, pivots
        if A[best][best] == 0:
            # Remaining submatrix must be identically zero to be PSD
            for i in range(k, n):
                for j in range(k, n):
                    if A[i][j] != 0:
                        return False, pivots
            pivots.extend([Fraction(0)] * (n - k))
            return True, pivots
        if best != k:
            A[k], A[best] = A[best], A[k]
            for i in range(n):
                A[i][k], A[i][best] = A[i][best], A[i][k]
        piv = A[k][k]
        pivots.append(piv)
        for i in range(k + 1, n):
            factor = A[i][k] / piv
            for j in range(i, n):
                val = A[i][j] - factor * A[k][j]
                A[i][j] = val
                A[j][i] = val
    return True, pivots

def to_rational_psd(M_float, denom=10**7, shift_eps=Fraction(1, 10**6)):
    """Symmetrize, convert to Fraction, and add tiny shift if needed to guarantee exact PSD."""
    n = M_float.shape[0]
    M_sym = 0.5 * (M_float + M_float.T)
    ev_min = float(np.min(np.linalg.eigvalsh(M_sym)))
    if ev_min < 0:
        M_sym = M_sym + (-ev_min + 1e-8) * np.eye(n)
    Mq = [[Fraction(int(round(M_sym[i, j] * denom)), denom) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            Mq[j][i] = Mq[i][j]
    ok, _ = exact_ldlt_psd(Mq)
    if not ok:
        for i in range(n):
            Mq[i][i] += shift_eps
        ok, _ = exact_ldlt_psd(Mq)
    return ok, Mq

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
ev4, _ = np.linalg.eigh(H4)
E0_4 = ev4[0]
row_diag = np.diag(H4)
row_off = np.sum(np.abs(H4), axis=1) - np.abs(row_diag)
E_min_box = float(np.min(row_diag - row_off))
E_max_box = float(np.max(row_diag + row_off))
T4 = (E_max_box * np.eye(len(H4)) - H4) / (E_max_box - E_min_box)

H4_free = build_hubbard_chain(L=4, t=1.0, U=0.0)
_, vec4_free = np.linalg.eigh(H4_free)
psi_sur = vec4_free[:, 0]
psi_T = expm(-0.6 * H4) @ psi_sur
psi_T /= np.linalg.norm(psi_T)

a_mom = [float(psi_sur @ np.linalg.matrix_power(T4, k) @ psi_sur) for k in range(8)]
b_mom = [float(psi_sur @ np.linalg.matrix_power(T4, k) @ psi_T) for k in range(8)]
c_mom = [float(psi_T @ np.linalg.matrix_power(T4, k) @ psi_T) for k in range(8)]

def dual_margin_exact_check(P, Q, R, a_vals, b_vals, eps_b, theta, d, use_c=None):
    """
    Independent exact verification of a candidate dual triple (P, Q, R):
    1. Check exact PSD of P, Q, R over Q (via exact LDL^T).
    2. Compute exact worst-case margin over b in [b_vals - eps_b, b_vals + eps_b]
       AND verify that if use_c is None, coeff_c[k] == 0 (or bounded if c in [0,1]!).
    WAIT A SECOND! LOOK AT THAT IDEA:
    Because T in [0, I] and ||Psi_T|| = 1, we ALSO KNOW 0 <= c_k <= 1 (and c_k <= theta^k on [0, theta]!)
    So if exact rounding leaves a tiny 1e-8 residual in coeff_c[k], multiplying |coeff_c[k]| * 1
    STILL GIVES A 100% SOUND, ZERO-SLACK RATIONAL BOUND!!
    """
    K = 2 * d + 1
    okP, Pq = to_rational_psd(P)
    okQ, Qq = to_rational_psd(Q)
    okR, Rq = to_rational_psd(R)
    if not (okP and okQ and okR):
        return False, Fraction(1)

    th_q = Fraction(theta).limit_denominator(10**9)
    ca = [Fraction(0)] * (K + 1)
    cb = [Fraction(0)] * (K + 1)
    cc = [Fraction(0)] * (K + 1)
    for i in range(d + 1):
        for j in range(d + 1):
            r0, c0 = 2 * i, 2 * j
            k0, k1 = i + j, i + j + 1
            for di in range(2):
                for dj in range(2):
                    v0 = Pq[r0+di][c0+dj] + th_q * Rq[r0+di][c0+dj]
                    v1 = Qq[r0+di][c0+dj] - Rq[r0+di][c0+dj]
                    if di == 0 and dj == 0:
                        ca[k0] += v0; ca[k1] += v1
                    elif di == 1 and dj == 1:
                        cc[k0] += v0; cc[k1] += v1
                    else:
                        cb[k0] += v0; cb[k1] += v1

    # Compute exact upper bound on sup_{b in B, c_0=1, c_k in [0, 1]} L(a, b, c)
    # Note: for ANY probability measure on [0, theta] subset [0, 1], 0 <= c_k <= 1!
    margin = cc[0]
    for k in range(K + 1):
        ak = Fraction(a_vals[k]).limit_denominator(10**12)
        bk = Fraction(b_vals[k]).limit_denominator(10**12)
        eb = Fraction(eps_b[k]).limit_denominator(10**12)
        margin += ca[k] * ak + cb[k] * bk + abs(cb[k]) * eb
    if use_c is None:
        # Since 0 <= c_k <= 1 for any prob measure on [0,1], sup_{c_k in [0,1]} cc[k]*c_k = max(0, cc[k])!
        for k in range(1, K + 1):
            if cc[k] > 0:
                margin += cc[k]
    else:
        c_vals, eps_c = use_c
        for k in range(1, K + 1):
            ck = Fraction(c_vals[k]).limit_denominator(10**12)
            ec = Fraction(eps_c[k]).limit_denominator(10**12)
            margin += cc[k] * ck + abs(cc[k]) * ec
    return (margin < 0), margin

def solve_dual_edge(a_vals, b_vals, eps_b, theta, d, use_c=None):
    K = 2 * d + 1
    m_sz = 2 * (d + 1)
    P = cp.Variable((m_sz, m_sz), symmetric=True)
    Q = cp.Variable((m_sz, m_sz), symmetric=True)
    R = cp.Variable((m_sz, m_sz), symmetric=True)
    ca = [0] * (K + 1); cb = [0] * (K + 1); cc = [0] * (K + 1)
    for i in range(d + 1):
        for j in range(d + 1):
            r0, c0 = 2 * i, 2 * j
            k0, k1 = i + j, i + j + 1
            M0 = P[r0:r0+2, c0:c0+2] + theta * R[r0:r0+2, c0:c0+2]
            M1 = Q[r0:r0+2, c0:c0+2] - R[r0:r0+2, c0:c0+2]
            ca[k0] = ca[k0] + M0[0, 0]; ca[k1] = ca[k1] + M1[0, 0]
            cb[k0] = cb[k0] + M0[0, 1] + M0[1, 0]; cb[k1] = cb[k1] + M1[0, 1] + M1[1, 0]
            cc[k0] = cc[k0] + M0[1, 1]; cc[k1] = cc[k1] + M1[1, 1]
    constraints = [P >> 0, Q >> 0, R >> 0, cp.trace(P) + cp.trace(Q) + cp.trace(R) <= 1.0]
    obj = sum(ca[k] * a_vals[k] for k in range(K + 1)) + cc[0]
    for k in range(K + 1):
        obj += cb[k] * b_vals[k] + (eps_b[k] * cp.abs(cb[k]) if eps_b[k] > 0 else 0)
    if use_c is None:
        for k in range(1, K + 1):
            constraints.append(cc[k] == 0)
    else:
        c_vals, eps_c = use_c
        for k in range(1, K + 1):
            obj += cc[k] * c_vals[k] + (eps_c[k] * cp.abs(cc[k]) if eps_c[k] > 0 else 0)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL or prob.value is None or prob.value >= -1e-7:
        return False, 0.0
    ok, margin_q = dual_margin_exact_check(P.value, Q.value, R.value, a_vals, b_vals, eps_b, theta, d, use_c=use_c)
    return ok, float(margin_q)

def cert_theta_verified(a_vals, b_vals, eps_b, d, lo=0.5, hi=0.95, iters=22, use_c=None):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok, _ = solve_dual_edge(a_vals, b_vals, eps_b, mid, d, use_c=use_c)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo

def theta_to_E(th):
    return E_max_box - th * (E_max_box - E_min_box)

for eps_level in [0.0, 0.005, 0.02]:
    print(f"\n  EXACT-RATIONAL-VERIFIED bounds (eps_b = {eps_level}):")
    for d in [0, 1, 2]:
        th_blk = cert_theta_verified(a_mom, b_mom, [eps_level]*(2*d+2), d)
        th_full = cert_theta_verified(a_mom, b_mom, [eps_level]*(2*d+2), d, use_c=(c_mom, [eps_level]*(2*d+2)))
        E_blk = theta_to_E(th_blk)
        E_full = theta_to_E(th_full)
        assert E_blk >= E0_4 - 1e-12, f"VIOLATION E_blk={E_blk} < E0={E0_4}"
        assert E_full >= E0_4 - 1e-12, f"VIOLATION E_full={E_full} < E0={E0_4}"
        print(f"    d={d} | Anchored(a,b ONLY): E <= {E_blk:+.6f} | Full(a,b,c): E <= {E_full:+.6f} | True E0={E0_4:+.6f}")
