"""
Surrogate-Anchored Block Hausdorff Dual SDP (Single-Edge E0 and Two-Edge Spectral Gap)
with exact rational arithmetic LDL^T PSD verification over fractions.Fraction.
Includes explicit symmetry verification and rational moment-enclosure tolerances (eps_a, eps_b).
"""
import numpy as np
import cvxpy as cp
from fractions import Fraction


def exact_ldlt_psd(mat_q):
    """
    Exact rational symmetric-pivoted LDL^T positive-semidefiniteness decision procedure
    over Q (fractions.Fraction). Zero rounding error.
    Strictly verifies squareness AND exact symmetry A[i][j] == A[j][i] before factoring.
    Returns (is_psd: bool, pivots: list[Fraction]).
    """
    n = len(mat_q)
    if n == 0 or any(len(row) != n for row in mat_q):
        return False, []
    A = [[Fraction(x) for x in row] for row in mat_q]
    # Exact symmetry check (fail-closed on any asymmetry)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False, []
    pivots = []
    for k in range(n):
        best = max(range(k, n), key=lambda i: A[i][i])
        if A[best][best] < 0:
            return False, pivots
        if A[best][best] == 0:
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


def to_rational_psd(M_float: np.ndarray, denom: int = 10**9):
    """
    Symmetrize, quantize to Fraction, and apply the smallest adaptive diagonal shift
    (from 0 up to 1e-6) required to guarantee exact symmetric PSD over Q.
    """
    n = M_float.shape[0]
    M_sym = 0.5 * (M_float + M_float.T)
    ev_min = float(np.min(np.linalg.eigvalsh(M_sym)))
    if ev_min < 0:
        M_sym = M_sym + (-ev_min + 1e-10) * np.eye(n)
    Mq_base = [[Fraction(int(round(float(M_sym[i, j]) * denom)), denom) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            Mq_base[j][i] = Mq_base[i][j]

    for shift in (Fraction(0), Fraction(1, 10**10), Fraction(1, 10**9),
                  Fraction(1, 10**8), Fraction(1, 10**7), Fraction(1, 10**6)):
        if shift == 0:
            Mq = Mq_base
        else:
            Mq = [row[:] for row in Mq_base]
            for i in range(n):
                Mq[i][i] += shift
        ok, pivots = exact_ldlt_psd(Mq)
        if ok:
            return True, Mq, pivots
    return False, Mq_base, []


def serialize_q_mat(Mq):
    return [[f"{x.numerator}/{x.denominator}" for x in row] for row in Mq]


def deserialize_q_mat(Ms):
    return [[Fraction(s) for s in row] for row in Ms]


def verify_edge_certificate_exact(Pq, Qq, Rq, a_q, b_q, eps_b_q, theta_q, d,
                                  eps_a_q=None, c_q=None, eps_c_q=None):
    """
    100% exact rational verifier for a Single-Edge Block Hausdorff dual certificate.
    Verifies:
      1. 0 <= theta_q <= 1 and all tolerances eps_a_q, eps_b_q (, eps_c_q) >= 0.
      2. Pq, Qq, Rq are square, symmetric, and PSD over Q (exact pivoted LDL^T).
      3. Worst-case linear functional over |a_k - a_q[k]| <= eps_a_q[k],
         |Re(b_k) - b_q[k]| <= eps_b_q[k], and c_0 = 1, c_k in [0, 1]
         (or |c_k - c_q[k]| <= eps_c_q[k]) is STRICTLY < 0.
    """
    if theta_q < 0 or theta_q > 1:
        return False, Fraction(1), {"reason": "theta out of [0, 1]"}
    K = 2 * d + 1
    m_sz = 2 * (d + 1)
    if eps_a_q is None:
        eps_a_q = [Fraction(0)] * (K + 1)
    if any(x < 0 for x in eps_a_q) or any(x < 0 for x in eps_b_q):
        return False, Fraction(1), {"reason": "negative tolerance"}
    if c_q is not None and any(x < 0 for x in eps_c_q):
        return False, Fraction(1), {"reason": "negative eps_c tolerance"}

    for Mq in (Pq, Qq, Rq):
        if len(Mq) != m_sz:
            return False, Fraction(1), {"reason": "dimension mismatch"}
        ok, _ = exact_ldlt_psd(Mq)
        if not ok:
            return False, Fraction(1), {"reason": "PSD/symmetry check failed"}

    ca = [Fraction(0)] * (K + 1)
    cb = [Fraction(0)] * (K + 1)
    cc = [Fraction(0)] * (K + 1)
    for i in range(d + 1):
        for j in range(d + 1):
            r0, c0 = 2 * i, 2 * j
            k0, k1 = i + j, i + j + 1
            for di in range(2):
                for dj in range(2):
                    v0 = Pq[r0 + di][c0 + dj] + theta_q * Rq[r0 + di][c0 + dj]
                    v1 = Qq[r0 + di][c0 + dj] - Rq[r0 + di][c0 + dj]
                    if di == 0 and dj == 0:
                        ca[k0] += v0; ca[k1] += v1
                    elif di == 1 and dj == 1:
                        cc[k0] += v0; cc[k1] += v1
                    else:
                        cb[k0] += v0; cb[k1] += v1

    margin = cc[0]
    for k in range(K + 1):
        margin += ca[k] * a_q[k] + abs(ca[k]) * eps_a_q[k]
        margin += cb[k] * b_q[k] + abs(cb[k]) * eps_b_q[k]
    if c_q is None:
        for k in range(1, K + 1):
            if cc[k] > 0:
                margin += cc[k]
    else:
        for k in range(1, K + 1):
            margin += cc[k] * c_q[k] + abs(cc[k]) * eps_c_q[k]

    return (margin < 0), margin, {"ca": ca, "cb": cb, "cc": cc}


def verify_gap_certificate_exact(Pq, Qq, Rq, Gq, a_q, b_q, eps_b_q, th1_q, th2_q, d,
                                 eps_a_q=None, c_q=None, eps_c_q=None):
    """
    100% exact rational verifier for a Two-Edge Spectral-Gap Block Hausdorff dual certificate
    refuting support in [0, th1_q] U [th2_q, 1] (with 0 <= th1_q <= th2_q <= 1).
    """
    if th1_q < 0 or th2_q > 1 or th1_q > th2_q:
        return False, Fraction(1), {"reason": "invalid [th1, th2]"}
    K = 2 * d + 2
    m_sz = 2 * (d + 1)
    if eps_a_q is None:
        eps_a_q = [Fraction(0)] * (K + 1)
    if any(x < 0 for x in eps_a_q) or any(x < 0 for x in eps_b_q):
        return False, Fraction(1), {"reason": "negative tolerance"}
    if c_q is not None and any(x < 0 for x in eps_c_q):
        return False, Fraction(1), {"reason": "negative eps_c tolerance"}

    for Mq in (Pq, Qq, Rq, Gq):
        if len(Mq) != m_sz:
            return False, Fraction(1), {"reason": "dimension mismatch"}
        ok, _ = exact_ldlt_psd(Mq)
        if not ok:
            return False, Fraction(1), {"reason": "PSD/symmetry check failed"}

    ca = [Fraction(0)] * (K + 1)
    cb = [Fraction(0)] * (K + 1)
    cc = [Fraction(0)] * (K + 1)
    s_th = th1_q + th2_q
    p_th = th1_q * th2_q
    for i in range(d + 1):
        for j in range(d + 1):
            r0, c0 = 2 * i, 2 * j
            k0, k1, k2 = i + j, i + j + 1, i + j + 2
            for di in range(2):
                for dj in range(2):
                    v0 = Pq[r0 + di][c0 + dj] + Rq[r0 + di][c0 + dj] + p_th * Gq[r0 + di][c0 + dj]
                    v1 = Qq[r0 + di][c0 + dj] - Rq[r0 + di][c0 + dj] - s_th * Gq[r0 + di][c0 + dj]
                    v2 = Gq[r0 + di][c0 + dj]
                    if di == 0 and dj == 0:
                        ca[k0] += v0; ca[k1] += v1; ca[k2] += v2
                    elif di == 1 and dj == 1:
                        cc[k0] += v0; cc[k1] += v1; cc[k2] += v2
                    else:
                        cb[k0] += v0; cb[k1] += v1; cb[k2] += v2

    margin = cc[0]
    for k in range(K + 1):
        margin += ca[k] * a_q[k] + abs(ca[k]) * eps_a_q[k]
        margin += cb[k] * b_q[k] + abs(cb[k]) * eps_b_q[k]
    if c_q is None:
        for k in range(1, K + 1):
            if cc[k] > 0:
                margin += cc[k]
    else:
        for k in range(1, K + 1):
            margin += cc[k] * c_q[k] + abs(cc[k]) * eps_c_q[k]

    return (margin < 0), margin, {"ca": ca, "cb": cb, "cc": cc}


def solve_edge_dual(a_vals, b_vals, eps_b, theta, d, eps_a=None, use_c=None, return_cert=False):
    """
    Solve single-edge dual SDP (with exact Theorem 4.1 objective including max(0, cc_k))
    and validate candidate strictly in exact rational arithmetic.
    """
    K = 2 * d + 1
    m_sz = 2 * (d + 1)
    if eps_a is None:
        eps_a = [0.0] + [1e-10] * K  # a_0 = 1 exact; a_k >= 1 encloses float64 + quantization
    P = cp.Variable((m_sz, m_sz), symmetric=True)
    Q = cp.Variable((m_sz, m_sz), symmetric=True)
    R = cp.Variable((m_sz, m_sz), symmetric=True)
    ca = [0] * (K + 1); cb = [0] * (K + 1); cc = [0] * (K + 1)
    for i in range(d + 1):
        for j in range(d + 1):
            r0, c0 = 2 * i, 2 * j
            k0, k1 = i + j, i + j + 1
            M0 = P[r0:r0 + 2, c0:c0 + 2] + theta * R[r0:r0 + 2, c0:c0 + 2]
            M1 = Q[r0:r0 + 2, c0:c0 + 2] - R[r0:r0 + 2, c0:c0 + 2]
            ca[k0] = ca[k0] + M0[0, 0]; ca[k1] = ca[k1] + M1[0, 0]
            cb[k0] = cb[k0] + M0[0, 1] + M0[1, 0]; cb[k1] = cb[k1] + M1[0, 1] + M1[1, 0]
            cc[k0] = cc[k0] + M0[1, 1]; cc[k1] = cc[k1] + M1[1, 1]
    constraints = [P >> 0, Q >> 0, R >> 0, cp.trace(P) + cp.trace(Q) + cp.trace(R) <= 1.0]
    obj = cc[0]
    for k in range(K + 1):
        obj += ca[k] * a_vals[k] + (eps_a[k] * cp.abs(ca[k]) if eps_a[k] > 0 else 0)
        obj += cb[k] * b_vals[k] + (eps_b[k] * cp.abs(cb[k]) if eps_b[k] > 0 else 0)
    if use_c is None:
        for k in range(1, K + 1):
            obj += cp.pos(cc[k])
    else:
        c_vals, eps_c = use_c
        for k in range(1, K + 1):
            obj += cc[k] * c_vals[k] + (eps_c[k] * cp.abs(cc[k]) if eps_c[k] > 0 else 0)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    except Exception:
        prob.solve(solver=cp.CLARABEL)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or prob.value is None or prob.value >= -1e-9:
        return (False, 0.0, None) if return_cert else (False, 0.0)

    okP, Pq, _ = to_rational_psd(P.value)
    okQ, Qq, _ = to_rational_psd(Q.value)
    okR, Rq, _ = to_rational_psd(R.value)
    if not (okP and okQ and okR):
        return (False, 0.0, None) if return_cert else (False, 0.0)

    a_q = [Fraction(1) if k == 0 else Fraction(float(a_vals[k])).limit_denominator(10**12) for k in range(K + 1)]
    ea_q = [Fraction(0) if eps_a[k] == 0 else Fraction(float(eps_a[k])).limit_denominator(10**12) for k in range(K + 1)]
    b_q = [Fraction(float(b_vals[k])).limit_denominator(10**12) for k in range(K + 1)]
    eb_q = [Fraction(float(eps_b[k])).limit_denominator(10**12) for k in range(K + 1)]
    th_q = Fraction(float(theta)).limit_denominator(10**9)
    if use_c is None:
        c_q, ec_q = None, None
    else:
        c_q = [Fraction(1) if k == 0 else Fraction(float(use_c[0][k])).limit_denominator(10**12) for k in range(K + 1)]
        ec_q = [Fraction(float(use_c[1][k])).limit_denominator(10**12) for k in range(K + 1)]

    ok, margin_q, _ = verify_edge_certificate_exact(
        Pq, Qq, Rq, a_q, b_q, eb_q, th_q, d, eps_a_q=ea_q, c_q=c_q, eps_c_q=ec_q
    )
    cert = None
    if ok and return_cert:
        cert = {
            "kind": "single_edge",
            "degree": d,
            "theta": f"{th_q.numerator}/{th_q.denominator}",
            "a": [f"{x.numerator}/{x.denominator}" for x in a_q],
            "eps_a": [f"{x.numerator}/{x.denominator}" for x in ea_q],
            "b": [f"{x.numerator}/{x.denominator}" for x in b_q],
            "eps_b": [f"{x.numerator}/{x.denominator}" for x in eb_q],
            "c": None if c_q is None else [f"{x.numerator}/{x.denominator}" for x in c_q],
            "eps_c": None if ec_q is None else [f"{x.numerator}/{x.denominator}" for x in ec_q],
            "P": serialize_q_mat(Pq),
            "Q": serialize_q_mat(Qq),
            "R": serialize_q_mat(Rq),
            "exact_margin": f"{margin_q.numerator}/{margin_q.denominator}",
            "margin_float": float(margin_q),
        }
    return (ok, float(margin_q), cert) if return_cert else (ok, float(margin_q))


def solve_gap_dual(a_vals, b_vals, eps_b, th1, th2, d, eps_a=None, use_c=None, return_cert=False):
    """
    Solve two-edge spectral-gap dual SDP (with exact Theorem 4.1 objective including max(0, cc_k))
    and validate candidate strictly in exact rational arithmetic.
    """
    K = 2 * d + 2
    m_sz = 2 * (d + 1)
    if eps_a is None:
        eps_a = [0.0] + [1e-10] * K
    P = cp.Variable((m_sz, m_sz), symmetric=True)
    Q = cp.Variable((m_sz, m_sz), symmetric=True)
    R = cp.Variable((m_sz, m_sz), symmetric=True)
    G = cp.Variable((m_sz, m_sz), symmetric=True)
    ca = [0] * (K + 1); cb = [0] * (K + 1); cc = [0] * (K + 1)
    for i in range(d + 1):
        for j in range(d + 1):
            r0, c0 = 2 * i, 2 * j
            k0, k1, k2 = i + j, i + j + 1, i + j + 2
            M0 = P[r0:r0 + 2, c0:c0 + 2] + R[r0:r0 + 2, c0:c0 + 2] + (th1 * th2) * G[r0:r0 + 2, c0:c0 + 2]
            M1 = Q[r0:r0 + 2, c0:c0 + 2] - R[r0:r0 + 2, c0:c0 + 2] - (th1 + th2) * G[r0:r0 + 2, c0:c0 + 2]
            M2 = G[r0:r0 + 2, c0:c0 + 2]
            for kk, Mk in [(k0, M0), (k1, M1), (k2, M2)]:
                ca[kk] = ca[kk] + Mk[0, 0]
                cb[kk] = cb[kk] + Mk[0, 1] + Mk[1, 0]
                cc[kk] = cc[kk] + Mk[1, 1]
    constraints = [P >> 0, Q >> 0, R >> 0, G >> 0,
                   cp.trace(P) + cp.trace(Q) + cp.trace(R) + cp.trace(G) <= 1.0]
    obj = cc[0]
    for k in range(K + 1):
        obj += ca[k] * a_vals[k] + (eps_a[k] * cp.abs(ca[k]) if eps_a[k] > 0 else 0)
        obj += cb[k] * b_vals[k] + (eps_b[k] * cp.abs(cb[k]) if eps_b[k] > 0 else 0)
    if use_c is None:
        for k in range(1, K + 1):
            obj += cp.pos(cc[k])
    else:
        c_vals, eps_c = use_c
        for k in range(1, K + 1):
            obj += cc[k] * c_vals[k] + (eps_c[k] * cp.abs(cc[k]) if eps_c[k] > 0 else 0)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    except Exception:
        prob.solve(solver=cp.CLARABEL)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or prob.value is None or prob.value >= -1e-9:
        return (False, 0.0, None) if return_cert else (False, 0.0)

    okP, Pq, _ = to_rational_psd(P.value)
    okQ, Qq, _ = to_rational_psd(Q.value)
    okR, Rq, _ = to_rational_psd(R.value)
    okG, Gq, _ = to_rational_psd(G.value)
    if not (okP and okQ and okR and okG):
        return (False, 0.0, None) if return_cert else (False, 0.0)

    a_q = [Fraction(1) if k == 0 else Fraction(float(a_vals[k])).limit_denominator(10**12) for k in range(K + 1)]
    ea_q = [Fraction(0) if eps_a[k] == 0 else Fraction(float(eps_a[k])).limit_denominator(10**12) for k in range(K + 1)]
    b_q = [Fraction(float(b_vals[k])).limit_denominator(10**12) for k in range(K + 1)]
    eb_q = [Fraction(float(eps_b[k])).limit_denominator(10**12) for k in range(K + 1)]
    th1_q = Fraction(float(th1)).limit_denominator(10**9)
    th2_q = th2 if isinstance(th2, Fraction) else Fraction(float(th2)).limit_denominator(10**9)
    if use_c is None:
        c_q, ec_q = None, None
    else:
        c_q = [Fraction(1) if k == 0 else Fraction(float(use_c[0][k])).limit_denominator(10**12) for k in range(K + 1)]
        ec_q = [Fraction(float(use_c[1][k])).limit_denominator(10**12) for k in range(K + 1)]

    ok, margin_q, _ = verify_gap_certificate_exact(
        Pq, Qq, Rq, Gq, a_q, b_q, eb_q, th1_q, th2_q, d, eps_a_q=ea_q, c_q=c_q, eps_c_q=ec_q
    )
    cert = None
    if ok and return_cert:
        cert = {
            "kind": "two_edge_gap",
            "degree": d,
            "theta1": f"{th1_q.numerator}/{th1_q.denominator}",
            "theta2": f"{th2_q.numerator}/{th2_q.denominator}",
            "a": [f"{x.numerator}/{x.denominator}" for x in a_q],
            "eps_a": [f"{x.numerator}/{x.denominator}" for x in ea_q],
            "b": [f"{x.numerator}/{x.denominator}" for x in b_q],
            "eps_b": [f"{x.numerator}/{x.denominator}" for x in eb_q],
            "c": None if c_q is None else [f"{x.numerator}/{x.denominator}" for x in c_q],
            "eps_c": None if ec_q is None else [f"{x.numerator}/{x.denominator}" for x in ec_q],
            "P": serialize_q_mat(Pq),
            "Q": serialize_q_mat(Qq),
            "R": serialize_q_mat(Rq),
            "G": serialize_q_mat(Gq),
            "exact_margin": f"{margin_q.numerator}/{margin_q.denominator}",
            "margin_float": float(margin_q),
        }
    return (ok, float(margin_q), cert) if return_cert else (ok, float(margin_q))


def bisect_edge_bound(a_vals, b_vals, eps_b, d, lo=0.0, hi=0.95, iters=22, eps_a=None, use_c=None):
    """
    Bisect for the largest rational theta in [lo, hi] certified by an exact-Q dual certificate.
    Returns (certified_theta_float, best_cert). If no certificate succeeds on (lo, hi],
    checks lo itself; if still uncertified, returns (0.0, None) fail-closed.
    """
    best_cert = None
    best_th = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok, _, cert = solve_edge_dual(a_vals, b_vals, eps_b, mid, d, eps_a=eps_a, use_c=use_c, return_cert=True)
        if ok:
            lo = mid
            best_cert = cert
            best_th = float(Fraction(cert["theta"]))
        else:
            hi = mid
    if best_cert is None:
        ok, _, cert = solve_edge_dual(a_vals, b_vals, eps_b, lo, d, eps_a=eps_a, use_c=use_c, return_cert=True)
        if ok:
            best_cert = cert
            best_th = float(Fraction(cert["theta"]))
    return best_th, best_cert
