"""Exact fermionic Hubbard Hamiltonians and Slater-determinant algebra."""
import numpy as np
from itertools import combinations


def slater_vec(Phi: np.ndarray):
    """Map an N x eta orbital matrix Phi to its Plucker vector in wedge^eta C^N."""
    N, eta = Phi.shape
    basis = list(combinations(range(N), eta))
    v = np.zeros(len(basis), dtype=np.complex128)
    for idx, rows in enumerate(basis):
        v[idx] = np.linalg.det(Phi[list(rows), :])
    return v, basis


def build_two_site_hubbard(t: float = 1.0, U: float = 4.0):
    """
    Minimal correlated fermionic system: 2-site Hubbard model (N=4 spin-orbitals, eta=2, dim=6).
    Factorized exactly as H = B0 + (U/2) N0^2 + (U/2) N1^2 matching Jiang et al. Eq. (S2.9).
    """
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
    assert np.allclose(H, H.conj().T, atol=1e-13)
    return H, B0, B_list, basis


def build_hubbard_chain(L: int = 4, t: float = 1.0, U: float = 4.0, pbc: bool = False):
    """
    Exact 1D Hubbard chain at half-filling (N_up = N_dn = L/2).
    For L=4: 8 spin-orbitals, dim = C(4,2)^2 = 36.
    """
    assert L % 2 == 0
    up_states = list(combinations(range(L), L // 2))
    dn_states = list(combinations(range(L), L // 2))
    n_up, n_dn = len(up_states), len(dn_states)
    up_idx = {s: i for i, s in enumerate(up_states)}
    dn_idx = {s: i for i, s in enumerate(dn_states)}
    D = n_up * n_dn
    H = np.zeros((D, D), dtype=np.float64)

    edges = [(i, i + 1) for i in range(L - 1)] + ([(L - 1, 0)] if pbc else [])

    def hop_mat(states, idx_map):
        M = np.zeros((len(states), len(states)), dtype=np.float64)
        for a, st in enumerate(states):
            st_set = set(st)
            for i, j in edges:
                for src, dst in [(i, j), (j, i)]:
                    if src in st_set and dst not in st_set:
                        lo, hi = min(src, dst), max(src, dst)
                        parity = sum(1 for x in st if lo < x < hi)
                        new_st = tuple(sorted((st_set - {src}) | {dst}))
                        M[idx_map[new_st], a] += -t * ((-1.0) ** parity)
        return M

    H += np.kron(hop_mat(up_states, up_idx), np.eye(n_dn))
    H += np.kron(np.eye(n_up), hop_mat(dn_states, dn_idx))
    for iu, su in enumerate(up_states):
        su_set = set(su)
        for idn, sd in enumerate(dn_states):
            H[iu * n_dn + idn, iu * n_dn + idn] += U * len(su_set & set(sd))
    return H, up_states, dn_states


def gershgorin_box(H: np.ndarray):
    """Oracle-free certified spectral enclosure [E_min_box, E_max_box] via Gershgorin's theorem."""
    diag = np.real(np.diag(H))
    off = np.sum(np.abs(H), axis=1) - np.abs(diag)
    return float(np.min(diag - off)), float(np.max(diag + off))
