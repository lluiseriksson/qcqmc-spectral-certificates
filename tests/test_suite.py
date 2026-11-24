"""
Comprehensive Gate-2 verification & falsification test suite (pytest).
"""
import glob
import json
import os
import sys
from fractions import Fraction
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.hubbard import build_two_site_hubbard, build_hubbard_chain, gershgorin_box, slater_vec
from src.surrogates import backflow_overlap, antiholomorphic_greens_defect, plucker_distance_and_projection
from src.shadows import simulate_linear_shadow_moments, exact_shadow_variance_bound
from src.hausdorff_sdp import (
    exact_ldlt_psd,
    deserialize_q_mat,
    verify_edge_certificate_exact,
    verify_gap_certificate_exact,
)


def test_linear_projector_collapse():
    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    P = Phi @ np.linalg.inv(Phi.conj().T @ Phi) @ Phi.conj().T
    G0 = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    A = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    lhs = (G0 + P @ A).conj().T @ Phi
    rhs = (G0 + A).conj().T @ Phi
    assert np.allclose(lhs, rhs, atol=1e-13)


def test_plucker_distance_and_projection_sharpness():
    rng = np.random.default_rng(2)
    G0 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.complex128) / np.sqrt(2.0)
    A_bf = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]], dtype=np.complex128) / np.sqrt(2.0)
    M_obs = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
    walkers = [rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)) for _ in range(10)]
    f = lambda Ph: backflow_overlap(Ph, G0, A_bf, M_obs, 0.5)
    res = plucker_distance_and_projection(f, walkers)
    assert res["eta_pl"] > 0.1
    assert abs(res["eta_pl"] - res["dist_to_proj"]) < 1e-12
    for _ in range(50):
        psi = rng.normal(size=6) + 1j * rng.normal(size=6)
        v = (res["W"].conj().T @ psi).conj()
        assert np.linalg.norm(res["f_S"] - v) >= res["eta_pl"] - 1e-12


def test_exact_ldlt_psd_and_falsifier():
    # Valid PSD matrix
    ok, _ = exact_ldlt_psd([[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]])
    assert ok
    # Rank-deficient PSD matrix
    ok2, _ = exact_ldlt_psd([[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]])
    assert ok2
    # Indefinite matrix must be rejected
    ok3, _ = exact_ldlt_psd([[Fraction(1), Fraction(2)], [Fraction(2), Fraction(1)]])
    assert not ok3
    # Zero diagonal with nonzero off-diagonal must be rejected
    ok4, _ = exact_ldlt_psd([[Fraction(0), Fraction(1)], [Fraction(1), Fraction(1)]])
    assert not ok4
    # Asymmetric matrix MUST be rejected (adversarial regression test!)
    ok5, _ = exact_ldlt_psd([[Fraction(1), Fraction(100)], [Fraction(0), Fraction(1)]])
    assert not ok5
    # Non-square matrix must be rejected
    ok6, _ = exact_ldlt_psd([[Fraction(1), Fraction(0)], [Fraction(0)]])
    assert not ok6


def test_complex_shadow_variance_bound_purely_imaginary():
    """Verify exact 3-design bound Var <= 4 a_{2k} holds even when b_k is purely imaginary."""
    rng = np.random.default_rng(42)
    psi_sur = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    psi_T = np.array([1j, 0.0, 0.0, 0.0], dtype=np.complex128)  # b_0 = i (purely imaginary!)
    T = np.eye(4, dtype=np.complex128)
    samps = simulate_linear_shadow_moments(psi_sur, psi_T, T, K_max=0, n_snapshots=4000, rng=rng)
    # Exact 3-design variance for D=4, dim=5, a_0=1, |b_0|=1, Re(b_0)=0 is (6/7)*4 = 3.4286 > 3.0!
    emp_var = float(np.var(samps[:, 0], ddof=1))
    assert emp_var < exact_shadow_variance_bound([1.0], complex_states=True)[0] + 0.25
    assert abs(float(np.mean(samps[:, 0]))) < 0.12  # unbiased around Re(i) = 0


def test_exported_rational_certificates_exact_replay():
    cert_dir = os.path.join(ROOT, "results", "rational_certificates")
    files = sorted(glob.glob(os.path.join(cert_dir, "*.cert.json")))
    assert len(files) >= 5, f"Expected >= 5 exact rational certificates, found {len(files)}"
    for fp in files:
        with open(fp) as f:
            c = json.load(f)
        Pq = deserialize_q_mat(c["P"])
        Qq = deserialize_q_mat(c["Q"])
        Rq = deserialize_q_mat(c["R"])
        a_q = [Fraction(x) for x in c["a"]]
        ea_q = [Fraction(x) for x in c["eps_a"]] if "eps_a" in c else None
        b_q = [Fraction(x) for x in c["b"]]
        eb_q = [Fraction(x) for x in c["eps_b"]]
        c_q = None if c["c"] is None else [Fraction(x) for x in c["c"]]
        ec_q = None if c["eps_c"] is None else [Fraction(x) for x in c["eps_c"]]
        if c["kind"] == "single_edge":
            ok, margin, _ = verify_edge_certificate_exact(
                Pq, Qq, Rq, a_q, b_q, eb_q, Fraction(c["theta"]), c["degree"],
                eps_a_q=ea_q, c_q=c_q, eps_c_q=ec_q
            )
            assert ok, f"Exact edge cert failed: {fp}"
            assert margin == Fraction(c["exact_margin"])
            assert margin < 0
        elif c["kind"] == "two_edge_gap":
            Gq = deserialize_q_mat(c["G"])
            ok, margin, _ = verify_gap_certificate_exact(
                Pq, Qq, Rq, Gq, a_q, b_q, eb_q, Fraction(c["theta1"]), Fraction(c["theta2"]),
                c["degree"], eps_a_q=ea_q, c_q=c_q, eps_c_q=ec_q
            )
            assert ok, f"Exact gap cert failed: {fp}"
            assert margin == Fraction(c["exact_margin"])
            assert margin < 0


def test_gershgorin_enclosure_soundness():
    H4, _, _ = build_hubbard_chain(L=4, t=1.0, U=4.0)
    ev = np.linalg.eigvalsh(H4)
    lo, hi = gershgorin_box(H4)
    assert lo <= ev[0] + 1e-12 and ev[-1] <= hi + 1e-12
