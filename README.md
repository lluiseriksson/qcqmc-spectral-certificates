# Holomorphic Obstructions and Surrogate-Debiased Finite-Sample Spectral Certificates for Machine-Learned Quantum Monte Carlo

Self-contained research artifact (LaTeX manuscript, zero-dependency Lean 4 formalization, exact rational symmetry-checked LDLᵀ certificate verifier, and reproducible Python campaign) prepared for **AIRR.SCIENCE** (`ARR-SCREEN-1.0`).

## Main Contributions

1. **Theorem 2.1 (Holomorphic Characterization on the Complex Grassmannian):** A $C^1$ $GL(\eta,\mathbb{C})$-equivariant overlap surrogate $f(\Phi U) = \det(U) f(\Phi)$ ($1 \le \eta < N$) represents a physical fermionic trial state $|\Psi\rangle \in \bigwedge^\eta \mathbb{C}^N$ if and only if $\bar{\partial} f \equiv 0$, equivalently if and only if the anti-holomorphic Green's defect $G^{\bar{\partial}}(\Phi)$ vanishes identically, equivalently if and only if $f$ annihilates every finite linear Slater-determinant syzygy. For $\eta = N$, equivariance alone forces $f(\Phi) = c\det(\Phi)$.
2. **Remark 2.3 (Left-Projector Collapse) & Theorems 2.4–2.5 (Exact Plücker-Kernel Distance and Mixed-Estimator Bias Decomposition):** Left-projector affine backflow $G_0 + P_\Phi A$ collapses identically to a fixed Slater determinant, whereas general affine-trace or nonlinear backflow NQS leave $H^0(\mathrm{Gr}(\eta,N),\mathcal{O}(1))$, exhibit exact minimum distance $\eta_{\mathrm{Pl}}(f;S) = \|(I - S S^+) f_S^*\|_2$ to all physical states (attained by $|\Psi_f^{\mathrm{proj}}\rangle = W S^+ f_S^*$), and incur representation-dependent mixed-estimator bias even on zero-walker-error decompositions of the exact ground state $|\Psi_0\rangle$ (including unphysical energies $E_{\mathrm{mix}}^\theta < E_0$).
3. **Theorem 3.1 (Universal Mixed-Energy Variational-Boost Law):** For any Hermitian $H \preceq E_{\max} I$ and unit states $|\Psi_{\mathrm{sur}}\rangle, |\Psi_T\rangle$ with $|b_0| < 1$ and $E_{\mathrm{sur}} < E_{\max}$,
   $$E_0 \le E_{\mathrm{sur}} - \frac{|\langle \Psi_{\mathrm{sur}} | (E_{\mathrm{sur}} I - H) | \Psi_T \rangle|^2}{(E_{\max} - E_{\mathrm{sur}})(1 - |b_0|^2)},$$
   which is sharp and turns the non-variational mixed overlap/energy data of QC-QMC into a strict variational upper bound on $E_0$ without measuring $\langle \Psi_T | H | \Psi_T \rangle$.
4. **Theorem 4.1 & Proposition 4.2 (Surrogate-Anchored Block Hausdorff Certificates & Authentic Born-Rule PPI Debiasing):** Generalizes Theorem 3.1 to arbitrary degree $d \ge 0$ for both single-edge ground-state bounds and oracle-free two-edge visible spectral-gap certificates using real symmetric PSD dual matrices, classical moments $a_k$, and distribution-free Median-of-Means confidence boxes on linear shadow cross-moments $\operatorname{Re}(b_k)$ (with exact 3-design bound $\operatorname{Var} < 4 a_{2k}$ for complex states), verified in exact rational arithmetic ($\mathbb{Q}$) via symmetry-checked pivoted LDLᵀ over explicit rational moment enclosures.

## Single-Command Gate-2 Verification

```powershell
# Verify Lean 4 (Level L2: 0 sorry, standard axioms), pytest + exact rational LDL^T replay, and SHA-256 manifest:
python scripts/verify_all.py

# Regenerate all numerical results, exact rational certificates, and figures from scratch:
python scripts/run_all_experiments.py
python scripts/make_figures.py
```

## Formal Verification Status (Lean 4)
- Toolchain: `leanprover/lean4:v4.35.0` (zero external dependencies; builds cleanly via `lake build` in ~2s).
- Formalized theorems (16 audited symbols in `lean/AuditAxioms.lean`): algebraic left-projector collapse, real & complex (`ComplexK`) Plücker orthogonal-projection distance floor, mixed-estimator bias decomposition & persistence, $2\times 2$ variational-boost law & sharpness identity, real & complex spectral-measure single-edge and two-edge Block Hausdorff conic-dual soundness, worst-case box transfer, and finite-population PPI expectation identity.
- Axioms audited via `lean/AuditAxioms.lean`: **0 `sorry`**, **0 `Classical.choice`**, standard Lean core axioms (`propext`, `Quot.sound`) only.
