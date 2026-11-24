"""
Generate publication-quality vector PDF and PNG figures directly from verified JSON outputs.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def make_fig1():
    with open(os.path.join(RES_DIR, "experiment1_backflow_obstruction.json")) as f:
        d = json.load(f)
    E0 = d["E0_exact"]
    rows = d["sweep"]
    kappas = [r["kappa"] for r in rows]
    E_comp = [r["E_mix_comp"] for r in rows]
    E_rot = [r["E_mix_rot"] for r in rows]
    E_proj = [r["E_var_projected"] for r in rows]
    eta_pl = [r["eta_plucker"] for r in rows]
    gdbar_rot = [r["G_dbar_norm_rot"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=300)

    ax1.axhline(E0, color="black", linestyle="-", linewidth=1.3, label=f"Exact $E_0 = {E0:.4f}$")
    ax1.plot(kappas, E_comp, "o-", color="#1f77b4", markersize=5, label=r"$E_{\mathrm{mix}}^\theta$ (comp. basis decomp.)")
    ax1.plot(kappas, E_rot, "s-", color="#d62728", markersize=5, label=r"$E_{\mathrm{mix}}^\theta$ (rotated decomp. of $|\Psi_0\rangle$)")
    ax1.plot(kappas, E_proj, "^--", color="#2ca02c", markersize=5, label=r"$E_{\mathrm{var}}(\Psi_f^{\mathrm{proj}})$ (restored)")
    ax1.fill_between(kappas, min(E_rot) - 0.12, E0, color="#d62728", alpha=0.07, label="Unphysical region ($E < E_0$)")
    ax1.set_xlabel(r"Backflow strength $\kappa$")
    ax1.set_ylabel("Energy (Ha / hopping units)")
    ax1.set_title("(a) Zero-Walker-Error Mixed-Energy Bias")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92)

    ax2.plot(kappas, eta_pl, "D-", color="#9467bd", markersize=5, label=r"Plücker-kernel witness $\eta_{\mathrm{Pl}}(f_\kappa; S)$")
    ax2.plot(kappas, gdbar_rot, "o--", color="#ff7f0e", markersize=5, label=r"Wirtinger defect $\|G^{\bar{\partial}}(\Phi_{\mathrm{rot}})\|_F$")
    ax2.set_xlabel(r"Backflow strength $\kappa$")
    ax2.set_ylabel("Non-physicality diagnostic")
    ax2.set_title("(b) Exact Non-Physicality Witnesses")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_backflow_bias_plucker.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig1_backflow_bias_plucker.png"))
    plt.close(fig)


def make_fig2():
    with open(os.path.join(RES_DIR, "experiment3_hubbard_certificates.json")) as f:
        d = json.load(f)
    E0 = d["E0_exact"]
    E_sur = d["E_sur_RHF"]
    E_T = d["E_trial_exact"]
    tbl = d["energy_table"]

    eps_vals = sorted(set(r["eps_b"] for r in tbl))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7), dpi=300)

    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    for deg in [0, 1, 2]:
        sub = [r for r in tbl if r["degree"] == deg]
        sub.sort(key=lambda x: x["eps_b"])
        xs = [r["eps_b"] for r in sub]
        ys_anch = [r["E_anchored_ab_only"] for r in sub]
        ys_cls = [r["E_classical_krylov"] for r in sub]
        ys_full = [r["E_full_abc"] for r in sub]
        ax1.plot(xs, ys_anch, "o-", color=colors[deg], linewidth=1.6, markersize=5,
                 label=f"Anchored $(a,b)$ only, $d={deg}$")
        ax1.plot(xs, ys_cls, ":", color=colors[deg], linewidth=1.2, alpha=0.8)
        if deg == 2:
            ax1.plot(xs, ys_full, "s--", color="#9467bd", linewidth=1.2, markersize=4,
                     label=f"Full $(a,b,c)$, $d=2$")

    ax1.axhline(E0, color="black", linestyle="-", linewidth=1.3, label=f"Exact $E_0 = {E0:.4f}$")
    ax1.axhline(E_sur, color="gray", linestyle="-.", linewidth=1.0, label=f"RHF $E_{{\\mathrm{{sur}}}} = {E_sur:.4f}$")
    ax1.set_xlabel(r"Linear-shadow confidence half-width $\varepsilon_b$")
    ax1.set_ylabel(r"Certified upper bound $E_{\mathrm{cert}} \geq E_0$")
    ax1.set_title("(a) 4-Site Hubbard Certified Ground-State Bounds")
    ax1.legend(loc="center right", fontsize=7.8, frameon=True, framealpha=0.92)

    # Panel (b): Two-edge spectral gap certificate margins
    gaps = d["gap_tests"]
    E_cuts = [g["E_cut"] for g in gaps]
    m0 = [max(1e-10, -g["margin_eps0"]) if g["certified_eps0"] else np.nan for g in gaps]
    mn = [max(1e-10, -g["margin_eps002"]) if g["certified_eps002"] else np.nan for g in gaps]
    ax2.semilogy(E_cuts, m0, "o-", color="#1f77b4", markersize=5.5, label=r"Exact-$\mathbb{Q}$ gap margin ($\varepsilon_b = 0$)")
    ax2.semilogy(E_cuts, mn, "s--", color="#d62728", markersize=5.5, label=r"Exact-$\mathbb{Q}$ gap margin ($\varepsilon_b = 0.002$)")
    ax2.axvline(1.9110, color="black", linestyle=":", linewidth=1.3, label=r"True dominant singlet $E_8 = +1.911$")
    ax2.set_xlabel(r"Excited-state cutoff $E_{\mathrm{cut}}$ (certifying spectrum in $(E_{\mathrm{cert}}, E_{\mathrm{cut}})$)")
    ax2.set_ylabel(r"Certified negative dual margin $-\Phi_{\mathrm{gap}} > 0$")
    ax2.set_title("(b) Oracle-Free Spectral-Gap Certificates ($d=2$)")
    ax2.invert_xaxis()
    ax2.legend(loc="upper right", fontsize=7.8, frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_hubbard_energy_gap_bounds.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig2_hubbard_energy_gap_bounds.png"))
    plt.close(fig)


def make_fig3():
    with open(os.path.join(RES_DIR, "experiment4_shadow_coverage_ppi.json")) as f:
        d = json.load(f)
    h4 = d["shadow_audit_4site"]
    ppi = d["ppi_debiasing_authentic_shadows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=300)

    ks = np.arange(4)
    w = 0.28
    ax1.bar(ks - w, h4["sample_var_b"], width=w, color="#1f77b4", label="Empirical shadow variance (20k shots)")
    ax1.bar(ks, h4["exact_3design_var"], width=w, color="#2ca02c", alpha=0.85, label="Exact 3-design theory")
    ax1.bar(ks + w, h4["theoretical_var_bound_4a2k"], width=w, color="#ff7f0e", alpha=0.85,
            label=r"Universal bound $4 a_{2k}$")
    ax1.set_xticks(ks)
    ax1.set_xticklabels([f"$k={k}$" for k in ks])
    ax1.set_ylabel("Single-shot variance")
    ax1.set_title("(a) 4-Site Hubbard Born-Rule Shadow Variance ($D=36$)")
    ax1.legend(loc="upper right", fontsize=7.8, frameon=True, framealpha=0.92)

    methods = ["Biased Backflow\nNQS Only", "Raw Subsampled\nHaar Shadows", "PPI Debiased\n(Prop. 4.2)"]
    rmses = [ppi["surrogate_only_rmse"], ppi["raw_subsampling_rmse"], ppi["ppi_rmse"]]
    biases = [abs(ppi["surrogate_only_bias"]), abs(ppi["raw_subsampling_bias"]), abs(ppi["ppi_bias"])]
    x = np.arange(3)
    ax2.bar(x - 0.18, rmses, width=0.34, color=["#d62728", "#7f7f7f", "#2ca02c"], label=f"RMSE ({ppi['n_trials']} trials)")
    ax2.bar(x + 0.18, biases, width=0.34, color=["#f5a0a0", "#c7c7c7", "#98df8a"], hatch="//", edgecolor="black",
            linewidth=0.5, label="|Empirical Bias|")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("Estimation error")
    ax2.set_title(f"(b) Authentic Born-Rule PPI ({ppi['variance_reduction_ratio_raw_over_ppi']:.1f}x Var. Reduction)")
    ax2.legend(loc="upper right", fontsize=7.8, frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_finite_sample_coverage_ppi.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig3_finite_sample_coverage_ppi.png"))
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    print("Generated all 3 publication figures in paper/figures/.")
