"""Phase 4 grid driver.

Three grids:
  1. Headline:  6 defenses x 5 schemes x 4 attack profiles at canonical (N=20, sigma=0.15).
                Feeds the composition heatmap (the paper figure).
  2. T4 diag:   D3 x {T1, T4_EigenTrust} x {AP1, AP4, AP5, AP6} x N={20,50,100}.
                Shows the two-part T4 failure mode.
  3. sigma sweep: {D3, D6} x {T1, T_max, T_meanvar_z} x {AP4, AP5, AP6c+active} x sigma in [0.05..0.25].
                Shows T_meanvar_z robustness across noise.

Saves three CSVs to /mnt/user-data/outputs/. Reproducible (seeded RNG).
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(__file__))
from itertools import product
from collections import defaultdict
import numpy as np
from phase4_sim import (run_cell, s_b, gen_er_digraph, assign_roles,
                        assign_edge_signals, compute_metrics, midpoint)
from trust_schemes import (T1_mean, T_max, T_median, make_T_meanvar_z,
                            T4_EigenTrust, T5_TFLDT)


def run_clique_cell(defense, profile, scheme, N=20, p_edge=0.3, rho_atk=0.3,
                    sigma=0.15, n_seeds=100, inclique=None, base_seed=0):
    """run_cell variant with forced attacker-clique topology (standard threat model)."""
    T_det = midpoint(defense)
    acc = defaultdict(list)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        edges = gen_er_digraph(N, p_edge, rng)
        roles = assign_roles(N, rho_atk, rng)
        attackers = [i for i in range(N) if roles[i] == 1]
        existing = set(edges)
        for i in attackers:
            for j in attackers:
                if i != j and (i, j) not in existing:
                    edges.append((i, j))
        sig = assign_edge_signals(edges, roles, defense, profile, rng,
                                  sigma=sigma, inclique=inclique)
        scores = scheme(edges, sig, N)
        m = compute_metrics(scores, roles, T_det)
        for k, v in m.items():
            if not (isinstance(v, float) and np.isnan(v)):
                acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def get_scheme(name, defense, sigma):
    if name == "T_meanvar_z":
        return make_T_meanvar_z(s_b(defense), sigma)
    return {"T1_mean": T1_mean, "T_max": T_max, "T_median": T_median,
            "T4_EigenTrust": T4_EigenTrust, "T5_TFLDT": T5_TFLDT}[name]


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_headline_grid(out_path):
    DEFENSES = ["D1_RF", "D2_CNN", "D3_RHMD", "D4_MTD", "D5_Stochastic", "D6_DRL"]
    SCHEMES  = ["T1_mean", "T_max", "T_meanvar_z", "T4_EigenTrust", "T5_TFLDT"]
    PROFILES = [
        ("AP1",            "AP1", {}, False),
        ("AP4_phi030",     "AP4", {"phi": 0.30}, False),
        ("AP5_phi100_p10", "AP5", {"phi": 1.0, "p": 10}, False),
        ("AP6c_active",    "AP6", {"inclique": 0.0}, True),
    ]
    SIGMA, N, N_SEEDS = 0.15, 20, 100
    rows = []
    cells = list(product(DEFENSES, SCHEMES, PROFILES))
    t0 = time.time()
    for i, (defense, sname, (label, prof, kw, clique)) in enumerate(cells):
        sch = get_scheme(sname, defense, SIGMA)
        if clique:
            r = run_clique_cell(defense, prof, sch, N=N, sigma=SIGMA, n_seeds=N_SEEDS, **kw)
        else:
            r = run_cell(defense, prof, scheme=sch, N=N, sigma=SIGMA, n_seeds=N_SEEDS, **kw)
        rows.append({
            'defense': defense, 'scheme': sname, 'profile': label,
            'N': N, 'sigma': SIGMA,
            'phi': kw.get('phi'), 'p': kw.get('p'),
            'inclique': kw.get('inclique'), 'clique_forced': clique,
            'n_seeds': N_SEEDS,
            'AUC': r.get('AUC'),
            'tpr_at_fpr01': r.get('tpr_at_fpr01'),
            'gap': r.get('gap'),
            'stealth': r.get('stealth'),
            'worst_vis': r.get('worst_vis'),
        })
        if (i+1) % 20 == 0 or i == len(cells)-1:
            print(f"  headline [{i+1}/{len(cells)}] t={time.time()-t0:.0f}s")
    write_csv(out_path, rows)
    print(f"  -> {out_path}  ({len(rows)} cells)")


def build_T4_diagnostic_grid(out_path):
    SIGMA, N_SEEDS = 0.15, 80
    PROFILES = [("AP1", "AP1", {}),
                ("AP4_phi030", "AP4", {"phi": 0.30}),
                ("AP5_phi100_p10", "AP5", {"phi": 1.0, "p": 10}),
                ("AP6", "AP6", {})]
    rows = []
    for N in [20, 50, 100]:
        for label, prof, kw in PROFILES:
            for sname, sch in [("T1_mean", T1_mean), ("T4_EigenTrust", T4_EigenTrust)]:
                r = run_cell("D3_RHMD", prof, scheme=sch, N=N, sigma=SIGMA, n_seeds=N_SEEDS, **kw)
                rows.append({
                    'defense': 'D3_RHMD', 'scheme': sname, 'profile': label,
                    'N': N, 'sigma': SIGMA,
                    'phi': kw.get('phi'), 'p': kw.get('p'),
                    'n_seeds': N_SEEDS,
                    'AUC': r.get('AUC'),
                    'tpr_at_fpr01': r.get('tpr_at_fpr01'),
                    'gap': r.get('gap'),
                })
    write_csv(out_path, rows)
    print(f"  -> {out_path}  ({len(rows)} cells)")


def build_sigma_sweep_grid(out_path):
    N_SEEDS = 100
    PROFILES = [
        ("AP4_phi030", "AP4", {"phi": 0.30}, False),
        ("AP5_phi100_p10", "AP5", {"phi": 1.0, "p": 10}, False),
        ("AP6c_active", "AP6", {"inclique": 0.0}, True),
    ]
    rows = []
    for SIGMA in [0.05, 0.10, 0.15, 0.20, 0.25]:
        for sname in ["T1_mean", "T_max", "T_meanvar_z"]:
            for defense in ["D3_RHMD", "D6_DRL"]:
                sch = get_scheme(sname, defense, SIGMA)
                for label, prof, kw, clique in PROFILES:
                    if clique:
                        r = run_clique_cell(defense, prof, sch, N=20, sigma=SIGMA, n_seeds=N_SEEDS, **kw)
                    else:
                        r = run_cell(defense, prof, scheme=sch, N=20, sigma=SIGMA, n_seeds=N_SEEDS, **kw)
                    rows.append({
                        'defense': defense, 'scheme': sname, 'profile': label,
                        'N': 20, 'sigma': SIGMA,
                        'phi': kw.get('phi'), 'p': kw.get('p'),
                        'inclique': kw.get('inclique'),
                        'n_seeds': N_SEEDS,
                        'AUC': r.get('AUC'),
                        'tpr_at_fpr01': r.get('tpr_at_fpr01'),
                    })
    write_csv(out_path, rows)
    print(f"  -> {out_path}  ({len(rows)} cells)")


if __name__ == "__main__":
    out_dir = "/mnt/user-data/outputs"
    print("Grid 1: headline composition heatmap data")
    build_headline_grid(f"{out_dir}/phase4_grid.csv")
    print("Grid 2: T4 EigenTrust diagnostic data")
    build_T4_diagnostic_grid(f"{out_dir}/phase4_T4_diag.csv")
    print("Grid 3: sigma sweep data")
    build_sigma_sweep_grid(f"{out_dir}/phase4_sigma_sweep.csv")
    print("\nGrid complete.")
