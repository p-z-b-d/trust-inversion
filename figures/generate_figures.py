#!/usr/bin/env python3
"""
generate_figures.py — regenerate all paper figures from saved experiment results.

Run this AFTER run_v2.py has completed all 48 conditions.

Usage:
    python generate_figures.py

Reads:
    outputs/checkpoint_v2.csv
    outputs/stored_v2_small.pkl
    outputs/histories_v2/*.csv

Writes (to outputs/):
    fig_v2_real_data.{pdf,png}
    fig_v2_trajectories.{pdf,png}
    fig_v2_convergence.{pdf,png}
    fig_v2_separation.{pdf,png}
    fig_v2_scale.{pdf,png}
    fig_v2_conv_speed.{pdf,png}
    fig_v2_delay_violin.{pdf,png}
    fig_v2_single_edge.{pdf,png}
    fig_v2_robustness.{pdf,png}
    baseline_comparison_v2.csv
"""
import sys, os, pickle, math, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_pipeline_v2 import *
from trust_baselines import SubjectiveLogicTrust, BetaReputationTrust, EMATrust
from trust_eval_iot import NodeTrustEvaluator, BehaviorRecord

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
HIST_DIR   = os.path.join(OUTPUT_DIR, 'histories_v2')

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  Saved {name}')

# ── Load results ─────────────────────────────────────────────────────────────
print('Loading results and trust histories...')
results_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'checkpoint_v2.csv'))
stored     = pickle.load(open(os.path.join(OUTPUT_DIR, 'stored_v2_small.pkl'), 'rb'))

for key in stored:
    topo, N, profile = key
    hf = os.path.join(HIST_DIR, f'{topo}_N{N}_{profile}.csv')
    if os.path.exists(hf):
        hdf = pd.read_csv(hf)
        stored[key]['th'] = {nd: grp.sort_values('step')['trust'].tolist()
                             for nd, grp in hdf.groupby('node')}

# ── Patch SCALES to match what was run ───────────────────────────────────────
import experiment_pipeline_v2 as ep
ep.SCALES = [20, 100, 200, 500]

# ── Generate all figures ─────────────────────────────────────────────────────
print('Generating figures...')
save(fig_trust_trajectories(stored),             'fig_v2_trajectories')
save(fig_convergence_curves(stored, results_df),  'fig_v2_convergence')
save(fig_separation_heatmap(results_df),          'fig_v2_separation')
save(fig_convergence_speed(results_df),           'fig_v2_conv_speed')
save(fig_scale_trajectories(stored, results_df),  'fig_v2_scale')
save(fig_delay_vs_topology(stored),               'fig_v2_delay_violin')

# ── Real-data evaluation figure ───────────────────────────────────────────────
print('Generating real-data figure...')
from scipy.stats import lognorm, spearmanr
import matplotlib.gridspec as gridspec

real_df = load_real_data()
ev = NodeTrustEvaluator(theta=THETA, lam=LAMBDA, phi=PHI, xi=XI)
np.random.seed(42)
D_MAX = lognorm.ppf(0.95, s=LN_SIGMA, scale=math.exp(LN_MU))
real_df['timeliness'] = 1.0 - np.clip(
    np.random.lognormal(LN_MU, LN_SIGMA, len(real_df)) / D_MAX, 0, 1)
th_real, ft_real = run_trust(real_df, 20, ev)

node_abb = real_df.groupby('src')['abb'].mean().sort_index()
trust_vals = [ft_real[n] for n in range(20)]
abb_vals   = [node_abb[n] for n in range(20)]
r, p       = spearmanr(trust_vals, abb_vals)

fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
cmap = plt.cm.RdYlGn
norm = plt.Normalize(vmin=min(abb_vals), vmax=max(abb_vals))

ax1 = fig.add_subplot(gs[0, :])
for node in range(20):
    ax1.plot(range(1, len(th_real[node])+1), th_real[node],
             color=cmap(norm(node_abb[node])), lw=0.9, alpha=0.75)
ax1.axhline(0.8, color='#1a6b3c', lw=0.9, ls=':', alpha=0.6)
ax1.axhline(0.4, color='#c0392b', lw=0.9, ls=':', alpha=0.6)
ax1.set_xlabel('Interaction step'); ax1.set_ylabel('Global trust')
ax1.set_title('Per-node trust trajectories (real data, 20 nodes)')
ax1.set_ylim(0.35, 0.85)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
fig.colorbar(sm, ax=ax1, label='Mean abb', pad=0.01, shrink=0.8)

ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(abb_vals, trust_vals, c=[cmap(norm(a)) for a in abb_vals],
            s=60, zorder=3, edgecolors='#333', lw=0.5)
for n in range(20):
    ax2.annotate(str(n), (abb_vals[n], trust_vals[n]),
                 textcoords='offset points', xytext=(4,3), fontsize=6.5)
ax2.set_xlabel('Mean abb'); ax2.set_ylabel('Final trust')
ax2.set_title(f'Final trust vs mean abb\nr={r:.3f}, p={p:.3f}')

ax3 = fig.add_subplot(gs[1, 1])
ns = sorted(range(20), key=lambda n: ft_real[n])
ax3.barh(range(20), [ft_real[n] for n in ns],
         color=[cmap(norm(node_abb[n])) for n in ns], height=0.75)
ax3.set_yticks(range(20)); ax3.set_yticklabels([f'Node {n}' for n in ns], fontsize=7)
ax3.axvline(0.8, color='#1a6b3c', lw=0.9, ls=':')
ax3.axvline(0.4, color='#c0392b', lw=0.9, ls=':')
ax3.set_xlabel('Final trust'); ax3.set_title('Final trust per node (sorted)')
fig.suptitle('Real data evaluation — 4,057 interactions, 20 nodes', y=1.01)
save(fig, 'fig_v2_real_data')

# ── Single-edge comparison figure ────────────────────────────────────────────
print('Generating single-edge and robustness figures...')
ITERS = 120
patterns = {
    'Pattern 1\n(Always benign)':    lambda t: (0.92, 0.90),
    'Pattern 2\n(Always malicious)': lambda t: (0.08, 0.15),
    'Pattern 3\n(Alternating 1:1)':  lambda t: (0.92,0.90) if t%2==0 else (0.08,0.15),
    'Pattern 5\n(Sporadic 3:1)':     lambda t: (0.08,0.15) if t%4==0 else (0.92,0.90),
    'Pattern 6\n(Sleeper x30)':      lambda t: (0.08,0.15) if t>30 else (0.92,0.90),
}
METHOD_STYLE = {
    'Proposed':         {'color':'#1a6b3c','ls':'-', 'lw':2.0},
    'Subjective Logic': {'color':'#2980b9','ls':'--','lw':1.6},
    'Beta Reputation':  {'color':'#e67e22','ls':'-.','lw':1.5},
    'EMA':              {'color':'#95a5a6','ls':':','lw':1.4},
}
fig, axes = plt.subplots(1, len(patterns), figsize=(14, 3.8), sharey=True)
fig.subplots_adjust(wspace=0.12)
for ax, (pat_name, pat_fn) in zip(axes, patterns.items()):
    ev2 = NodeTrustEvaluator(theta=THETA, lam=LAMBDA, phi=PHI, xi=XI)
    records=[]; prop_v=[]
    for t in range(1, ITERS+1):
        a,d = pat_fn(t)
        records.append(BehaviorRecord(node_id=0, abb=a, timeliness=d, timestamp=float(t)))
        r2 = ev2.evaluate(records,[],[10]*5,[0]*5,float(t))
        prop_v.append(r2['global_trust'])
    sl=SubjectiveLogicTrust(rho=0.9); bt=BetaReputationTrust(decay=0.95); em=EMATrust(alpha=0.2)
    sl_v=[]; bt_v=[]; em_v=[]
    for t in range(1, ITERS+1):
        a,d=pat_fn(t); sl_v.append(sl.update(a,d)); bt_v.append(bt.update(a,d)); em_v.append(em.update(a,d))
    for name, vals in [('Proposed',prop_v),('Subjective Logic',sl_v),('Beta Reputation',bt_v),('EMA',em_v)]:
        s=METHOD_STYLE[name]; ax.plot(range(1,ITERS+1), vals, color=s['color'], ls=s['ls'], lw=s['lw'], label=name)
    ax.axhline(0.8, color='#1a6b3c', lw=0.7, ls=':', alpha=0.55)
    ax.axhline(0.4, color='#c0392b', lw=0.7, ls=':', alpha=0.55)
    if 'Sleeper' in pat_name: ax.axvline(30, color='#555', lw=0.8, ls='--', alpha=0.4)
    ax.set_title(pat_name, fontsize=8.5); ax.set_xlabel('Interaction step'); ax.set_ylim(-0.02,1.05)
axes[0].set_ylabel('Trust value')
handles,labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5,-0.02), fontsize=8.5)
fig.suptitle('Single-edge trust evolution: proposed vs baselines', y=1.01)
fig.tight_layout()
save(fig, 'fig_v2_single_edge')

# ── Robustness figure ────────────────────────────────────────────────────────
FAIL_AT=40; FAIL_LEN=3
def pat_fail(t): return (0.08,0.15) if FAIL_AT<=t<FAIL_AT+FAIL_LEN else (0.92,0.90)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ev3=NodeTrustEvaluator(theta=THETA,lam=LAMBDA,phi=PHI,xi=XI)
records=[]; prop_v=[]
for t in range(1,101):
    a,d=pat_fail(t)
    records.append(BehaviorRecord(node_id=0,abb=a,timeliness=d,timestamp=float(t)))
    r3=ev3.evaluate(records,[],[10]*5,[0]*5,float(t)); prop_v.append(r3['global_trust'])
sl=SubjectiveLogicTrust(rho=0.9); bt=BetaReputationTrust(decay=0.95); em=EMATrust(alpha=0.2)
sl_v=[]; bt_v=[]; em_v=[]
for t in range(1,101):
    a,d=pat_fail(t); sl_v.append(sl.update(a,d)); bt_v.append(bt.update(a,d)); em_v.append(em.update(a,d))
for name,vals in [('Proposed',prop_v),('Subjective Logic',sl_v),('Beta Reputation',bt_v),('EMA',em_v)]:
    s=METHOD_STYLE[name]; ax.plot(range(1,101),vals,color=s['color'],ls=s['ls'],lw=s['lw'],label=name)
ax.axvspan(FAIL_AT,FAIL_AT+FAIL_LEN-1,alpha=0.10,color='#c0392b',zorder=0)
ax.axhline(0.8,color='#1a6b3c',lw=0.8,ls=':',alpha=0.55); ax.axhline(0.4,color='#c0392b',lw=0.8,ls=':',alpha=0.55)
ax.set_xlabel('Interaction step'); ax.set_ylabel('Trust value'); ax.legend(loc='lower right',fontsize=8.5)
ax.set_title('Robustness to transient failure (3 steps at step 40)'); fig.tight_layout()
save(fig, 'fig_v2_robustness')

print('\nAll figures saved to', OUTPUT_DIR)
