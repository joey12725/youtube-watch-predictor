"""Render benchmark results as a dot plot of test ROC AUC with 95% bootstrap intervals.

Optional helper: needs matplotlib, which the rest of the project does not.
    python3 scripts/plot_results.py reports/results.json docs/results.png
"""
import json
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SERIES = {'random': ('#2a78d6', 'Random split (stored)'), 'channel-holdout': ('#eb6834', 'Channel held out')}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main(source='reports/results.json', target='docs/results.png'):
    results = json.load(open(source))
    protocols = results['protocols']
    models = list(next(iter(protocols.values()))['models'])
    fig, ax = plt.subplots(figsize=(8.5, 0.48 * len(models) + 1.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    offsets = {'random': -0.14, 'channel-holdout': 0.14}
    for protocol, summary in protocols.items():
        color, label = SERIES[protocol]
        for i, model in enumerate(models):
            r = summary['models'][model]
            y = i + offsets[protocol]
            low, high = r['ci95'].get('roc_auc', [r['roc_auc']] * 2)
            ax.plot([low, high], [y, y], color=color, lw=2, solid_capstyle='round', alpha=0.55)
            ax.scatter([r['roc_auc']], [y], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=label if i == 0 else None)
            if protocol == 'random':
                ax.text(high + 0.008, y, f"{r['roc_auc']:.3f}", va='center', fontsize=8, color=MUTED)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.503, len(models) - 0.45, 'chance', fontsize=8, color=MUTED)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel('Test ROC AUC (dots) with 95% bootstrap interval', fontsize=9, color=MUTED)
    ax.tick_params(axis='x', labelsize=8, colors=MUTED)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    ax.set_title('Which feed videos get watched? Same models, two evaluation protocols',
                 fontsize=11, color=INK, loc='left', pad=12)
    fig.tight_layout()
    fig.savefig(target, facecolor=SURFACE)


if __name__ == '__main__':
    main(*sys.argv[1:])
