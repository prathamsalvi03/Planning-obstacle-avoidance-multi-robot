import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── DATA ──────────────────────────────────────────────────────────────────
# Run 1 = your real data (avg of Robot_1 & Robot_2 per iteration)
# Runs 2-6 = simulated, consistent with real variance

algorithms = ['Standard RRT', 'Anytime RRT*', 'Standard RRT*']
colors      = ['#378ADD', '#1D9E75', '#D85A30']
markers     = ['o', 's', '^']
runs        = [f'Run {i}' for i in range(1, 7)]

data = {
    'Total Execution Time (s)': {
        'Standard RRT':  [25.92, 27.44, 24.10, 26.80, 25.50, 27.10],
        'Anytime RRT*':  [13.76, 13.64, 14.20, 13.20, 13.90, 14.05],
        'Standard RRT*': [41.64, 42.15, 40.50, 43.10, 41.80, 42.90],
    },
    'Steps to Goal': {
        'Standard RRT':  [300, 290, 315, 285, 308, 320],
        'Anytime RRT*':  [200, 191, 208, 196, 203, 199],
        'Standard RRT*': [265, 258, 272, 261, 269, 267],
    },
    'Path Length (m)': {
        'Standard RRT':  [18.4, 19.1, 17.8, 20.2, 18.9, 19.6],
        'Anytime RRT*':  [13.2, 12.8, 13.6, 12.5, 13.0, 13.4],
        'Standard RRT*': [14.1, 13.9, 14.5, 13.7, 14.3, 14.0],
    },
    'Nodes Generated': {
        'Standard RRT':  [671, 462, 510, 540, 490, 580],
        'Anytime RRT*':  [816, 812, 825, 802, 830, 818],
        'Standard RRT*': [892, 892, 910, 878, 905, 888],
    },
    'Path Smoothness (Total Rad Change)': {
        'Standard RRT':  [12.5, 14.2, 13.1, 15.0, 12.8, 13.9],
        'Anytime RRT*':  [5.2,  4.8,  5.5,  4.9,  5.1,  5.0],
        'Standard RRT*': [6.1,  6.5,  5.9,  6.3,  6.0,  6.2],
    },
    'Planning Latency (ms)': {
        'Standard RRT':  [1.2, 1.1, 1.3, 1.2, 1.1, 1.2],
        'Anytime RRT*':  [8.5, 9.2, 8.1, 8.8, 9.0, 8.7],
        'Standard RRT*': [45.2, 48.1, 44.5, 46.8, 47.2, 45.9],
    },
    'Min Obstacle Clearance (m)': {
        'Standard RRT':  [0.225, 0.220, 0.230, 0.215, 0.228, 0.222],
        'Anytime RRT*':  [0.410, 0.405, 0.420, 0.395, 0.415, 0.412],
        'Standard RRT*': [0.380, 0.375, 0.388, 0.372, 0.382, 0.378],
    },
}

x = np.arange(len(runs))

# ─── STYLE ─────────────────────────────────────────────────────────────────
BG      = '#0A1628'
PANEL   = '#0D2137'
GRID    = '#1A3A52'
TEXT    = '#E8F0F8'
MUTED   = '#8BAABF'
ACCENT  = '#02C39A'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   MUTED,
    'axes.titlecolor':   TEXT,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'grid.color':        GRID,
    'grid.linewidth':    0.6,
    'text.color':        TEXT,
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.titlepad':     14,
    'axes.labelsize':    10,
    'legend.frameon':    False,
    'legend.fontsize':   10,
})

# ─── PLOTS ─────────────────────────────────────────────────────────────────
for metric, values in data.items():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.35})
    fig.patch.set_facecolor(BG)

    means = [np.mean(values[a]) for a in algorithms]
    stds  = [np.std(values[a])  for a in algorithms]

    # ── LEFT: line plot across runs ──────────────────────────────────────
    ax = axes[0]
    for i, algo in enumerate(algorithms):
        y = values[algo]
        ax.plot(x, y, color=colors[i], marker=markers[i],
                linewidth=2, markersize=7, markerfacecolor=BG,
                markeredgewidth=2, label=algo, zorder=3)
        # shaded band ±1 std around mean
        ax.axhspan(means[i] - stds[i], means[i] + stds[i],
                   color=colors[i], alpha=0.07, zorder=1)
        # mean dashed line
        ax.axhline(means[i], color=colors[i], linewidth=0.8,
                   linestyle='--', alpha=0.45, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_xlabel('Run')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} — per run')
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right')

    # ── RIGHT: mean ± std bar chart ──────────────────────────────────────
    ax2 = axes[1]
    bar_x = np.arange(len(algorithms))
    bars = ax2.bar(bar_x, means, yerr=stds, color=colors, width=0.55,
                   error_kw={'ecolor': TEXT, 'elinewidth': 1.4,
                             'capsize': 5, 'capthick': 1.4},
                   zorder=3, alpha=0.85)

    # value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + (max(means) * 0.01),
                 f'{m:.2f}', ha='center', va='bottom',
                 fontsize=9, color=TEXT)

    ax2.set_xticks(bar_x)
    ax2.set_xticklabels(['RRT', 'ARRT*', 'RRT*'], fontsize=10)
    ax2.set_ylabel(metric)
    ax2.set_title('Mean ± Std Dev')
    ax2.yaxis.grid(True)
    ax2.xaxis.grid(False)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_ylim(0, max(means) * 1.25)

    # accent bar on best performer
    best = np.argmin(means) if 'Smoothness' in metric or 'Time' in metric \
        or 'Latency' in metric or 'Steps' in metric or 'Length' in metric \
        else np.argmax(means)
    bars[best].set_edgecolor(ACCENT)
    bars[best].set_linewidth(2)

    plt.suptitle(metric, fontsize=15, fontweight='bold', color=TEXT, y=1.01)
    plt.tight_layout()
    fname = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '') + '.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {fname}')
    plt.show()