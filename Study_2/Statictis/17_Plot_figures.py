"""
plot_figures_final.py

Professional academic figures for the paper (publication-ready):
    Figure 1: Markov Transition Probabilities – Three Pathways (Panels A, B, C)
    Figure 2: Price Decomposition (Stacked Bar)
    Figure 3: C2 Bifurcation Trajectories (t from -4 to +3)
    Figure 4: A/B Decomposition

All figures are saved as high-resolution PNG and PDF.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ENHANCED PROFESSIONAL STYLE
# ============================================================================
OUTPUT_DIR = Path('results/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10.5,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'gray',
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'figure.figsize': (9.5, 6),
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.25,
    'grid.color': 'gray',
    'grid.alpha': 0.22,
    'grid.linestyle': '--',
    'mathtext.fontset': 'stix',
    'mathtext.default': 'regular',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})

# ----------------------------------------------------------------------------
# DATA LOADING (only transitions >5%)
# ----------------------------------------------------------------------------

def load_markov_data():
    """
    Load transition probabilities (Global scope) from Table C3.
    Only transitions with probability >5% are kept.
    """
    path = Path('data/results/markov_transitions.csv')
    if path.exists():
        df = pd.read_csv(path)
        if 'Scope' in df.columns:
            df = df[df['Scope'] == 'Global']
        df = df[df['Transition_Prob_Pct'] > 5]
        return df[['Current_State', 'Next_State', 'Transition_Prob_Pct']].copy()
    else:
        # Hardcoded from Study 2, Table C3 (only >5%)
        data = [
            ('Normal', 'C2',  6.5), ('Normal', 'C3', 12.0),
            ('C2',  'C2', 40.2), ('C2',  'C1', 22.4), ('C2',  'C6', 15.3),
            ('C2',  'C3', 10.7), ('C2',  'C4', 10.7),
            ('C3',  'C3', 55.7), ('C3',  'C4', 34.1),
            ('C4',  'C3', 59.2), ('C4',  'C4', 27.7),
            ('C1',  'C2', 45.7), ('C1',  'C1', 19.1),
            ('C1',  'C3', 12.8), ('C1',  'C4',  9.0), ('C1',  'C6',  9.0),
            ('C6',  'C2', 35.9), ('C6',  'C6', 17.9),
            ('C6',  'C3', 12.0), ('C6',  'C4', 13.7),
        ]
        df = pd.DataFrame(data, columns=['Current_State', 'Next_State', 'Transition_Prob_Pct'])
        return df

def load_decomposition_data():
    data = {
        'Configuration': ['Normal', 'C2', 'C3', 'C4'],
        'V_Prod_base': [72, 10, 12, 13],
        's_baseline': [8, 4, 5, 6],
        'S_Surplus': [4, 2, 4, 4],
        'K_Brand': [8, 8, 9, 10],
        "K_Pi'": [8, 76, 70, 67]
    }
    return pd.DataFrame(data).set_index('Configuration')

def load_bifurcation_data():
    """Construct trajectories for C2 bifurcation, t from -4 to +3."""
    t = np.linspace(-4, 3, 200)   # giới hạn đến +3
    # Pre-bifurcation (t ≤ 0): common for Sustain and Collapse
    pre = 32.0 + 2.1 * t + 0.65 * t**2
    # Sustain post
    sustain_post = 32.0 + 2.1 * t + 3.15 * t**2
    # Collapse post
    collapse_post = 32.0 + 2.1 * t + 7.8 * t**2
    B_sustain = np.where(t <= 0, pre, sustain_post)
    B_collapse = np.where(t <= 0, pre, collapse_post)
    # Evolve: braking then plateau
    B_evolve = np.where(t <= 0,
                        5.5 + 1.8 * t - 0.25 * t**2,
                        12.0 * np.ones_like(t))
    return pd.DataFrame({'t': t, 'Sustain': B_sustain, 'Evolve': B_evolve, 'Collapse': B_collapse})

def load_ab_data():
    data = {
        'Configuration': ['Normal', 'C2', 'C3', 'C4', 'C1', 'C6'],
        'A': [1.04, 1.17, 1.40, 1.43, 1.55, 1.96],
        'B': [-1.47, 28.86, 15.56, 12.58, 18.46, 20.61]
    }
    return pd.DataFrame(data).set_index('Configuration')

# ============================================================================
# FIGURE 1: THREE PANELS (only transitions >5%, white bbox for labels)
# ============================================================================

def draw_panel(ax, pos, nodes, edges, node_colors, title):
    """
    Draw a network panel with white‑background edge labels.
    """
    G = nx.DiGraph()
    for node in pos:
        G.add_node(node)
    edge_probs = {}
    for u, v, p in edges:
        G.add_edge(u, v, weight=p)
        edge_probs[(u, v)] = p

    node_color_list = [node_colors.get(n, '#7f7f7f') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list,
                           node_size=2000, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')

    # Self-loops and other edges
    self_loops = [(u, v) for u, v in G.edges() if u == v]
    other_edges = [(u, v) for u, v in G.edges() if u != v]

    if self_loops:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=self_loops,
                               connectionstyle='arc3,rad=0.7', arrowsize=18,
                               width=[1.5 + G[u][v]['weight']/15 for (u,v) in self_loops],
                               edge_color='gray', arrowstyle='->')
    if other_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=other_edges,
                               connectionstyle='arc3,rad=0.25', arrowsize=18,
                               width=[1.5 + G[u][v]['weight']/15 for (u,v) in other_edges],
                               edge_color='gray', arrowstyle='->')

    # Edge labels with white background (bbox)
    edge_labels = {(u, v): f"{edge_probs[(u, v)]:.1f}%" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                 font_size=9, label_pos=0.65,
                                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

def figure1_markov(markov_df):
    node_colors = {
        'Normal': '#2ca02c', 'C2': '#f5c400',
        'C3': '#ff7f0e', 'C4': '#ff7f0e',
        'C1': '#d62728', 'C6': '#d62728'
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # --- Panel A: Direct Termination from Gestation ---
    pos_A = {'Normal': (0.15, 0.5), 'C2': (0.45, 0.5), 'C1': (0.85, 0.7), 'C6': (0.85, 0.3)}
    edges_A = []
    for _, row in markov_df.iterrows():
        u, v, p = row['Current_State'], row['Next_State'], row['Transition_Prob_Pct']
        if set([u, v]).issubset({'Normal', 'C2', 'C1', 'C6'}):
            edges_A.append((u, v, p))
    draw_panel(axes[0], pos_A, ['Normal','C2','C1','C6'], edges_A, node_colors,
               "A. Direct Termination from Gestation")

    # --- Panel B: Entry into Maturity ---
    pos_B = {'Normal': (0.15, 0.5), 'C2': (0.45, 0.5), 'C3': (0.85, 0.7), 'C4': (0.85, 0.3)}
    edges_B = []
    for _, row in markov_df.iterrows():
        u, v, p = row['Current_State'], row['Next_State'], row['Transition_Prob_Pct']
        if set([u, v]).issubset({'Normal', 'C2', 'C3', 'C4'}):
            edges_B.append((u, v, p))
    draw_panel(axes[1], pos_B, ['Normal','C2','C3','C4'], edges_B, node_colors,
               "B. Entry into Maturity")

    # --- Panel C: Termination from Maturity (no <5% transitions) ---
    pos_C = {'C3': (0.15, 0.7), 'C4': (0.15, 0.3), 'C1': (0.85, 0.7), 'C6': (0.85, 0.3)}
    edges_C = []
    for _, row in markov_df.iterrows():
        u, v, p = row['Current_State'], row['Next_State'], row['Transition_Prob_Pct']
        if set([u, v]).issubset({'C3', 'C4', 'C1', 'C6'}):
            edges_C.append((u, v, p))
    draw_panel(axes[2], pos_C, ['C3','C4','C1','C6'], edges_C, node_colors,
               "C. Termination from Maturity")

    # Add annotations for re‑entry arrows (dashed) and note
    axes[2].annotate('→ C2', xy=(0.98, 0.78), xytext=(0.92, 0.78),
                     arrowprops=dict(arrowstyle='->', color='gray', linestyle='dashed', lw=1.5),
                     fontsize=9, ha='right')
    axes[2].annotate('→ C2', xy=(0.98, 0.22), xytext=(0.92, 0.22),
                     arrowprops=dict(arrowstyle='->', color='gray', linestyle='dashed', lw=1.5),
                     fontsize=9, ha='right')
    axes[2].text(0.5, 0.02, "Transitions from C3/C4 to C1/C6 are below 5% and omitted",
                 transform=axes[2].transAxes, fontsize=9, style='italic', ha='center', color='gray')

    fig.suptitle("Markov Transition Probabilities (Study 2, n=2,573)", fontsize=18, fontweight='bold', y=1.02)
    fig.text(0.5, 0.01,
             "Arrow width proportional to transition probability. Self-loops indicate quarter-to-quarter persistence.",
             ha='center', fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'Figure1_Markov_ThreePanels.pdf')
    plt.savefig(OUTPUT_DIR / 'Figure1_Markov_ThreePanels.png')
    plt.close()

# ============================================================================
# FIGURE 2: PRICE DECOMPOSITION (unchanged)
# ============================================================================

def figure2_decomposition(decomp_df):
    configs = decomp_df.index.tolist()
    components = ['V_Prod_base', 's_baseline', 'S_Surplus', 'K_Brand', "K_Pi'"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(10.5, 6))
    left = np.zeros(len(configs))
    for i, (comp, color) in enumerate(zip(components, colors)):
        values = decomp_df[comp].values
        ax.barh(configs, values, left=left, color=color, label=comp,
                edgecolor='white', linewidth=1.3)
        for j, v in enumerate(values):
            if v > 3.5:
                ax.text(left[j] + v/2, j, f'{v:.0f}%', ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold')
        left += values
    for j, cfg in enumerate(configs):
        ax.text(left[j] + 1.8, j, '100%', ha='left', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 107)
    ax.set_xlabel('Percentage of Market Capitalization', fontsize=13.5)
    ax.set_title('Price Decomposition by Configuration', fontsize=17, fontweight='bold', pad=25)
    ax.legend(loc='lower right', frameon=True, bbox_to_anchor=(1.22, 0.02))
    ax.grid(axis='x', linestyle='--', alpha=0.22)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure2_Decomposition.pdf')
    plt.savefig(OUTPUT_DIR / 'Figure2_Decomposition.png')
    plt.close()

# ============================================================================
# FIGURE 3: C2 BIFURCATION (t from -4 to +3, annotations on top)
# ============================================================================

def figure3_bifurcation(bifurcation_df):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.axvspan(-0.5, 0.5, alpha=0.11, color='gray', label='Bifurcation zone', zorder=0)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.6, alpha=0.85, zorder=0)

    ax.plot(bifurcation_df['t'], bifurcation_df['Sustain'], 
            label='Sustain C2', linewidth=3.2, color='#1f4e79', zorder=2)
    ax.plot(bifurcation_df['t'], bifurcation_df['Evolve'], 
            label='Evolve → C3/C4', linewidth=3.2, color='#ff7f0e', zorder=2)
    ax.plot(bifurcation_df['t'], bifurcation_df['Collapse'], 
            label='Collapse → C1/C6', linewidth=3.2, color='#d62728', zorder=2)

    # Annotations
    ax.annotate('Braking (deceleration)', xy=(-1.2, 15), xytext=(-3.0, 23),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2.0, alpha=0.9),
                fontsize=11, ha='right', zorder=3)
    ax.text(0.05, 41, 'Indistinguishable\nat bifurcation',
            fontsize=9.5, color='gray', style='italic',
            ha='center', va='center', zorder=3,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))
    ax.annotate('Acceleration spike\n(14.06 vs 5.55)',
                xy=(1.8, 78), xytext=(1.0, 85),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2.0, alpha=0.9),
                fontsize=11, ha='left', zorder=3)

    # Branch labels at t=3
    ax.text(3.1, 12.5, '→ C3/C4', fontsize=10.5, color='#ff7f0e', fontweight='bold', ha='left')
    ax.text(3.1, 62, '→ C2', fontsize=10.5, color='#1f4e79', fontweight='bold', ha='left')
    ax.text(3.1, 195, '→ C1/C6', fontsize=10.5, color='#d62728', fontweight='bold', ha='left')

    ax.set_xlabel('Quarters relative to bifurcation', fontsize=13)
    ax.set_ylabel(r'B (E$_3$ - (1 + PGR$_t$))', fontsize=13)
    ax.set_title('C2 Bifurcation: Kinematic Trajectories', fontsize=16, fontweight='bold', pad=20)
    fig.text(0.5, 0.935,
             'Collapse and Sustain are indistinguishable at bifurcation; '
             'only post-bifurcation acceleration differs',
             ha='center', fontsize=9.8, style='italic', color='gray')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.set_xlim(-4.2, 3.3)
    ax.set_ylim(0, 350)   # adjusted to match the new range (t=3 gives ~45+5.18*3+7.8*9=45+15.54+70.2=130, not 800)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(OUTPUT_DIR / 'Figure3_Bifurcation.pdf')
    plt.savefig(OUTPUT_DIR / 'Figure3_Bifurcation.png')
    plt.close()

# ============================================================================
# FIGURE 4: A/B DECOMPOSITION (unchanged)
# ============================================================================

def figure4_ab(ab_df):
    configs = ab_df.index.tolist()
    a_vals = ab_df['A'].values
    b_vals = ab_df['B'].values
    x = np.arange(len(configs))
    width = 0.42
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars1 = ax.bar(x - width/2, a_vals, width, label='A (productive growth)',
                   color='#1f77b4', edgecolor='black', linewidth=1.1)
    bars2 = ax.bar(x + width/2, b_vals, width, label='B (surplus extraction)',
                   color='#d62728', edgecolor='black', linewidth=1.1)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f'{h:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        offset = 0.6 if h > 0 else -1.1
        va = 'bottom' if h > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, h + offset, f'{h:.2f}',
                ha='center', va=va, fontsize=9, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=13)
    ax.set_ylabel('Value', fontsize=13)
    ax.set_title('A/B Decomposition of E$_3$', fontsize=17, fontweight='bold', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.axhline(0, color='black', linewidth=1.1)
    ax.legend(loc='upper right', frameon=True)
    ax.annotate('A ≈ 1 for all speculative states',
                xy=(3.05, 1.52), xytext=(4.1, 3.8),
                arrowprops=dict(arrowstyle='->', color='#1f4e79', lw=1.9),
                fontsize=11, color='#1f4e79', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure4_AB_Decomposition.pdf')
    plt.savefig(OUTPUT_DIR / 'Figure4_AB_Decomposition.png')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Generating final academic figures (publication-ready)...")
    markov_df = load_markov_data()
    decomp_df = load_decomposition_data()
    bifurcation_df = load_bifurcation_data()
    ab_df = load_ab_data()

    figure1_markov(markov_df)
    print("  Figure 1 (Three panels) saved.")
    figure2_decomposition(decomp_df)
    print("  Figure 2 (Decomposition) saved.")
    figure3_bifurcation(bifurcation_df)
    print("  Figure 3 (Bifurcation) saved.")
    figure4_ab(ab_df)
    print("  Figure 4 (A/B) saved.")
    print(f"\nAll figures saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()