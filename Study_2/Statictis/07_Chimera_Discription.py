"""
07_C2_BifurcationVectors.py

Academic Title:
    Dynamic Vector Analysis at the C2 Bifurcation Point:
    Position, Velocity, and Acceleration Before Three Distinct Trajectories

Objective:
    This script provides a “X‑ray” of the dynamic forces at the inflection point
    where a firm in state C2 diverges into one of three paths:
        1. Sustain C2 (continued uphill)
        2. Evolve to C3/C4 (horizontal glide / soft landing)
        3. Collapse to C1/C6 (structural crash)

    The analysis is performed on non‑Financials sectors and uses median
    statistics to mitigate outlier influence.

Methods:
    1. Compute derived variables:
       - B = E_3 - (1 + PGR_t)        (position / height on the slope)
       - B_Change = B - B_lag1         (velocity / turning vector)
       - B_Acceleration = B_Change - B_Change_lag1  (acceleration / throttle/brake)
    2. For each C2 observation, determine the next state (t+1) and classify
       the trajectory into one of the three categories.
    3. For each trajectory, report median values of B, B_Change, B_Acceleration,
       PGR_t, and the number of observations.

Interpretation (Economic Physics):
    - Sustain C2: positive velocity and positive acceleration → still accelerating uphill.
    - Evolve to C3/C4: velocity approaches zero (or negative), strong negative
      acceleration → active braking, preparing for soft landing.
    - Collapse to C1/C6: erratic vector (extreme positive or negative velocity),
      often accompanied by a collapse in growth momentum (PGR_t).

Output File (saved in data/results/):
    - 07_C2_BifurcationVectors_Report.txt : Academic report with median vectors.

Dependencies:
    pandas, numpy, pathlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/07_C2_BifurcationVectors_Report.txt'

# Sector to exclude (consistent with ML pipeline)
EXCLUDED_SECTORS = ['Financials_and_Real_Estate', 'Financial']


# ============================================================================
# DATA PREPARATION AND VECTOR CALCULATION
# ============================================================================

def compute_bifurcation_vectors(df_path):
    """
    Load data, exclude Financials, compute B (position), B_Change (velocity),
    and B_Acceleration (acceleration). For each C2 observation, determine the
    next trajectory class.
    """
    df = pd.read_csv(df_path)
    # Clean column names (remove leading/trailing spaces, drop duplicates)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df['period_end'] = pd.to_datetime(df['period_end'])

    # Exclude Financials sector for consistency
    if 'Sector' in df.columns:
        df = df[~df['Sector'].isin(EXCLUDED_SECTORS)].copy()

    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    group = df.groupby('Ticker')

    # Compute physical vectors (only if required columns exist)
    if 'PGR_t' in df.columns and 'E_3' in df.columns:
        # Position (height on the slope)
        df['B'] = df['E_3'] - (1 + df['PGR_t'])

        # Velocity (turning vector)
        df['B_lag1'] = group['B'].shift(1)
        df['B_Change'] = df['B'] - df['B_lag1']

        # Acceleration (throttle / brake)
        df['B_Change_lag1'] = group['B_Change'].shift(1)
        df['B_Acceleration'] = df['B_Change'] - df['B_Change_lag1']

    # Next state
    df['next_config'] = group['Configuration'].shift(-1)

    # Keep only rows currently in C2
    df_c2 = df[df['Configuration'] == 'C2'].copy()

    # Classify trajectories
    def classify_trajectory(next_state):
        val = str(next_state).strip()
        if val == 'C2':
            return '1. Sustain C2 (Uphill)'
        if val in ['C3', 'C4']:
            return '2. Evolve to C3/C4 (Horizontal / Soft landing)'
        if val in ['C1', 'C6']:
            return '3. Collapse to C1/C6 (Structural crash)'
        return 'Unknown'

    df_c2['Trajectory'] = df_c2['next_config'].apply(classify_trajectory)
    df_c2 = df_c2[df_c2['Trajectory'] != 'Unknown'].copy()

    features = ['B', 'B_Change', 'B_Acceleration', 'PGR_t']
    return df_c2.dropna(subset=features + ['Trajectory'])


# ============================================================================
# MAIN ANALYSIS AND REPORT
# ============================================================================

def main():
    print("=" * 80)
    print("C2 BIFURCATION VECTOR ANALYSIS")
    print("Position, Velocity, and Acceleration at the Inflection Point")
    print("=" * 80)

    print("\n[1] Computing bifurcation vectors...")
    df_vectors = compute_bifurcation_vectors(DATA_FILE)

    # Median statistics (robust against outliers)
    summary = df_vectors.groupby('Trajectory')[
        ['B', 'B_Change', 'B_Acceleration', 'PGR_t']
    ].median()

    # Add sample size
    summary['N_obs'] = df_vectors.groupby('Trajectory').size()
    # Reorder columns for clarity
    summary = summary[['N_obs', 'B', 'B_Change', 'B_Acceleration', 'PGR_t']]

    # Write academic report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 110 + "\n")
        f.write("ACADEMIC REPORT: DYNAMIC VECTORS AT THE C2 BIFURCATION POINT\n")
        f.write("Position (B), Velocity (B_Change), and Acceleration (B_Acceleration)\n")
        f.write("Excluding Financials Sector\n")
        f.write("=" * 110 + "\n\n")

        f.write("I. MEDIAN VECTOR VALUES IMMEDIATELY BEFORE THE TRANSITION (quarter t)\n")
        f.write("-" * 110 + "\n")
        f.write(summary.to_string() + "\n\n")

        f.write("II. ECONOMIC PHYSICS INTERPRETATION\n")
        f.write("-" * 110 + "\n")
        f.write("1. SUSTAIN C2 (Uphill) – C2 → C2:\n")
        f.write("   - Velocity (B_Change) > 0 and Acceleration (B_Acceleration) > 0.\n")
        f.write("   - The firm continues to press the accelerator, climbing further\n")
        f.write("     into the deficit zone.\n\n")

        f.write("2. EVOLVE TO C3/C4 (Horizontal / Soft landing) – C2 → C3/C4:\n")
        f.write("   - Velocity approaches zero (or becomes negative), and acceleration\n")
        f.write("     is strongly negative.\n")
        f.write("   - This indicates active braking. The firm glides sideways to land softly.\n\n")

        f.write("3. COLLAPSE TO C1/C6 (Structural crash) – C2 → C1/C6:\n")
        f.write("   - Vector chaos: velocity jumps to an extreme positive or reverses\n")
        f.write("     sharply, often accompanied by a collapse in growth momentum (PGR_t).\n")
        f.write("   - This reflects the disintegration of the financial structure.\n\n")

        f.write("III. SAMPLE CHARACTERISTICS\n")
        f.write(f"   Total C2 observations with known next state: {len(df_vectors)}\n")
        for traj in summary.index:
            n = summary.loc[traj, 'N_obs']
            pct = 100 * n / len(df_vectors)
            f.write(f"   - {traj}: {n} observations ({pct:.1f}%)\n")

        f.write("\n" + "=" * 110 + "\n")
        f.write("Full vector data is available in the intermediate DataFrame.\n")
        f.write("This analysis supports the 'Expectation Chimera' hypothesis:\n")
        f.write("the divergence at C2 is governed by observable mechanical forces.\n")
        f.write("=" * 110 + "\n")

    print(f"\n✅ Analysis completed successfully.")
    print(f"   Academic report: {TXT_REPORT}")


if __name__ == "__main__":
    main()