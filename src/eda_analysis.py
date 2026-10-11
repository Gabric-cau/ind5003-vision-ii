import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(features_csv: str = "data/processed/penalties_features.csv", output_dir: str = "outputs"):
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Run pipeline first: {features_csv} not found.")
    df = pd.read_csv(features_csv)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Top-Down Trajectories Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([-3.66, 3.66], [0, 0], color='red', lw=4, label='Goal (7.32m)')
    ax.scatter([0], [11.0], color='black', s=100, zorder=5, label='Penalty Spot')
    for _, row in df.iterrows():
        color = '#1f77b4' if row["kicker_footedness"] == "right" else '#ff7f0e'
        ax.annotate('', xy=(0, 11.0), xytext=(row["kicker_start_x_m"], row["kicker_start_y_m"]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7))
    ax.set_xlim(-12, 12); ax.set_ylim(-1, 20)
    ax.set_title("Reconstructed Top-Down Kicker Run-Up Trajectories", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "kicker_runup_topdown_trajectories.png"), dpi=300)
    plt.close()

    # 2. Game Theory Matrix Plot
    matrix = pd.crosstab(df["gk_dive_direction"], df["shot_direction"], values=df["outcome"] == "goal", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100, ax=ax)
    ax.set_title("Kicker vs Goalkeeper Strategy Matrix (Goal %)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "game_theory_gk_vs_kicker.png"), dpi=300)
    plt.close()

    print(f"[+] Generated EDA plots in: {plots_dir}")

if __name__ == "__main__":
    run_eda()