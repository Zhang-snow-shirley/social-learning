import sys
import os

# ─── PATH SETUP ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from simulation_visualization import Simulation, SimulationConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch

# ─── EXPERIMENTAL PARAMETERS ───────────────────────────────────────────────────────

network_sizes = [25, 50, 75, 100]      # x‐axis
matrix_rows_list = [5, 10, 15, 20]     # y‐axis

agent_ratios = {"M0": 0.2, "M1": 0.3, "M3": 0.5}
network_type = "small_world"
connectivity = 0.2
matrix_cols = 4
total_runs = 1000
noise_update_frequency = 10
noise_sigma = 1
random_seed = 42

# ─── RUNNING ALL COMBINATIONS ─────────────────────────────────────────────────────

records = []

for size in network_sizes:
    for rows in matrix_rows_list:
        config = SimulationConfig(
            network_size=size,
            network_type=network_type,
            connectivity=connectivity,
            agent_ratios=agent_ratios,
            matrix_rows=rows,
            matrix_cols=matrix_cols,
            total_runs=total_runs,
            noise_update_frequency=noise_update_frequency,
            noise_sigma=noise_sigma,
            random_seed=random_seed
        )

        sim = Simulation(config)
        print(f"Running: network_size={size}, matrix_rows={rows}")
        sim.run()

        results_df = sim.get_results()
        mean_errors = (
            results_df
            .groupby("agent_type")["abs_error"]
            .mean()
            .reset_index()
            .rename(columns={"abs_error": "mean_abs_error"})
        )

        for _, row in mean_errors.iterrows():
            records.append({
                "network_size": size,
                "matrix_rows": rows,
                "agent_type": row["agent_type"],
                "mean_abs_error": row["mean_abs_error"]
            })

df = pd.DataFrame(records)

# Build the 2D grids for X (= network_size), Y (= matrix_rows)
X_vals = np.array(network_sizes)
Y_vals = np.array(matrix_rows_list)
X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)

agent_types = df["agent_type"].unique()
Z_grids = {}
for agent in agent_types:
    Z = np.zeros_like(X_grid, dtype=float)
    subset = df[df["agent_type"] == agent]
    for i, rows in enumerate(matrix_rows_list):
        for j, size in enumerate(network_sizes):
            sel = subset[
                (subset["network_size"] == size) &
                (subset["matrix_rows"] == rows)
            ]
            if not sel.empty:
                Z[i, j] = sel["mean_abs_error"].values[0]
            else:
                Z[i, j] = np.nan
    Z_grids[agent] = Z

# ─── PLOTTING THE 3D SURFACES ───────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# 1) Define a fixed color for each agent type:
color_map = {
    "M0": "tab:blue",
    "M1": "tab:orange",
    "M3": "tab:green"
}

# 2) Plot each agent’s surface with its fixed color, plus a bit of transparency:
for agent in agent_types:
    Z = Z_grids[agent]
    ax.plot_surface(
        X_grid,            # network_size axis (x)
        Y_grid,            # matrix_rows axis (y)
        Z,                 # mean_abs_error (z)
        color=color_map[agent],
        alpha=0.7,
        linewidth=0,
        antialiased=True
    )

# 3) Build proxy legend handles (one Patch per agent) with matching colors:
legend_handles = [
    Patch(facecolor=color_map[agent], edgecolor='k', label=f"Agent {agent}")
    for agent in agent_types
]

ax.set_xlabel("Network Size")
ax.set_ylabel("Matrix Rows")
ax.set_zlabel("Mean Absolute Error")
ax.set_title("3D Surface: Mean Error vs. (network_size, matrix_rows)")

# 4) Show the legend using our proxy patches:
ax.legend(handles=legend_handles, loc="upper left")

plt.tight_layout()
plt.show()
