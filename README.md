# Agent-Based Social Inference & Adaptive Learning Framework

This repository presents a flexible experimental platform combining large-scale agent-based simulations and reinforcement learning to study how individuals infer missing social information and adapt strategies over time. Designed with social psychology principles in mind, it integrates network topology variation, matrix-based data generation, and dynamic learning to explore:

1. **How network structure shapes inference accuracy.**
2. **How different belief‑update heuristics perform under noise.**
3. **How agents can adapt their inference strategies through reinforcement learning.**

## Table of Contents

* [Background & Objectives](#background--objectives)
* [Experimental Design](#experimental-design)

  * [Simulation Module](#simulation-module)
  * [Reinforcement Learning Module](#reinforcement-learning-module)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Results & Discussion](#results--discussion)
* [Requirements](#requirements)
* [Contributing](#contributing)
* [License](#license)

## Background & Objectives

Social inference—predicting unseen attributes or behaviors of peers based on observed interactions—is central to understanding collective dynamics. This project builds on social psychology experiments by simulating:

* **Network effects:** How topology (random, small‑world, scale‑free) alters information diffusion.
* **Inference strategies:** From simple averaging (mean‑field) to threshold or multi‑step heuristics.
* **Adaptive learning:** Allowing agents to discover optimal update rules via reinforcement learning, addressing overfitting and dynamics in social contexts.

## Experimental Design

### Simulation Module

All simulation experiments are orchestrated in `simulation_dask.py`, which performs a full parameter sweep across:

* **Network Topologies:** Erdős–Rényi random, Watts–Strogatz small‑world, Barabási–Albert scale‑free.
* **Connectivity Levels:** Adjustable probability or neighbor count per topology.
* **Data Dimensions & Noise:** Correlated matrices of size up to 1e6×4 with controllable covariance; Gaussian noise with tunable variance.
* **Agent Strategies:** Static types (`M0Agent`, `M1Agent`, `M3Agent`) representing threshold, mean‑field, and multi‑step inference.

**Workflow:**

1. **Initialization**

   * **Social Environment**
     We initialized $N 	imes C$ matrices to store the behavioral tendencies of each Object in each Context. Contexts C1 and C2 are correlated, as are C3 and C4. To mimic dynamic changes in the social environment, Gaussian noise is injected into the environment every 10 rounds.
   * **Network**
     Build the interaction topology (random, small-world, or scale-free) using `NetworkBuilder`, assigning each agent its list of neighbors.
   * **Agent**

     * **M0 (Model-Free):** Performs purely random inference for each missing cell. When infer_cell is called, the agent first checks its internal storage for a previously inferred value at (row, col); if found, it returns this stored value. Otherwise, it draws a fresh sample from a standard normal distribution N(0,1). There is no neighbor communication or use of the covariance structure. 
     * **M1 (Gossiper):**  Utilizes neighbor information with a capped gossip protocol:

1. **Storage Check:** On `infer_cell`, returns a stored prediction if available for (row, col).  
2. **Neighbor Exploration:** Iterates through all neighbors, collecting up to `maximal_neighbors` (20) values:  
   - **M0/M3 Peers:** If the neighbor has a value for (row, col), calls `infer_cell` to retrieve it; if not, records zero.  
   - **M1 Peers:** Retrieves previously gossiped values via `get_stored_value` without invoking further inference.  
3. **Aggregation:** Computes the arithmetic mean of collected neighbor values.  
4. **Fallback:** If no neighbor values are gathered, samples a draw from N(0,1).  
5. **Memory & Broadcast:** Stores the new inference for future reuse and broadcasts it to all neighbors for their caches.  
     * **M3 (Model-Based):** M3Agents maintain a memory store of all past observed values for every matrix cell (row, column). To infer a missing value at (row, target\_column), an agent:
2. Selects the paired `reference_column` corresponding to the target (C1⇔C2, C3⇔C4).
3. Retrieves from memory all past pairs of observed values for the target and reference columns across previous rounds.
4. Ensures at least `pairs_needed` data points are available; if not, returns a predefined fallback estimate.
5. Executes the JIT-compiled `infer_element_dict` regression on the full historical sample to compute slope and intercept.
6. Predicts the target as `intercept + slope * reference_value` using the current reference observation.

This memory-driven approach leverages the complete history of observations, enabling robust linear inference that adapts as the environment evolves.

2. **Simulation Loop**

   * **Rounds:** The simulation proceeds for `T` discrete rounds (configured in `simulation_dask.py`). In each round:

     1. **Inference Phase:** Every agent with a missing value at (row, col) retrieves the current environment value for its reference contexts and applies its inference strategy (M0, M1, or M3) to predict the target cell.
     2. **Memory Update:** Agents store the newly observed true value and their own prediction in their internal memory (M3 agents append to their full-history store; M1/M0 record neighbor-shared values).
     3. **Communication Phase:** Agents broadcast their predictions to neighbors, updating neighbor caches for the next round’s gossip or threshold checks.
     4. **Error Logging:** Each agent computes absolute and squared errors for its prediction and appends these metrics to its `prediction_errors` log.
   * **Environmental Noise:** Every 10 rounds, Gaussian noise (σ configured in the script) is injected into the social environment matrices to simulate evolving behavioral tendencies.
   * **Data Aggregation:** After completion of all rounds across network, noise, and agent configurations, results (e.g., MAE, MSE distributions) are aggregated into JSON for downstream statistical analysis.

### Reinforcement Learning Module### Reinforcement Learning Module### Reinforcement Learning Module

The adaptive component in `reinforment_learning.py` defines a custom Gym environment where agents:

* **Observations:** Time‑series feature vectors and metadata per step.
* **Actions:** Select an inference strategy from a discrete set.
* **Rewards:** Composite of prediction performance (`performance_reward`), inference accuracy bonus, and temporal consistency bonus to discourage erratic switches.

**Key Innovations:**

* **Dynamic Strategy Adaptation:** Agents can switch heuristics mid‑simulation to optimize cumulative reward.
* **Robust Evaluation:** Early stopping on validation sets, corrected overfitting controls, and model checkpointing (`_final.zip`).
* **Analysis:** Logs training history, visualizations of learning curves, and reports dynamic strategy success rates.

## Installation

```bash
git clone https://github.com/yourusername/agent-inference-framework.git
cd agent-inference-framework
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run Full Simulation Sweep

```bash
python simulation_dask.py
```

Outputs consolidated JSON with performance metrics across all experimental conditions.

### Train Adaptive Agents

```bash
python reinforment_learning.py --config configs/rl_config.json
```

Generates trained models, learning-curve plots, and summary of dynamic adaptation efficacy.

## Project Structure

```plain
.
├── simulation_dask.py       # Parameter-sweep orchestrator with Dask
├── reinforment_learning.py  # Gym environment and RL training loop
├── util/
│   ├── social_environment.py  # Matrix generators & inference routines
│   ├── agents.py              # BaseAgent and specialized heuristics
│   └── network.py             # NetworkBuilder for graph topologies
├── output/                   # JSON results and model artifacts
├── configs/                  # Configuration files (e.g., rl_config.json)
├── requirements.txt
└── README.md
```

## Results & Discussion

Preliminary analyses demonstrate:

* **Topology Effects:** Small‑world networks yield faster convergence but higher variance under noise.
* **Strategy Comparison:** `M1Agent` (mean‑field) outperforms simple threshold (`M0Agent`) at low noise; multi‑step (`M3Agent`) excels when correlation is strong.
* **Adaptive Learning:** RL agents achieve up to 15% reduction in cumulative error by switching strategies dynamically.

Full result tables and figures are available in `output/` and via Jupyter notebooks in `notebooks/` for deeper statistical analysis.

## Requirements

* Python 3.7+
* `numpy`, `scipy`, `networkx`, `numba`, `dask`, `distributed`, `gymnasium`, `pandas`, `matplotlib`

Install with:

```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions that extend simulation scenarios, introduce new agent heuristics, or improve RL strategies. Please fork, branch, and submit pull requests with tests and clear documentation.

## License

Released under the MIT License. Feel free to use and adapt for research purposes.
