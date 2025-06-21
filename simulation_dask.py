import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from dataclasses import dataclass
from typing import Dict, List, Any
import json
import itertools
from dask import delayed, compute
from util.social_environment import simulate_correlated_matrix_dict, add_noise_to_matrix, batch_infer_with_dask, batch_create_noise_matrices_dask
from util.agents import BaseAgent, M0Agent, M1Agent, M3Agent
from util.network import NetworkBuilder

warnings.filterwarnings("ignore")

@dataclass
class SimulationConfig:
    network_size: int = 100
    network_type: str = "random" # Options: "small_world", "random", "scale_free", "complete"
    connectivity: float = 0.3
    agent_ratios: Dict[str, float] = None
    matrix_rows: int = 10
    matrix_cols: int = 4
    total_runs: int = 100
    noise_update_frequency: int = 100
    noise_sigma: float = 1
    random_seed: int = 42

    def __post_init__(self):
        if self.agent_ratios is None:
            self.agent_ratios = {"M0": 0.1, "M1": 0.3, "M3": 0.6}
        total = sum(self.agent_ratios.values())
        if abs(total - 1.0) > 1e-6:
            for k in self.agent_ratios:
                self.agent_ratios[k] /= total

class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.agents = []
        self.network = None
        self.current_run = 0
        self.baseline_matrix = None
        self.noisy_matrix = None
        self.noise_matrices = []  # Pre-created noise matrices
        self.current_noise_matrix_index = 0
        
        # Storage for snapshots
        self.agent_matrix_snapshots = []  # List of lists: [run][agent_id] -> matrix_dict.data
        self.noisy_matrix_snapshots = []  # List: [run] -> noisy_matrix.data

        self._initialize()

    def _initialize(self):
        np.random.seed(self.config.random_seed)
        # 1) simulate true baseline
        self.baseline_matrix = simulate_correlated_matrix_dict(
            A=self.config.matrix_rows,
            C=self.config.matrix_cols,
            random_seed=self.config.random_seed
        )
        
        # 2) Pre-create all noise matrices using Dask
        num_noise_matrices = self.config.total_runs // self.config.noise_update_frequency + 1
        sigma_list = [self.config.noise_sigma] * num_noise_matrices
        print(f"Pre-creating {num_noise_matrices} noise matrices using Dask...")
        self.noise_matrices = batch_create_noise_matrices_dask(
            self.baseline_matrix, 
            sigma_list, 
            base_seed=self.config.random_seed
        )
        self.current_noise_matrix_index = 0
        self.noisy_matrix = self.noise_matrices[0]
        print("Noise matrices created successfully!")
        
        # 3) create agents
        self._create_agents()
        # 4) build network
        builder = NetworkBuilder(
            network_type=self.config.network_type,
            connectivity=self.config.connectivity,
            random_seed=self.config.random_seed
        )
        self.network = builder.build(self.agents)

    def _create_agents(self):
        counts = {}
        remaining = self.config.network_size
        types = list(self.config.agent_ratios)
        # assign counts by ratio
        for t in types[:-1]:
            cnt = int(self.config.network_size * self.config.agent_ratios[t])
            counts[t] = cnt
            remaining -= cnt
        counts[types[-1]] = remaining

        agent_id = 0
        shape = (self.config.matrix_rows, self.config.matrix_cols)
        for t, cnt in counts.items():
            for _ in range(cnt):
                if t == "M0":
                    self.agents.append(M0Agent(agent_id, shape))
                elif t == "M1":
                    self.agents.append(M1Agent(agent_id, shape))
                elif t == "M3":
                    self.agents.append(M3Agent(agent_id, shape))
                agent_id += 1

        np.random.shuffle(self.agents)

    def run_single_iteration(self):
        # periodically refresh noise matrix by selecting pre-created one
        if self.current_run > 0 and self.current_run % self.config.noise_update_frequency == 0:
            self.current_noise_matrix_index += 1
            if self.current_noise_matrix_index < len(self.noise_matrices):
                self.noisy_matrix = self.noise_matrices[self.current_noise_matrix_index]

        # Collect M3 agents and their tasks for batch processing
        m3_agents = []
        m3_tasks = []
        m3_positions = []  # To remember (row, col) for each M3 agent
        
        # Process M0 and M1 agents individually, collect M3 agent tasks
        for agent in self.agents:
            row = np.random.randint(0, self.config.matrix_rows)
            col = np.random.randint(0, self.config.matrix_cols)
            
            if agent.agent_type == "M3":
                # Collect for batch processing
                m3_agents.append(agent)
                # Determine reference column based on M3Agent logic
                if col == 0:
                    ref_col = 1
                elif col == 1:
                    ref_col = 0
                elif col == 2:
                    ref_col = 3
                elif col == 3:
                    ref_col = 2
                
                m3_tasks.append((col, row, ref_col))
                m3_positions.append((row, col))
            else:
                # Process M0 and M1 agents individually
                inf = agent.infer_cell(row, col)
                true = self.noisy_matrix.__getitem__((row, col))
                agent.record_error(true, inf)
                agent.store_true_value(row, col, true)
        
        # Batch process M3 agents if any
        if m3_agents:
            # Get mean matrices for batch inference
            m3_matrices = [agent.mean_matrix for agent in m3_agents]
            
            # Perform batch inference using Dask
            batch_results = batch_infer_with_dask(m3_matrices, m3_tasks, pairs_needed=100)
            
            # Process results for each M3 agent
            for i, (agent, (row, col)) in enumerate(zip(m3_agents, m3_positions)):
                inf = batch_results[i]
                true = self.noisy_matrix.__getitem__((row, col))
                agent.record_error(true, inf)
                agent.store_true_value(row, col, true)

        # Store snapshots at the end of each run
        current_agent_snapshots = []
        for agent in self.agents:
            current_agent_snapshots.append(agent.matrix_dict.data.copy())
        self.agent_matrix_snapshots.append(current_agent_snapshots)
        
        # Store current noisy matrix snapshot
        self.noisy_matrix_snapshots.append(self.noisy_matrix.data.copy())

        self.current_run += 1

    def run(self):
        print(f"Starting simulation with {len(self.agents)} agents…")
        m3_count = sum(1 for agent in self.agents if agent.agent_type == "M3")
        print(f"Using Dask parallelization for {m3_count} M3 agents per run")
        
        for run in range(self.config.total_runs):
            if run % 10 == 0:
                print(f" Run {run}/{self.config.total_runs}")
            self.run_single_iteration()
        print("Simulation complete!")

    def get_results(self) -> pd.DataFrame:
        rows = []
        for ag in self.agents:
            for i, err in enumerate(ag.prediction_errors):
                rows.append({
                    "agent_id":     ag.agent_id,
                    "agent_type":   ag.agent_type,
                    "run":          i,
                    "prediction_error":    err,
                    "abs_error":          abs(err),
                    "matrix_dict":  self.agent_matrix_snapshots[i][ag.agent_id],
                    "noisy_matrix": self.noisy_matrix_snapshots[i],
                })
        return pd.DataFrame(rows)
    
    def _compute_network_features(self):
        """Compute topological features of the network"""
        features = {}
        
        # Basic network properties
        features["num_nodes"] = self.network.number_of_nodes()
        features["num_edges"] = self.network.number_of_edges()
        features["density"] = nx.density(self.network)
        
        # Degree statistics
        degrees = [d for n, d in self.network.degree()]
        features["average_degree"] = np.mean(degrees)
        features["degree_std"] = np.std(degrees)
        features["min_degree"] = min(degrees)
        features["max_degree"] = max(degrees)
        
        # Clustering
        features["average_clustering"] = nx.average_clustering(self.network)
        
        # Path length metrics (only for connected graphs)
        if nx.is_connected(self.network):
            features["average_shortest_path_length"] = nx.average_shortest_path_length(self.network)
            features["diameter"] = nx.diameter(self.network)
            features["radius"] = nx.radius(self.network)
        else:
            # For disconnected graphs, compute on largest component
            largest_cc = max(nx.connected_components(self.network), key=len)
            subgraph = self.network.subgraph(largest_cc)
            features["largest_component_size"] = len(largest_cc)
            features["num_connected_components"] = nx.number_connected_components(self.network)
            features["average_shortest_path_length"] = nx.average_shortest_path_length(subgraph)
            features["diameter"] = nx.diameter(subgraph)
            features["radius"] = nx.radius(subgraph)
        
        # Centrality measures (top 5 nodes for each measure)
        betweenness = nx.betweenness_centrality(self.network)
        closeness = nx.closeness_centrality(self.network)
        
        features["top_betweenness_centrality"] = dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5])
        features["top_closeness_centrality"] = dict(sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Degree distribution
        degree_sequence = sorted(degrees, reverse=True)
        degree_count = {}
        for deg in degree_sequence:
            degree_count[deg] = degree_count.get(deg, 0) + 1
        features["degree_distribution"] = degree_count
        
        return features
    
    def get_simulation_data(self):
        """Return the simulation data for this configuration"""
        # Compute network topological features
        network_features = self._compute_network_features()
        
        # Create the nested structure
        result = {
            "network_size": self.config.network_size,
            "network_type": self.config.network_type,
            "connectivity": self.config.connectivity,
            "agent_ratios": self.config.agent_ratios,
            "matrix_rows": self.config.matrix_rows,
            "matrix_cols": self.config.matrix_cols,
            "total_runs": self.config.total_runs,
            "noise_update_frequency": self.config.noise_update_frequency,
            "noise_sigma": self.config.noise_sigma,
            "random_seed": self.config.random_seed,
            "network_features": network_features,
            "runs": {}
        }
        
        # Populate runs and agents
        for run_idx in range(self.config.total_runs):
            result["runs"][str(run_idx)] = {
                "noisy_matrix": {str(k): v for k, v in self.noisy_matrix_snapshots[run_idx].items()},
                "agents": {}
            }
            
            for agent in self.agents:
                if run_idx < len(agent.prediction_errors):
                    agent_data = {
                        "agent_type": agent.agent_type,
                        "prediction_error": agent.prediction_errors[run_idx],
                        "matrix_dict": {str(k): v for k, v in self.agent_matrix_snapshots[run_idx][agent.agent_id].items()}
                    }
                    result["runs"][str(run_idx)]["agents"][str(agent.agent_id)] = agent_data
        
        return result

    def write_results_to_json(self, filename: str):
        """Write single simulation results to JSON file (keeping original method)"""
        result = {"initialization": self.get_simulation_data()}
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {filename}")

    def plot(self):
        results_df = self.get_results()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean absolute error over time by agent type
        mean_errors = results_df.groupby(['agent_type', 'run'])['abs_error'].mean().reset_index()
        for agent_type in mean_errors['agent_type'].unique():
            data = mean_errors[mean_errors['agent_type'] == agent_type]
            axes[0, 0].plot(data['run'], data['abs_error'], label=f'Agent {agent_type}', linewidth=2)
        
        axes[0, 0].set_xlabel('Run')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('Mean Absolute Error Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of errors by agent type
        for i, agent_type in enumerate(['M0', 'M1', 'M3']):
            if agent_type in results_df['agent_type'].values:
                data = results_df[results_df['agent_type'] == agent_type]['prediction_error']
                axes[0, 1].hist(data, alpha=0.7, label=f'Agent {agent_type}', bins=30)
        
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Prediction Errors')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Network visualization
        pos = nx.spring_layout(self.network, seed=42)
        colors = {'M0': 'red', 'M1': 'blue', 'M3': 'green'}
        node_colors = [colors[agent.agent_type] for agent in self.agents]
        
        nx.draw(self.network, pos, ax=axes[1, 0], node_color=node_colors, 
                node_size=50, alpha=0.7, with_labels=False)
        axes[1, 0].set_title('Agent Network')
        
        # Create legend for network plot
        for agent_type, color in colors.items():
            axes[1, 0].scatter([], [], c=color, label=f'Agent {agent_type}', s=50)
        axes[1, 0].legend()
        
        # Plot 4: Performance comparison
        final_performance = results_df.groupby('agent_type')['abs_error'].agg(['mean', 'std']).reset_index()
        
        x_pos = range(len(final_performance))
        axes[1, 1].bar([x - 0.2 for x in x_pos], final_performance['mean'], 
                      width=0.4, label='Mean', alpha=0.7)
        axes[1, 1].errorbar([x - 0.2 for x in x_pos], final_performance['mean'], 
                           yerr=final_performance['std'], fmt='none', color='black', capsize=5)
        
        axes[1, 1].set_xlabel('Agent Type')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Final Performance Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(final_performance['agent_type'])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def generate_parameter_combinations():
    """Generate different parameter combinations for the simulation sweep"""
    
    # Define parameter ranges/options to sweep over
    param_grid = {
        'network_size': [50, 100, 200, 300],
        'network_type': ["small_world", "random", "scale_free"],
        'connectivity': [0.1, 0.2, 0.3],
        'agent_ratios': [
            {"M0": 0.1, "M1": 0.3, "M3": 0.6},
            {"M0": 0.6, "M1": 0.4, "M3": 0.1},
            {"M0": 0.3, "M1": 0.3, "M3": 0.4}
        ],
        'noise_sigma': [0.5, 1.0, 1.5],
        'total_runs': [50, 250, 500],  # 5 times of matrix_rows
        'noise_update_frequency': [10, 50],
        'matrix_rows': [10, 50, 100],  
        'matrix_cols': [4],   # Keep fixed
    }
    
    # Generate all combinations
    combinations = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


# Helper function to run a single simulation configuration
def _run_single(params: dict, i: int):
    # Create unique random seed for each configuration
    params['random_seed'] = 42 + i
    
    # Initialize and run the simulation
    config = SimulationConfig(**params)
    sim = Simulation(config)
    sim.run()
    
    # Retrieve simulation data and performance summary
    sim_data = sim.get_simulation_data()
    results_df = sim.get_results()
    final_performance = (
        results_df.groupby('agent_type')['abs_error']
                  .agg(['mean', 'std'])
                  .reset_index()
    )
    perf_stats = final_performance  # DataFrame for later printing
    init_key = f"initialization_{i}"
    return init_key, sim_data, perf_stats


def run_parameter_sweep(output_filename: str = "output/simulation_results_sweep.json"):
    """Run simulations across different parameter combinations in parallel using Dask"""
    
    # Generate all parameter combinations
    param_combinations = generate_parameter_combinations()
    print(f"Running {len(param_combinations)} parameter combinations in parallel...")
    
    # Create delayed tasks for each configuration
    tasks = [delayed(_run_single)(params.copy(), i) 
             for i, params in enumerate(param_combinations)]
    
    # Execute all tasks in parallel
    results = compute(*tasks)
    
    # Collect results and print performance summaries
    all_results = {}
    for init_key, sim_data, perf_stats in results:
        all_results[init_key] = sim_data
        print(f"\nPerformance summary for {init_key}:")
        for _, row in perf_stats.iterrows():
            print(f"  {row['agent_type']}: {row['mean']:.4f} ± {row['std']:.4f}")
    
    # Write all results to JSON
    print(f"\nWriting all results to {output_filename}...")
    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Complete! Results written to {output_filename}")
    
    return all_results



if __name__ == "__main__":
    # Run parameter sweep instead of single simulation
    all_results = run_parameter_sweep("output/simulation_results_sweep.json")
    
    print(f"\nParameter sweep complete! Generated {len(all_results)} different initializations.")