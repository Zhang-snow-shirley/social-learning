import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from dataclasses import dataclass
from typing import Dict

from util.social_environment import simulate_correlated_matrix_dict, add_noise_to_matrix
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

        self._initialize()

    def _initialize(self):
        np.random.seed(self.config.random_seed)
        # 1) simulate true baseline
        self.baseline_matrix = simulate_correlated_matrix_dict(
            A=self.config.matrix_rows,
            C=self.config.matrix_cols,
            random_seed=self.config.random_seed
        )
        # 2) add initial noise
        self.noisy_matrix = add_noise_to_matrix(
            self.baseline_matrix,
            sigma=self.config.noise_sigma,
            random_seed=self.config.random_seed
        )
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
        # periodically refresh noise
        if self.current_run > 0 and self.current_run % self.config.noise_update_frequency == 0:
            self.noisy_matrix = add_noise_to_matrix(
                self.baseline_matrix,
                sigma=self.config.noise_sigma,
                random_seed=self.config.random_seed + self.current_run
            )

        for agent in self.agents:
            row = np.random.randint(0, self.config.matrix_rows)
            col = np.random.randint(0, self.config.matrix_cols)
            inf = agent.infer_cell(row, col)
            true = self.noisy_matrix.__getitem__((row, col))
            agent.record_error(true, inf)
            agent.store_true_value(row, col, true)

        self.current_run += 1

    def run(self):
        print(f"Starting simulation with {len(self.agents)} agents…")
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
                    "abs_error":          abs(err)
                })
        return pd.DataFrame(rows)

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
    
if __name__ == "__main__":
    config = SimulationConfig(
        network_size=75,
        network_type="small_world",
        connectivity=0.2,
        agent_ratios={"M0": 0.2, "M1": 0.3, "M3": 0.5},
        matrix_rows=10,
        matrix_cols=4,
        total_runs=1000,
        noise_update_frequency=10,
        noise_sigma=1,
        random_seed=42
    )
    
    sim = Simulation(config)
    sim.run()
    results_df = sim.get_results()
    print(results_df.head(100))
    # Report final performance
    final_performance = results_df.groupby('agent_type')['abs_error'].agg(['mean', 'std']).reset_index()
    final_performance.columns = ['agent_type', 'mean_abs_error', 'std_abs_error']
    print("Final Performance:")
    print(final_performance)
    
    fig = sim.plot()
    plt.show()
    #save the figure if needed
    fig.savefig("output/simulation_results.png", dpi=300, bbox_inches='tight')

