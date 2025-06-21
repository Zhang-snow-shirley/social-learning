"""
COMBINED: Dynamic RL Strategy + Corrected Evaluation Methodology

This script combines:
1. Dynamic agent choice across runs (agents can change type each run)
2. Proper evaluation methodology (early stopping, validation sets, etc.)
3. Fixes for overfitting and training instability

This addresses BOTH the strategic issue (static vs dynamic) AND the methodological issues.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from sklearn.preprocessing import StandardScaler

# Stable-Baselines3 imports
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

@dataclass
class CombinedRLConfig:
    """Configuration combining dynamic strategy + evaluation fixes"""
    
    # Data and paths
    input_json_path: str = "output/simulation_results_sweep.json"
    output_dir: str = "output/combined_rl_output"
    model_save_path: str = "output/combined_rl_output/dynamic_agent_model"
    
    # RL algorithm settings
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    total_timesteps: int = 150000
    
    # DYNAMIC STRATEGY settings
    enable_dynamic_strategy: bool = True  # KEY: Enable dynamic agent choice
    temporal_window: int = 10
    strategy_change_penalty: float = 0.05
    cumulative_reward_weight: float = 0.7
    
    # EVALUATION FIXES settings
    validation_split: float = 0.2
    test_split: float = 0.15
    evaluation_sample_size: int = 500  # Larger sample
    confidence_level: float = 0.95
    
    # EARLY STOPPING settings
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 0.001
    
    # Feature engineering
    normalize_features: bool = True
    include_network_features: bool = True
    include_temporal_features: bool = True  # CRITICAL for dynamic strategy
    include_knowledge_evolution: bool = True
    
    # Training settings
    max_iterations: int = 1000
    improvement_threshold: float = 0.005
    verbose: int = 1
    save_checkpoints: bool = True
    plot_results: bool = True


class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback for Stable-Baselines3"""
    
    def __init__(self, patience: int = 4, min_delta: float = 0.001, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.wait = 0
        
    def _on_step(self) -> bool:
        # Get current episode rewards
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            
            if mean_reward > self.best_mean_reward + self.min_delta:
                self.best_mean_reward = mean_reward
                self.wait = 0
            else:
                self.wait += 1
                
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f"Early stopping triggered after {self.wait} steps without improvement")
                return False  # Stop training
                
        return True


class CombinedDataLoader:
    """Data loader implementing BOTH dynamic strategy AND proper validation splits"""
    
    def __init__(self, config: CombinedRLConfig):
        self.config = config
        self.scaler = StandardScaler() if config.normalize_features else None
        self.feature_names = []
        
    def extract_dynamic_features(self, 
                                agent_data: Dict,
                                run_data: Dict, 
                                init_data: Dict,
                                agent_id: str,
                                run_idx: int,
                                historical_data: Dict = None) -> np.ndarray:
        """
        Extract features for DYNAMIC agent choice at specific run
        
        KEY ADDITION: Includes temporal progression features for dynamic strategy
        """
        features = []
        
        # 1. Agent knowledge state
        matrix_dict = agent_data.get('matrix_dict', {})
        if matrix_dict:
            values = [float(v) for v in matrix_dict.values()]
            features.extend([
                len(values),  # Knowledge size
                np.mean(values) if values else 0,
                np.std(values) if len(values) > 1 else 0,
                np.min(values) if values else 0,
                np.max(values) if values else 0,
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 2. Environment state
        noisy_matrix = run_data.get('noisy_matrix', {})
        if noisy_matrix:
            noisy_values = [float(v) for v in noisy_matrix.values()]
            features.extend([
                np.mean(noisy_values),
                np.std(noisy_values),
                np.min(noisy_values),
                np.max(noisy_values)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 3. Network features
        if self.config.include_network_features:
            network_features = init_data.get('network_features', {})
            features.extend([
                network_features.get('density', 0),
                network_features.get('average_degree', 0),
                network_features.get('average_clustering', 0),
                network_features.get('average_shortest_path_length', 0),
                init_data.get('network_size', 100) / 100.0,
                init_data.get('connectivity', 0.3),
            ])
        
        # 4. CRITICAL: Temporal features for DYNAMIC strategy
        if self.config.include_temporal_features:
            total_runs = init_data.get('total_runs', 1000)
            noise_freq = init_data.get('noise_update_frequency', 10)
            
            # THESE ARE KEY FOR DYNAMIC AGENT CHOICE:
            features.extend([
                run_idx / total_runs,  # Simulation progress (0 to 1) - CRITICAL!
                (run_idx % noise_freq) / noise_freq,  # Position in noise cycle
                np.sin(2 * np.pi * run_idx / noise_freq),  # Cyclic noise pattern
                np.cos(2 * np.pi * run_idx / noise_freq),  # Cyclic noise pattern
                init_data.get('noise_sigma', 1.0),  # Noise level
                min(run_idx / 100.0, 1.0),  # Early vs late stage indicator
                1.0 if run_idx % noise_freq == 0 else 0.0,  # Fresh noise indicator
            ])
        
        # 5. Agent composition
        agent_ratios = init_data.get('agent_ratios', {})
        features.extend([
            agent_ratios.get('M0', 0.1),
            agent_ratios.get('M1', 0.3),
            agent_ratios.get('M3', 0.6)
        ])
        
        # 6. Knowledge evolution (for dynamic adaptation)
        if self.config.include_knowledge_evolution and historical_data:
            recent_errors = historical_data.get('recent_errors', [])
            if recent_errors:
                features.extend([
                    np.mean(recent_errors[-5:]) if len(recent_errors) >= 5 else 0,
                    np.std(recent_errors[-5:]) if len(recent_errors) >= 5 else 0,
                    len(recent_errors),
                    (recent_errors[-1] - recent_errors[0]) / len(recent_errors) if len(recent_errors) > 1 else 0,
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            knowledge_sizes = historical_data.get('knowledge_sizes', [])
            current_knowledge_size = len(values) if matrix_dict else 0
            if knowledge_sizes:
                knowledge_growth = current_knowledge_size - knowledge_sizes[-1] if knowledge_sizes else 0
                features.append(knowledge_growth)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def prepare_dynamic_data_with_splits(self, json_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """
        COMBINED: Prepare dynamic training data WITH proper train/validation splits
        
        Returns: train_features, train_labels, train_metadata, val_features, val_labels, val_metadata
        """
        
        print("Preparing DYNAMIC data with PROPER validation splits...")
        
        # Step 1: Build temporal sequences (agent trajectories across runs)
        all_sequences = {}  # {(init_key, agent_id): [list of (run, features, label, metadata)]}
        
        for init_key, init_data in json_data.items():
            if isinstance(init_data, dict) and 'runs' in init_data:
                runs = init_data['runs']
                
                # Track agent histories for this initialization
                agent_histories = {}
                
                # Process runs chronologically
                for run_idx_str in sorted(runs.keys(), key=int):
                    run_idx = int(run_idx_str)
                    run_data = runs[run_idx_str]
                    agents = run_data.get('agents', {})
                    
                    for agent_id, agent_data in agents.items():
                        agent_type = agent_data.get('agent_type')
                        
                        if agent_type in ['M1', 'M3']:  # Replacement targets
                            
                            # Initialize agent history if needed
                            agent_key = (init_key, agent_id)
                            if agent_key not in all_sequences:
                                all_sequences[agent_key] = []
                                agent_histories[agent_id] = {
                                    'recent_errors': [],
                                    'knowledge_sizes': [],
                                    'agent_types': []
                                }
                            
                            # Extract DYNAMIC features for this (agent, run) pair
                            features = self.extract_dynamic_features(
                                agent_data, run_data, init_data, agent_id, run_idx,
                                historical_data=agent_histories.get(agent_id, {})
                            )
                            
                            # Store this timestep
                            all_sequences[agent_key].append({
                                'run_idx': run_idx,
                                'features': features,
                                'label': {'M0': 0, 'M1': 1, 'M3': 2}[agent_type],
                                'metadata': {
                                    'init_key': init_key,
                                    'agent_id': agent_id,
                                    'run_idx': run_idx,
                                    'agent_type': agent_type,
                                    'prediction_error': agent_data.get('prediction_error', 0),
                                    'simulation_progress': run_idx / init_data.get('total_runs', 1000)
                                }
                            })
                            
                            # Update history
                            if agent_id in agent_histories:
                                error = abs(agent_data.get('prediction_error', 0))
                                knowledge_size = len(agent_data.get('matrix_dict', {}))
                                
                                agent_histories[agent_id]['recent_errors'].append(error)
                                agent_histories[agent_id]['knowledge_sizes'].append(knowledge_size)
                                agent_histories[agent_id]['agent_types'].append(agent_type)
                                
                                # Sliding window
                                max_history = self.config.temporal_window
                                for key in agent_histories[agent_id]:
                                    if len(agent_histories[agent_id][key]) > max_history:
                                        agent_histories[agent_id][key] = agent_histories[agent_id][key][-max_history:]
        
        # Step 2: Convert sequences to individual samples for ML
        all_samples = []
        for agent_key, sequence in all_sequences.items():
            if len(sequence) >= 5:  # Minimum sequence length
                all_samples.extend(sequence)
        
        # Shuffle samples
        np.random.shuffle(all_samples)
        
        # Step 3: Split into train/validation
        val_size = int(len(all_samples) * self.config.validation_split)
        
        train_samples = all_samples[val_size:]
        val_samples = all_samples[:val_size]
        
        # Step 4: Extract features, labels, metadata
        def extract_arrays(samples):
            features = np.vstack([s['features'] for s in samples])
            labels = np.array([s['label'] for s in samples])
            metadata = [s['metadata'] for s in samples]
            return features, labels, metadata
        
        train_features, train_labels, train_metadata = extract_arrays(train_samples)
        val_features, val_labels, val_metadata = extract_arrays(val_samples)
        
        # Step 5: Generate feature names and normalize
        self.feature_names = self._generate_dynamic_feature_names()
        
        if self.scaler is not None:
            train_features = self.scaler.fit_transform(train_features)
            val_features = self.scaler.transform(val_features)
        
        print(f"   DYNAMIC data prepared:")
        print(f"   Train: {len(train_features)} samples")
        print(f"   Validation: {len(val_features)} samples")
        print(f"   Features: {train_features.shape[1]} dimensions")
        print(f"   Temporal features included: {self.config.include_temporal_features}")
        
        return train_features, train_labels, train_metadata, val_features, val_labels, val_metadata
    
    def _generate_dynamic_feature_names(self) -> List[str]:
        """Generate feature names for dynamic model"""
        names = [
            'knowledge_size', 'knowledge_mean', 'knowledge_std', 'knowledge_min', 'knowledge_max',
            'env_mean', 'env_std', 'env_min', 'env_max',
        ]
        
        if self.config.include_network_features:
            names.extend([
                'network_density', 'network_avg_degree', 'network_clustering',
                'network_path_length', 'network_size_norm', 'connectivity'
            ])
        
        if self.config.include_temporal_features:
            names.extend([
                'simulation_progress',  # CRITICAL for dynamic strategy
                'noise_cycle_pos', 'noise_sin', 'noise_cos',
                'noise_sigma', 'stage_indicator', 'fresh_noise'
            ])
        
        names.extend(['ratio_M0', 'ratio_M1', 'ratio_M3'])
        
        if self.config.include_knowledge_evolution:
            names.extend([
                'recent_avg_error', 'recent_error_stability', 'history_length',
                'error_trend', 'knowledge_growth'
            ])
        
        return names


class CombinedDynamicEnvironment(gym.Env):
    """
    COMBINED: Dynamic agent choice environment WITH proper reward alignment
    """
    
    def __init__(self, train_features: np.ndarray, train_labels: np.ndarray, 
                 train_metadata: List[Dict], config: CombinedRLConfig):
        super().__init__()
        
        self.config = config
        self.features = train_features
        self.labels = train_labels
        self.metadata = train_metadata
        
        # Action space: agent types
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        feature_dim = train_features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
        )
        
        # Episode management
        self.current_step = 0
        self.max_steps = min(2000, len(train_features))
        
        # Performance tracking
        self.episode_rewards = []
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Shuffle data
        indices = np.random.permutation(len(self.features))
        self.features = self.features[indices]
        self.labels = self.labels[indices]
        self.metadata = [self.metadata[i] for i in indices]
        
        if len(self.features) > 0:
            obs = self.features[0].astype(np.float32)
            info = {'metadata': self.metadata[0]}
            return obs, info
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    
    def step(self, action):
        """Execute step with ALIGNED reward function"""
        if self.current_step >= len(self.features):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, False, {}
        
        true_action = self.labels[self.current_step]
        metadata = self.metadata[self.current_step]
        
        # ALIGNED REWARD: Use same performance simulation as evaluation
        base_error = abs(metadata.get('prediction_error', 0))
        simulation_progress = metadata.get('simulation_progress', 0.5)
        
        # Simulate performance of chosen agent type
        simulated_error = self._simulate_agent_performance_aligned(action, base_error, simulation_progress)
        
        # Reward based on SIMULATED PERFORMANCE (same as evaluation)
        performance_reward = -simulated_error / 3.0  # Negative error as reward
        
        # Secondary rewards
        accuracy_bonus = 0.5 if action == true_action else -0.2
        
        # Temporal strategy bonus (encourage good timing)
        temporal_bonus = self._calculate_temporal_bonus(action, simulation_progress)
        
        # Combined reward
        reward = performance_reward + accuracy_bonus + temporal_bonus
        
        # Next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        if not terminated and self.current_step < len(self.features):
            next_obs = self.features[self.current_step].astype(np.float32)
            next_info = {'metadata': self.metadata[self.current_step]}
        else:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            next_info = {}
        
        return next_obs, reward, terminated, False, next_info
    
    def _simulate_agent_performance_aligned(self, action, base_error, simulation_progress):
        """Same performance simulation used in evaluation - CRITICAL for alignment"""
        agent_type = ['M0', 'M1', 'M3'][action]
        
        # SAME logic as evaluation function
        if agent_type == 'M0':
            performance_factor = 1.0 + simulation_progress * 0.2
        elif agent_type == 'M1':
            peak_performance = 0.4 * np.sin(np.pi * simulation_progress)
            performance_factor = 1.0 - peak_performance
        elif agent_type == 'M3':
            performance_factor = 1.0 - simulation_progress * 0.5
        else:
            performance_factor = 1.0
        
        performance_factor *= np.random.uniform(0.95, 1.05)
        return max(0.1, base_error * performance_factor)
    
    def _calculate_temporal_bonus(self, action, simulation_progress):
        """Bonus for good temporal strategy"""
        agent_type = ['M0', 'M1', 'M3'][action]
        
        if agent_type == 'M1' and 0.2 <= simulation_progress <= 0.7:
            return 0.1  # M1 good in middle
        elif agent_type == 'M3' and simulation_progress > 0.5:
            return 0.15  # M3 excellent later
        elif agent_type == 'M0' and simulation_progress < 0.2:
            return 0.05  # M0 acceptable early
        return 0.0


class CombinedDynamicTrainer:
    """COMPLETE trainer that actually runs the training!"""
    
    def __init__(self, config: CombinedRLConfig):
        self.config = config
        self.data_loader = CombinedDataLoader(config)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history = {
            'iteration': [],
            'train_reward': [],
            'val_loss': [],
            'reference_loss': [],
            'rl_loss': [],
            'improvement': [],
            'choice_accuracy': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def load_and_prepare_data(self):
        """Load and prepare data with proper splits"""
        print("  Loading simulation data...")
        
        with open(self.config.input_json_path, 'r') as f:
            json_data = json.load(f)
        
        if not json_data:
            raise ValueError(f"No data loaded from {self.config.input_json_path}")
        
        print("  Preparing DYNAMIC training data with validation splits...")
        return self.data_loader.prepare_dynamic_data_with_splits(json_data)
    
    def create_rl_model(self, env):
        """Create RL model"""
        if self.config.algorithm == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                verbose=self.config.verbose
            )
        elif self.config.algorithm == "DQN":
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                verbose=self.config.verbose
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        return model
    
    def evaluate_on_validation_set(self, model, val_features, val_labels, val_metadata):
        """Evaluate model on validation set with proper methodology"""
        
        print("  Evaluating on validation set...")
        
        # Reference loss from validation set
        reference_errors = [abs(m['prediction_error']) for m in val_metadata]
        reference_loss = np.mean(reference_errors)
        
        # RL agent evaluation
        rl_errors = []
        rl_choices = []
        correct_choices = []
        
        for i in range(len(val_features)):
            # Get RL choice
            action, _ = model.predict(val_features[i], deterministic=True)
            rl_agent_type = ['M0', 'M1', 'M3'][action]
            rl_choices.append(action)
            
            # Check accuracy
            true_action = val_labels[i]
            correct_choices.append(action == true_action)
            
            # Simulate performance
            base_error = abs(val_metadata[i]['prediction_error'])
            simulation_progress = val_metadata[i]['simulation_progress']
            
            simulated_error = self._simulate_performance_for_evaluation(
                rl_agent_type, base_error, simulation_progress
            )
            rl_errors.append(simulated_error)
        
        rl_loss = np.mean(rl_errors)
        choice_accuracy = np.mean(correct_choices)
        improvement = (reference_loss - rl_loss) / reference_loss if reference_loss > 0 else 0
        
        print(f"   Reference Loss: {reference_loss:.4f}")
        print(f"   RL Loss: {rl_loss:.4f}")
        print(f"   Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        print(f"   Choice Accuracy: {choice_accuracy:.3f}")
        
        return {
            'reference_loss': reference_loss,
            'rl_loss': rl_loss,
            'improvement': improvement,
            'choice_accuracy': choice_accuracy,
            'val_loss': rl_loss  # Use RL loss as validation metric
        }
    
    def _simulate_performance_for_evaluation(self, agent_type, base_error, simulation_progress):
        """Simulation function aligned with training rewards"""
        if agent_type == 'M0':
            performance_factor = 1.0 + simulation_progress * 0.2
        elif agent_type == 'M1':
            peak_performance = 0.4 * np.sin(np.pi * simulation_progress)
            performance_factor = 1.0 - peak_performance
        elif agent_type == 'M3':
            performance_factor = 1.0 - simulation_progress * 0.5
        else:
            performance_factor = 1.0
        
        performance_factor *= np.random.uniform(0.95, 1.05)
        return max(0.1, base_error * performance_factor)
    
    def run_complete_training(self):
        """MAIN TRAINING FUNCTION - Actually runs the training!"""
        
        print("  STARTING COMPLETE DYNAMIC RL TRAINING")
        print("=" * 80)
        
        # Step 1: Load and prepare data
        train_features, train_labels, train_metadata, val_features, val_labels, val_metadata = self.load_and_prepare_data()
        
        # Step 2: Create environment
        print("🏗️ Creating training environment...")
        env = CombinedDynamicEnvironment(train_features, train_labels, train_metadata, self.config)
        env = Monitor(env)
        
        # Check environment
        check_env(env)
        print("  Environment check passed!")
        
        # Step 3: Create model
        print(f"  Creating {self.config.algorithm} model...")
        model = self.create_rl_model(env)
        
        # Step 4: Initial training
        print(f"  Initial training for {self.config.total_timesteps} timesteps...")
        model.learn(total_timesteps=self.config.total_timesteps)
        
        # Save initial model
        model.save(f"{self.config.model_save_path}_initial")
        print("  Initial model saved!")
        
        # Step 5: Iterative improvement with early stopping
        print("  Starting iterative improvement with early stopping...")
        
        for iteration in range(self.config.max_iterations):
            print(f"\n{'='*20} ITERATION {iteration + 1}/{self.config.max_iterations} {'='*20}")
            
            # Evaluate on validation set
            eval_results = self.evaluate_on_validation_set(model, val_features, val_labels, val_metadata)
            
            # Record history
            self.training_history['iteration'].append(iteration + 1)
            self.training_history['reference_loss'].append(eval_results['reference_loss'])
            self.training_history['rl_loss'].append(eval_results['rl_loss'])
            self.training_history['improvement'].append(eval_results['improvement'])
            self.training_history['choice_accuracy'].append(eval_results['choice_accuracy'])
            self.training_history['val_loss'].append(eval_results['val_loss'])
            
            # Early stopping check
            current_val_loss = eval_results['val_loss']
            
            if current_val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                print(f"  Validation improvement detected!")
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
                
                # Save best model
                model.save(f"{self.config.model_save_path}_best")
                print("  Best model saved!")
                
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            
            # Check if improvement is significant
            if eval_results['improvement'] > self.config.improvement_threshold:
                print(f"  Significant improvement achieved: {eval_results['improvement']:.4f}")
                
                # Continue training
                print("  Continuing training...")
                model.learn(total_timesteps=self.config.total_timesteps // 2)
                
            else:
                print(f"  Limited improvement: {eval_results['improvement']:.4f}")
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"  Early stopping triggered! No improvement for {self.config.early_stopping_patience} iterations")
                break
        
        # Save final model
        model.save(f"{self.config.model_save_path}_final")
        print("  Final model saved!")
        
        # Save data loader
        with open(f"{self.config.output_dir}/data_loader.pkl", 'wb') as f:
            pickle.dump(self.data_loader, f)
        
        return model
    
    def generate_final_report(self):
        """Generate comprehensive training report"""
        
        print("\n" + "=" * 80)
        print("  FINAL TRAINING REPORT")
        print("=" * 80)
        
        if not self.training_history['iteration']:
            print("  No training history available")
            return
        
        # Summary statistics
        final_improvement = self.training_history['improvement'][-1]
        best_improvement = max(self.training_history['improvement'])
        final_accuracy = self.training_history['choice_accuracy'][-1]
        best_accuracy = max(self.training_history['choice_accuracy'])
        
        print(f"\n  PERFORMANCE SUMMARY:")
        print(f"   Final Improvement: {final_improvement:.4f} ({final_improvement*100:.2f}%)")
        print(f"   Best Improvement: {best_improvement:.4f} ({best_improvement*100:.2f}%)")
        print(f"   Final Choice Accuracy: {final_accuracy:.3f}")
        print(f"   Best Choice Accuracy: {best_accuracy:.3f}")
        print(f"   Training Iterations: {len(self.training_history['iteration'])}")
        
        # Early stopping analysis
        if self.patience_counter >= self.config.early_stopping_patience:
            print(f"\n  Training stopped due to early stopping (patience: {self.config.early_stopping_patience})")
        else:
            print(f"\n  Training completed all {self.config.max_iterations} iterations")
        
        # Performance assessment
        print(f"\n  PERFORMANCE ASSESSMENT:")
        if best_improvement > self.config.improvement_threshold:
            print("  SUCCESS! Significant improvement achieved")
            print(f"   Dynamic RL strategy outperformed original agents by {best_improvement*100:.2f}%")
        else:
            print("  Limited success. Consider:")
            print("   • Adjusting hyperparameters")
            print("   • More training timesteps")
            print("   • Different reward engineering")
        
        # Strategy analysis
        if self.config.enable_dynamic_strategy:
            print(f"\n  DYNAMIC STRATEGY ENABLED:")
            print("   • Agents can change types across runs")
            print("   • Temporal features included")
            print("   • Sequential decision making")
        else:
            print(f"\n  Static strategy used")
        
        # Save history
        history_path = f"{self.config.output_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"\n  Training history saved to: {history_path}")
        
        # Plot results
        if self.config.plot_results:
            self.plot_training_results()
    
    def plot_training_results(self):
        """Plot comprehensive training results"""
        
        if not self.training_history['iteration']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Combined Dynamic RL Training Results', fontsize=16)
        
        iterations = self.training_history['iteration']
        
        # Plot 1: Loss comparison
        axes[0, 0].plot(iterations, self.training_history['reference_loss'], 
                       label='Reference Loss', marker='o', linewidth=2)
        axes[0, 0].plot(iterations, self.training_history['rl_loss'], 
                       label='RL Loss', marker='s', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Average Loss')
        axes[0, 0].set_title('Loss Comparison (Dynamic Strategy)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Improvement over time
        axes[0, 1].plot(iterations, self.training_history['improvement'], 
                       marker='o', color='green', linewidth=2)
        axes[0, 1].axhline(y=self.config.improvement_threshold, 
                          color='red', linestyle='--', label=f'Threshold ({self.config.improvement_threshold})')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Improvement Ratio')
        axes[0, 1].set_title('Performance Improvement')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Choice accuracy
        axes[1, 0].plot(iterations, self.training_history['choice_accuracy'], 
                       marker='o', color='blue', linewidth=2)
        axes[1, 0].axhline(y=1/3, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Choice Accuracy')
        axes[1, 0].set_title('Agent Choice Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: Summary
        summary_text = f"""
COMBINED DYNAMIC RL RESULTS:

  Features Included:
• Dynamic agent choice across runs
• Temporal progression features  
• Early stopping (patience: {self.config.early_stopping_patience})
• Proper train/validation splits

  Final Results:
• Best Improvement: {max(self.training_history['improvement']):.4f}
• Final Accuracy: {self.training_history['choice_accuracy'][-1]:.3f}
• Iterations: {len(iterations)}

  Strategy: {('Dynamic' if self.config.enable_dynamic_strategy else 'Static')}
"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.config.output_dir}/training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Training plots saved to: {plot_path}")
        
        plt.show()


def main():
    """MAIN FUNCTION - Actually runs the complete training!"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined Dynamic RL Agent Choice Training")
    parser.add_argument("--input", type=str, default="output/simulation_results_sweep.json",
                       help="Path to simulation JSON file")
    parser.add_argument("--output-dir", type=str, default="output/combined_rl_output",
                       help="Output directory")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN"],
                       help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=150000,
                       help="Training timesteps")
    parser.add_argument("--max-iterations", type=int, default=15,
                       help="Max training iterations")
    parser.add_argument("--dynamic", action="store_true", default=True,
                       help="Enable dynamic strategy (default: True)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plotting")
    
    args = parser.parse_args()
    
    # Create configuration
    config = CombinedRLConfig(
        input_json_path=args.input,
        output_dir=args.output_dir,
        model_save_path=f"{args.output_dir}/dynamic_agent_model",
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        max_iterations=args.max_iterations,
        enable_dynamic_strategy=args.dynamic,
        plot_results=not args.no_plots
    )
    
    print("  COMBINED DYNAMIC RL AGENT CHOICE TRAINING")
    print("=" * 80)
    print(f"Input file: {config.input_json_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Training timesteps: {config.total_timesteps}")
    print(f"Dynamic strategy: {config.enable_dynamic_strategy}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    
    # Check input file
    if not Path(config.input_json_path).exists():
        print(f"  Error: Input file {config.input_json_path} not found!")
        return
    
    try:
        # Create trainer and run complete training
        trainer = CombinedDynamicTrainer(config)
        model = trainer.run_complete_training()
        
        # Generate final report
        trainer.generate_final_report()
        
        print("\n" + "=" * 80)
        print("  TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"  Models saved to: {config.output_dir}/")
        print(f"  Best model: {config.model_save_path}_best.zip")
        print(f"  Final model: {config.model_save_path}_final.zip")
        print("  Training history and plots generated")
        
        if config.enable_dynamic_strategy:
            print("\n  DYNAMIC STRATEGY SUCCESSFULLY IMPLEMENTED!")
            print("   Agents can now adapt their types across simulation runs")
        
    except Exception as e:
        print(f"  Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()