import numpy as np
from typing import Tuple, Optional
from util.social_environment import MatrixDict, infer_element_dict

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, agent_id: int, agent_type: str, matrix_shape: Tuple[int, int]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.matrix_dict = MatrixDict(shape=matrix_shape)
        self.neighbors = []
        self.prediction_errors = []
        self.inference_history = {}  # (row, col) -> inferred_value
        
    def add_neighbor(self, neighbor):
        """Add a neighbor agent"""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
    
    def has_value(self, row: int, col: int) -> bool:
        """Check if agent has a value for the given cell"""
        return (row, col) in self.matrix_dict
    
    def get_stored_value(self, row: int, col: int) -> Optional[float]:
        """Get stored value if it exists"""
        if self.has_value(row, col):
            return self.matrix_dict[(row, col)]
        return None
    
    def store_true_value(self, row: int, col: int, true_value: float):
        """Store the true value after getting feedback"""
        self.matrix_dict[(row, col)] = true_value
    
    def record_error(self, true_value: float, inferred_value: float):
        """Record prediction error"""
        error = inferred_value - true_value  # As specified: -(true - inferred)
        self.prediction_errors.append(error)
    
    def infer_cell(self, row: int, col: int) -> float:
        """Base inference method - to be overridden by subclasses"""
        raise NotImplementedError

class M0Agent(BaseAgent):
    """Agent that makes random inferences from N(0,1)"""
    
    def __init__(self, agent_id: int, matrix_shape: Tuple[int, int]):
        super().__init__(agent_id, "M0", matrix_shape)
    
    def infer_cell(self, row: int, col: int) -> float:
        """Return random value from standard normal distribution"""
        # Check if already inferred
        stored_value = self.get_stored_value(row, col)
        if stored_value is not None:
            return stored_value
        
        # Make random inference
        inference = np.random.normal(0, 1)
        return inference

class M3Agent(BaseAgent):
    """Agent that uses the infer_element_dict function with mean tracking"""
    
    def __init__(self, agent_id: int, matrix_shape: Tuple[int, int]):
        super().__init__(agent_id, "M3", matrix_shape)
        # M3 has two additional matrices for tracking means and counts
        self.count_matrix = MatrixDict(shape=matrix_shape)  # Records number of times each cell was inferred
        self.mean_matrix = MatrixDict(shape=matrix_shape)   # Stores mean of past true values for each cell
        # The inherited matrix_dict is not used for M3, we use mean_matrix instead
    
    def has_value(self, row: int, col: int) -> bool:
        """Check if agent has a mean value for the given cell"""
        return (row, col) in self.mean_matrix
    
    def get_stored_value(self, row: int, col: int) -> Optional[float]:
        """Get stored mean value if it exists"""
        if self.has_value(row, col):
            return self.mean_matrix[(row, col)]
        return None
    
    def store_true_value(self, row: int, col: int, true_value: float):
        """Update running mean and count for the cell"""
        # Get current count and mean (MatrixDict returns np.nan for missing keys)
        current_count_val = self.count_matrix[(row, col)]
        current_count = 0 if np.isnan(current_count_val) else current_count_val
        
        current_mean_val = self.mean_matrix[(row, col)]
        current_mean = 0.0 if np.isnan(current_mean_val) else current_mean_val
        
        # Update count
        new_count = current_count + 1
        self.count_matrix[(row, col)] = new_count
        
        # Update running mean using the formula: new_mean = (old_mean * old_count + new_value) / new_count
        if current_count == 0:
            new_mean = true_value
        else:
            new_mean = (current_mean * current_count + true_value) / new_count
        
        self.mean_matrix[(row, col)] = new_mean
        
        # Also store in the inherited matrix_dict for compatibility (though we don't use it for inference)
        self.matrix_dict[(row, col)] = new_mean
    
    def infer_cell(self, row: int, col: int) -> float:
        """Use infer_element_dict to make inference using mean values"""
        # Check if we already have a mean value for this cell
        
        if col == 0:
            ref_col = 1
        elif col == 1:
            ref_col = 0
        elif col == 2:
            ref_col = 3
        elif col == 3:
            ref_col = 2
    
        return infer_element_dict(
                        self.mean_matrix,
                        target_column=col,
                        target_row=row,
                        reference_column=ref_col,
                        pairs_needed=100,
                        backup_value=0.0
                    )
    
    def get_cell_statistics(self, row: int, col: int) -> Tuple[int, float]:
        """Get count and mean for a specific cell (for debugging/analysis)"""
        count_val = self.count_matrix[(row, col)]
        count = 0 if np.isnan(count_val) else int(count_val)
        
        mean_val = self.mean_matrix[(row, col)]
        mean = 0.0 if np.isnan(mean_val) else mean_val
        
        return count, mean

class M1Agent(BaseAgent):
    """Agent that uses neighbor information for inference"""
    
    def __init__(self, agent_id: int, matrix_shape: Tuple[int, int]):
        super().__init__(agent_id, "M1", matrix_shape)
    
    def infer_cell(self, row: int, col: int) -> float:
        """Use neighbor information to make inference"""
        # Check if already inferred
        stored_value = self.get_stored_value(row, col)
        if stored_value is not None:
            return stored_value
        
        neighbor_values = []
        maximal_neighbors = 20
        
        for neighbor in self.neighbors:
            if neighbor.agent_type in ["M0", "M3"]:
                # Get neighbor's inference 
                neighbor_inference = 0 if not neighbor.has_value(row, col) else neighbor.infer_cell(row, col)
                neighbor_values.append(neighbor_inference)
            
            elif neighbor.agent_type == "M1":
                # Get neighbor's gossiped value (stored value if available)
                gossiped_value = neighbor.get_stored_value(row, col)
                if gossiped_value is not None and len(neighbor_values) < maximal_neighbors:
                    neighbor_values.append(gossiped_value)
                # If neighbor doesn't have the value, skip (don't recursive call)
        
        # If no neighbor values available, return random value
        if not neighbor_values:
            return np.random.normal(0, 1)
        
        # Return mean of neighbor values
        return np.mean(neighbor_values)
