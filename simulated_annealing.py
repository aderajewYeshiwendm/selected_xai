"""
Simulated Annealing for Feature Selection
Uses bit-flip neighbor generation
"""

import numpy as np


class SimulatedAnnealing:
    def __init__(self, fitness_func, n_features, n_iterations=100,
                 initial_temp=100, cooling_rate=0.95, min_temp=0.01):
        """
        Initialize Simulated Annealing
        
        Args:
            fitness_func: Function to evaluate feature subsets (lower is better)
            n_features: Total number of features
            n_iterations: Number of iterations
            initial_temp: Starting temperature
            cooling_rate: Temperature decay rate (0 < rate < 1)
            min_temp: Minimum temperature threshold
        """
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.n_iterations = n_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
    def initialize_solution(self):
        """Initialize random binary solution"""
        solution = np.random.randint(0, 2, self.n_features)
        # Ensure at least one feature is selected
        if np.sum(solution) == 0:
            solution[np.random.randint(self.n_features)] = 1
        return solution
    
    def generate_neighbor(self, solution):
        """Generate neighbor by flipping random bits"""
        neighbor = solution.copy()
        
        # Flip 1-3 random bits
        n_flips = np.random.randint(1, 4)
        flip_indices = np.random.choice(self.n_features, n_flips, replace=False)
        
        for idx in flip_indices:
            neighbor[idx] = 1 - neighbor[idx]
        
        # Ensure at least one feature is selected
        if np.sum(neighbor) == 0:
            neighbor[np.random.randint(self.n_features)] = 1
            
        return neighbor
    
    def acceptance_probability(self, current_fitness, neighbor_fitness, temperature):
        """Calculate probability of accepting worse solution"""
        if neighbor_fitness < current_fitness:
            return 1.0
        else:
            delta = neighbor_fitness - current_fitness
            return np.exp(-delta / temperature)
    
    def optimize(self):
        """Run simulated annealing"""
        # Initialize
        current_solution = self.initialize_solution()
        current_fitness = self.fitness_func(current_solution)
        
        # Track best solution
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        fitness_history = [best_fitness]
        
        temperature = self.initial_temp
        
        # SA main loop
        for iteration in range(self.n_iterations):
            # Generate neighbor
            neighbor_solution = self.generate_neighbor(current_solution)
            neighbor_fitness = self.fitness_func(neighbor_solution)
            
            # Decide whether to accept neighbor
            accept_prob = self.acceptance_probability(
                current_fitness, neighbor_fitness, temperature
            )
            
            if np.random.random() < accept_prob:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                
                # Update best solution if better
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            
            # Cool down temperature
            temperature = max(temperature * self.cooling_rate, self.min_temp)
            
            fitness_history.append(best_fitness)
        
        return best_solution, best_fitness, fitness_history