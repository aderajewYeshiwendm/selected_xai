"""
Particle Swarm Optimization for Feature Selection
Using sigmoid transfer function for binary conversion
"""

import numpy as np


class ParticleSwarmOptimization:
    def __init__(self, fitness_func, n_features, n_particles=50, n_iterations=100,
                 w=0.7, c1=1.5, c2=1.5, v_max=6.0):
        """
        Initialize Particle Swarm Optimization
        
        Args:
            fitness_func: Function to evaluate feature subsets (lower is better)
            n_features: Total number of features
            n_particles: Number of particles in swarm
            n_iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            v_max: Maximum velocity
        """
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        
    def sigmoid(self, x):
        """Sigmoid transfer function"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def initialize_swarm(self):
        """Initialize particles with random positions and velocities"""
        # Initialize positions (continuous)
        positions = np.random.uniform(-4, 4, (self.n_particles, self.n_features))
        
        # Initialize velocities
        velocities = np.random.uniform(-self.v_max, self.v_max, 
                                      (self.n_particles, self.n_features))
        
        # Convert to binary using sigmoid
        binary_positions = (self.sigmoid(positions) > 0.5).astype(int)
        
        # Ensure at least one feature per particle
        for i in range(self.n_particles):
            if np.sum(binary_positions[i]) == 0:
                binary_positions[i][np.random.randint(self.n_features)] = 1
        
        return positions, velocities, binary_positions
    
    def evaluate_swarm(self, binary_positions):
        """Evaluate fitness for all particles"""
        fitness_values = []
        for position in binary_positions:
            if np.sum(position) == 0:
                fitness_values.append(1.0)
            else:
                fitness_values.append(self.fitness_func(position))
        return np.array(fitness_values)
    
    def optimize(self):
        """Run particle swarm optimization"""
        # Initialize swarm
        positions, velocities, binary_positions = self.initialize_swarm()
        fitness_values = self.evaluate_swarm(binary_positions)
        
        # Initialize personal best
        pbest_positions = positions.copy()
        pbest_binary = binary_positions.copy()
        pbest_fitness = fitness_values.copy()
        
        # Initialize global best
        gbest_idx = np.argmin(fitness_values)
        gbest_position = positions[gbest_idx].copy()
        gbest_binary = binary_positions[gbest_idx].copy()
        gbest_fitness = fitness_values[gbest_idx]
        
        fitness_history = [gbest_fitness]
        
        # PSO main loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1 = np.random.random(self.n_features)
                r2 = np.random.random(self.n_features)
                
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Limit velocity
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Convert to binary
                binary_positions[i] = (self.sigmoid(positions[i]) > 0.5).astype(int)
                
                # Ensure at least one feature
                if np.sum(binary_positions[i]) == 0:
                    binary_positions[i][np.random.randint(self.n_features)] = 1
                
                # Evaluate fitness
                fitness_values[i] = self.fitness_func(binary_positions[i])
                
                # Update personal best
                if fitness_values[i] < pbest_fitness[i]:
                    pbest_fitness[i] = fitness_values[i]
                    pbest_positions[i] = positions[i].copy()
                    pbest_binary[i] = binary_positions[i].copy()
                
                # Update global best
                if fitness_values[i] < gbest_fitness:
                    gbest_fitness = fitness_values[i]
                    gbest_position = positions[i].copy()
                    gbest_binary = binary_positions[i].copy()
            
            fitness_history.append(gbest_fitness)
            
            # Adaptive inertia weight (linearly decreasing)
            self.w = 0.9 - (0.9 - 0.4) * (iteration / self.n_iterations)
        
        return gbest_binary, gbest_fitness, fitness_history