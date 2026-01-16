"""
Genetic Algorithm for Feature Selection
Binary encoding: 1 = feature selected, 0 = feature not selected
"""

import numpy as np


class GeneticAlgorithm:
    def __init__(self, fitness_func, n_features, pop_size=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.1, tournament_size=3):
        """
        Initialize Genetic Algorithm
        
        Args:
            fitness_func: Function to evaluate feature subsets (lower is better)
            n_features: Total number of features
            pop_size: Population size
            n_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of bit flip mutation
            tournament_size: Size of tournament for selection
        """
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
    def initialize_population(self):
        """Initialize random binary population"""
        population = []
        for _ in range(self.pop_size):
            # Ensure at least one feature is selected
            individual = np.random.randint(0, 2, self.n_features)
            if np.sum(individual) == 0:
                individual[np.random.randint(self.n_features)] = 1
            population.append(individual)
        return np.array(population)
    
    def evaluate_population(self, population):
        """Evaluate fitness for entire population"""
        fitness_values = []
        for individual in population:
            if np.sum(individual) == 0:
                fitness_values.append(1.0)  # Worst fitness
            else:
                fitness_values.append(self.fitness_func(individual))
        return np.array(fitness_values)
    
    def tournament_selection(self, population, fitness_values):
        """Select parent using tournament selection"""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness_values[indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.random() < self.crossover_rate:
            point = np.random.randint(1, self.n_features)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            
            # Ensure at least one feature selected
            if np.sum(child1) == 0:
                child1[np.random.randint(self.n_features)] = 1
            if np.sum(child2) == 0:
                child2[np.random.randint(self.n_features)] = 1
                
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Bit-flip mutation"""
        mutated = individual.copy()
        for i in range(self.n_features):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # Ensure at least one feature selected
        if np.sum(mutated) == 0:
            mutated[np.random.randint(self.n_features)] = 1
            
        return mutated
    
    def optimize(self):
        """Run the genetic algorithm"""
        # Initialize
        population = self.initialize_population()
        fitness_values = self.evaluate_population(population)
        
        # Track best solution
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        fitness_history = [best_fitness]
        
        # Evolution loop
        for generation in range(self.n_generations):
            new_population = []
            
            # Elitism: keep best solution
            new_population.append(best_solution.copy())
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # Update population
            population = np.array(new_population[:self.pop_size])
            fitness_values = self.evaluate_population(population)
            
            # Update best solution
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_solution = population[current_best_idx].copy()
            
            fitness_history.append(best_fitness)
        
        return best_solution, best_fitness, fitness_history