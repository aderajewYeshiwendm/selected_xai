"""
SHAP-based Feature Selection Baseline
Uses SHAP values to rank and iteratively select features
"""

import numpy as np
import shap
from sklearn.base import clone


class SHAPFeatureSelector:
    def __init__(self, model, n_iterations=100):
        """
        Initialize SHAP-based feature selector
        
        Args:
            model: Scikit-learn compatible model
            n_iterations: Number of iterations for progressive feature addition
        """
        self.model = model
        self.n_iterations = n_iterations
        
    def compute_shap_values(self, X, y):
        """Compute SHAP values for feature importance"""
        # Train model
        model = clone(self.model)
        model.fit(X, y)
        
        # Create SHAP explainer
        # Use TreeExplainer for tree-based models
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # For multi-class, average across classes
            if isinstance(shap_values, list):
                shap_values = np.abs(shap_values).mean(axis=0)
            else:
                shap_values = np.abs(shap_values)
                
        except Exception:
            # Fallback to KernelExplainer for other models
            background = shap.sample(X, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X[:100])
            
            if isinstance(shap_values, list):
                shap_values = np.abs(shap_values).mean(axis=0)
            else:
                shap_values = np.abs(shap_values)
        
        # Average SHAP values across samples for each feature
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        return feature_importance
    
    def optimize(self, X, y, fitness_func):
        """
        Progressive feature selection based on SHAP importance
        
        Args:
            X: Training features
            y: Training labels
            fitness_func: Fitness function for evaluation
            
        Returns:
            best_solution: Binary feature mask
            best_fitness: Best fitness value
            fitness_history: History of fitness values
        """
        n_features = X.shape[1]
        
        # Compute SHAP importance
        feature_importance = self.compute_shap_values(X, y)
        
        # Rank features by importance
        feature_ranking = np.argsort(feature_importance)[::-1]
        
        # Initialize with no features
        current_solution = np.zeros(n_features, dtype=int)
        best_solution = None
        best_fitness = float('inf')
        fitness_history = []
        
        # Progressive feature addition
        n_steps = min(self.n_iterations, n_features)
        features_per_step = max(1, n_features // n_steps)
        
        for iteration in range(n_steps):
            # Add top features progressively
            n_features_to_add = min((iteration + 1) * features_per_step, n_features)
            current_solution = np.zeros(n_features, dtype=int)
            current_solution[feature_ranking[:n_features_to_add]] = 1
            
            # Evaluate fitness
            if np.sum(current_solution) > 0:
                current_fitness = fitness_func(current_solution)
                
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_solution = current_solution.copy()
                
                fitness_history.append(best_fitness)
            else:
                fitness_history.append(best_fitness if best_fitness != float('inf') else 1.0)
        
        # Also try greedy backward elimination from all features
        current_solution = np.ones(n_features, dtype=int)
        current_fitness = fitness_func(current_solution)
        
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution.copy()
        
        # Remove features one by one (least important first)
        for i in range(len(feature_ranking) - 1, -1, -1):
            if len(fitness_history) >= self.n_iterations:
                break
                
            feature_to_remove = feature_ranking[i]
            if current_solution[feature_to_remove] == 1:
                current_solution[feature_to_remove] = 0
                
                if np.sum(current_solution) > 0:
                    current_fitness = fitness_func(current_solution)
                    
                    if current_fitness < best_fitness:
                        best_fitness = current_fitness
                        best_solution = current_solution.copy()
                    else:
                        # Restore feature if removal didn't improve
                        current_solution[feature_to_remove] = 1
                
                fitness_history.append(best_fitness)
        
        # Pad fitness history to match n_iterations
        while len(fitness_history) < self.n_iterations + 1:
            fitness_history.append(best_fitness)
        
        return best_solution, best_fitness, fitness_history[:self.n_iterations + 1]