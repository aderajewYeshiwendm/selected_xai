"""
Main execution script for XAI Feature Selection using Metaheuristics
Compares GA, PSO, SA against SHAP baseline
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time
import json
from scipy.stats import ranksums

from genetic_algorithm import GeneticAlgorithm
from pso import ParticleSwarmOptimization
from simulated_annealing import SimulatedAnnealing
from baseline_shap import SHAPFeatureSelector
from utils import evaluate_feature_subset, create_fitness_function
from visualize import plot_convergence_curves, plot_feature_comparison


def load_dataset(dataset_name='breast_cancer'):
    """Load and preprocess dataset"""
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'digits':
        data = load_digits()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names


def run_experiment(X_train, X_test, y_train, y_test, feature_names, 
                   n_runs=30, max_iterations=100):
    """Run all algorithms and compare results"""
    
    print("=" * 80)
    print("XAI FEATURE SELECTION EXPERIMENT")
    print("=" * 80)
    print(f"\nDataset shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of samples: Train={len(y_train)}, Test={len(y_test)}")
    print(f"Number of runs: {n_runs}")
    print(f"Max iterations per run: {max_iterations}\n")
    
    # Initialize model for fitness evaluation
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create fitness function
    fitness_func = create_fitness_function(X_train, y_train, base_model, alpha=0.01)
    
    n_features = X_train.shape[1]
    
    # Storage for results
    results = {
        'GA': {'fitness_history': [], 'best_solutions': [], 'times': []},
        'PSO': {'fitness_history': [], 'best_solutions': [], 'times': []},
        'SA': {'fitness_history': [], 'best_solutions': [], 'times': []},
        'SHAP': {'fitness_history': [], 'best_solutions': [], 'times': []}
    }
    
    # Run experiments
    for run in range(n_runs):
        print(f"\n{'=' * 80}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'=' * 80}")
        
        # 1. Genetic Algorithm
        print("\n[1/4] Running Genetic Algorithm...")
        start_time = time.time()
        ga = GeneticAlgorithm(
            fitness_func=fitness_func,
            n_features=n_features,
            pop_size=50,
            n_generations=max_iterations,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        best_solution_ga, best_fitness_ga, history_ga = ga.optimize()
        time_ga = time.time() - start_time
        
        results['GA']['fitness_history'].append(history_ga)
        results['GA']['best_solutions'].append(best_solution_ga)
        results['GA']['times'].append(time_ga)
        print(f"   Best fitness: {best_fitness_ga:.4f}, Features: {np.sum(best_solution_ga)}, Time: {time_ga:.2f}s")
        
        # 2. Particle Swarm Optimization
        print("\n[2/4] Running Particle Swarm Optimization...")
        start_time = time.time()
        pso = ParticleSwarmOptimization(
            fitness_func=fitness_func,
            n_features=n_features,
            n_particles=50,
            n_iterations=max_iterations,
            w=0.7,
            c1=1.5,
            c2=1.5
        )
        best_solution_pso, best_fitness_pso, history_pso = pso.optimize()
        time_pso = time.time() - start_time
        
        results['PSO']['fitness_history'].append(history_pso)
        results['PSO']['best_solutions'].append(best_solution_pso)
        results['PSO']['times'].append(time_pso)
        print(f"   Best fitness: {best_fitness_pso:.4f}, Features: {np.sum(best_solution_pso)}, Time: {time_pso:.2f}s")
        
        # 3. Simulated Annealing
        print("\n[3/4] Running Simulated Annealing...")
        start_time = time.time()
        sa = SimulatedAnnealing(
            fitness_func=fitness_func,
            n_features=n_features,
            n_iterations=max_iterations,
            initial_temp=100,
            cooling_rate=0.95
        )
        best_solution_sa, best_fitness_sa, history_sa = sa.optimize()
        time_sa = time.time() - start_time
        
        results['SA']['fitness_history'].append(history_sa)
        results['SA']['best_solutions'].append(best_solution_sa)
        results['SA']['times'].append(time_sa)
        print(f"   Best fitness: {best_fitness_sa:.4f}, Features: {np.sum(best_solution_sa)}, Time: {time_sa:.2f}s")
        
        # 4. SHAP Baseline
        print("\n[4/4] Running SHAP Baseline...")
        start_time = time.time()
        shap_selector = SHAPFeatureSelector(
            model=base_model,
            n_iterations=max_iterations
        )
        best_solution_shap, best_fitness_shap, history_shap = shap_selector.optimize(
            X_train, y_train, fitness_func
        )
        time_shap = time.time() - start_time
        
        results['SHAP']['fitness_history'].append(history_shap)
        results['SHAP']['best_solutions'].append(best_solution_shap)
        results['SHAP']['times'].append(time_shap)
        print(f"   Best fitness: {best_fitness_shap:.4f}, Features: {np.sum(best_solution_shap)}, Time: {time_shap:.2f}s")
    
    return results


def analyze_results(results, X_test, y_test, feature_names):
    """Analyze and report results"""
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Calculate statistics
    stats = {}
    for method in results.keys():
        fitness_vals = [hist[-1] for hist in results[method]['fitness_history']]
        n_features = [np.sum(sol) for sol in results[method]['best_solutions']]
        times = results[method]['times']
        
        stats[method] = {
            'mean_fitness': np.mean(fitness_vals),
            'std_fitness': np.std(fitness_vals),
            'mean_features': np.mean(n_features),
            'std_features': np.std(n_features),
            'mean_time': np.mean(times),
            'best_fitness': np.min(fitness_vals),
            'fitness_values': fitness_vals
        }
    
    # Print summary table
    print("\nSummary Statistics (30 runs):")
    print("-" * 80)
    print(f"{'Method':<10} {'Mean Fitness':<15} {'Best Fitness':<15} {'Avg Features':<15} {'Avg Time (s)'}")
    print("-" * 80)
    
    for method in ['GA', 'PSO', 'SA', 'SHAP']:
        s = stats[method]
        print(f"{method:<10} {s['mean_fitness']:.4f}±{s['std_fitness']:.4f}  "
              f"{s['best_fitness']:.4f}        "
              f"{s['mean_features']:.1f}±{s['std_features']:.1f}    "
              f"{s['mean_time']:.2f}")
    
    print("-" * 80)
    
    # Wilcoxon rank-sum tests
    print("\nWilcoxon Rank-Sum Tests (vs SHAP baseline):")
    print("-" * 80)
    shap_fitness = stats['SHAP']['fitness_values']
    
    for method in ['GA', 'PSO', 'SA']:
        method_fitness = stats[method]['fitness_values']
        statistic, p_value = ranksums(method_fitness, shap_fitness)
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        comparison = "better" if np.mean(method_fitness) < np.mean(shap_fitness) else "worse"
        
        print(f"{method} vs SHAP: p-value = {p_value:.4f} {significance} ({comparison})")
    
    print("-" * 80)
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    
    return stats


def main():
    """Main execution function"""
    
    # Configuration
    DATASET = 'breast_cancer'  # Options: 'breast_cancer', 'wine', 'digits'
    N_RUNS = 30
    MAX_ITERATIONS = 100
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load dataset
    print("Loading dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_dataset(DATASET)
    
    # Run experiments
    results = run_experiment(
        X_train, X_test, y_train, y_test, feature_names,
        n_runs=N_RUNS,
        max_iterations=MAX_ITERATIONS
    )
    
    # Analyze results
    stats = analyze_results(results, X_test, y_test, feature_names)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_convergence_curves(results, save_path='convergence_curves.png')
    plot_feature_comparison(stats, save_path='feature_comparison.png')
    
    # Save results
    print("\nSaving results to JSON...")
    output_data = {
        'dataset': DATASET,
        'n_runs': N_RUNS,
        'max_iterations': MAX_ITERATIONS,
        'statistics': {
            method: {k: v for k, v in data.items() if k != 'fitness_values'}
            for method, data in stats.items()
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - convergence_curves.png")
    print("  - feature_comparison.png")
    print("  - results.json")
    print("\nThank you for using the XAI Feature Selection framework!")


if __name__ == "__main__":
    main()