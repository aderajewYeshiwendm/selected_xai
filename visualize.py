"""
Visualization functions for experiment results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_convergence_curves(results, save_path='convergence_curves.png'):
    """
    Plot convergence curves for all algorithms
    
    Args:
        results: Dictionary with fitness histories for each method
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 7))
    
    colors = {
        'GA': '#2ecc71',
        'PSO': '#9b59b6',
        'SA': '#e67e22',
        'SHAP': '#3498db'
    }
    
    for method in ['GA', 'PSO', 'SA', 'SHAP']:
        histories = results[method]['fitness_history']
        
        # Calculate mean and std across runs
        max_len = max(len(h) for h in histories)
        padded_histories = []
        
        for h in histories:
            if len(h) < max_len:
                # Pad with last value
                padded = list(h) + [h[-1]] * (max_len - len(h))
            else:
                padded = h
            padded_histories.append(padded)
        
        histories_array = np.array(padded_histories)
        mean_history = np.mean(histories_array, axis=0)
        std_history = np.std(histories_array, axis=0)
        
        iterations = np.arange(len(mean_history))
        
        # Plot mean line
        plt.plot(iterations, mean_history, label=method, 
                color=colors[method], linewidth=2.5)
        
        # Plot confidence interval
        plt.fill_between(iterations,
                        mean_history - std_history,
                        mean_history + std_history,
                        color=colors[method], alpha=0.2)
    
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Fitness Value (lower is better)', fontsize=14, fontweight='bold')
    plt.title('Convergence Curves: Metaheuristics vs SHAP Baseline', 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence curves to {save_path}")
    plt.close()


def plot_feature_comparison(stats, save_path='feature_comparison.png'):
    """
    Plot comparison of methods
    
    Args:
        stats: Dictionary with statistics for each method
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = ['GA', 'PSO', 'SA', 'SHAP']
    colors = ['#2ecc71', '#9b59b6', '#e67e22', '#3498db']
    
    # 1. Box plot of fitness values
    ax1 = axes[0, 0]
    fitness_data = [stats[m]['fitness_values'] for m in methods]
    bp = ax1.boxplot(fitness_data, labels=methods, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('Fitness Value', fontsize=12, fontweight='bold')
    ax1.set_title('Fitness Distribution (30 runs)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Bar plot of mean fitness
    ax2 = axes[0, 1]
    mean_fitness = [stats[m]['mean_fitness'] for m in methods]
    std_fitness = [stats[m]['std_fitness'] for m in methods]
    
    bars = ax2.bar(methods, mean_fitness, yerr=std_fitness, 
                   color=colors, alpha=0.7, capsize=5)
    ax2.set_ylabel('Mean Fitness ± Std', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Fitness Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, mean_fitness):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Bar plot of feature counts
    ax3 = axes[1, 0]
    mean_features = [stats[m]['mean_features'] for m in methods]
    std_features = [stats[m]['std_features'] for m in methods]
    
    bars = ax3.bar(methods, mean_features, yerr=std_features,
                   color=colors, alpha=0.7, capsize=5)
    ax3.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax3.set_title('Average Selected Features', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mean_features):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Bar plot of computation time
    ax4 = axes[1, 1]
    mean_times = [stats[m]['mean_time'] for m in methods]
    
    bars = ax4.bar(methods, mean_times, color=colors, alpha=0.7)
    ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('Average Computation Time', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mean_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature comparison to {save_path}")
    plt.close()


def plot_feature_selection_frequency(feature_stats, top_n=15, 
                                     save_path='feature_frequency.png'):
    """
    Plot feature selection frequency across runs
    
    Args:
        feature_stats: Statistics from calculate_feature_statistics
        top_n: Number of top features to show
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    sorted_features = feature_stats['sorted_features'][:top_n]
    feature_names = [f[0] for f in sorted_features]
    frequencies = [f[1] for f in sorted_features]
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(frequencies)))
    
    bars = plt.barh(range(len(feature_names)), frequencies, color=colors)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Selection Frequency', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Frequently Selected Features', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{freq:.2f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature frequency plot to {save_path}")
    plt.close()