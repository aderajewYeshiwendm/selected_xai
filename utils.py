"""
Utility functions for XAI feature selection
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone


def create_fitness_function(X, y, model, alpha=0.01, cv_folds=5):
    """
    Create fitness function for feature selection
    
    Fitness = (1 - accuracy) + alpha * (n_selected / n_total)
    
    Args:
        X: Training features
        y: Training labels
        model: Scikit-learn model
        alpha: Weight for feature count penalty
        cv_folds: Number of cross-validation folds
        
    Returns:
        fitness_func: Function that takes binary feature mask and returns fitness
    """
    n_total = X.shape[1]
    
    def fitness_func(feature_mask):
        """
        Evaluate fitness for a feature subset
        
        Args:
            feature_mask: Binary array indicating selected features
            
        Returns:
            fitness: Fitness value (lower is better)
        """
        # Count selected features
        n_selected = np.sum(feature_mask)
        
        # If no features selected, return worst fitness
        if n_selected == 0:
            return 1.0
        
        # Select features
        selected_indices = np.where(feature_mask == 1)[0]
        X_selected = X[:, selected_indices]
        
        # Train and evaluate model with cross-validation
        try:
            model_copy = clone(model)
            scores = cross_val_score(model_copy, X_selected, y, 
                                    cv=cv_folds, scoring='accuracy')
            accuracy = np.mean(scores)
        except Exception:
            # If model fails, return worst fitness
            return 1.0
        
        # Calculate fitness (minimize)
        # Lower accuracy -> higher fitness
        # More features -> higher fitness
        fitness = (1.0 - accuracy) + alpha * (n_selected / n_total)
        
        return fitness
    
    return fitness_func


def evaluate_feature_subset(X_train, X_test, y_train, y_test, 
                           feature_mask, model):
    """
    Evaluate a feature subset on test data
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        feature_mask: Binary array indicating selected features
        model: Scikit-learn model
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Select features
    selected_indices = np.where(feature_mask == 1)[0]
    
    if len(selected_indices) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'n_features': 0
        }
    
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    # Train model
    model_copy = clone(model)
    model_copy.fit(X_train_selected, y_train)
    
    # Predict
    y_pred = model_copy.predict(X_test_selected)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'n_features': len(selected_indices)
    }
    
    return metrics


def calculate_feature_statistics(solutions, feature_names):
    """
    Calculate statistics about feature selection across multiple runs
    
    Args:
        solutions: List of binary feature masks
        feature_names: List of feature names
        
    Returns:
        stats: Dictionary with feature selection frequencies
    """
    n_features = len(feature_names)
    n_runs = len(solutions)
    
    # Count how many times each feature was selected
    selection_counts = np.zeros(n_features)
    for solution in solutions:
        selection_counts += solution
    
    # Calculate selection frequency
    selection_freq = selection_counts / n_runs
    
    # Create statistics dictionary
    stats = {
        'feature_names': feature_names,
        'selection_frequency': selection_freq,
        'selection_counts': selection_counts,
        'n_runs': n_runs
    }
    
    # Sort by frequency
    sorted_indices = np.argsort(selection_freq)[::-1]
    stats['sorted_features'] = [(feature_names[i], selection_freq[i]) 
                                for i in sorted_indices]
    
    return stats


def compare_feature_subsets(solution1, solution2, feature_names=None):
    """
    Compare two feature subsets
    
    Args:
        solution1: First binary feature mask
        solution2: Second binary feature mask
        feature_names: Optional list of feature names
        
    Returns:
        comparison: Dictionary with comparison metrics
    """
    # Jaccard similarity
    intersection = np.sum(solution1 & solution2)
    union = np.sum(solution1 | solution2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Features in common, unique to each
    common_features = np.where((solution1 == 1) & (solution2 == 1))[0]
    unique_to_1 = np.where((solution1 == 1) & (solution2 == 0))[0]
    unique_to_2 = np.where((solution1 == 0) & (solution2 == 1))[0]
    
    comparison = {
        'jaccard_similarity': jaccard,
        'n_common': len(common_features),
        'n_unique_to_1': len(unique_to_1),
        'n_unique_to_2': len(unique_to_2),
        'common_indices': common_features,
        'unique_to_1_indices': unique_to_1,
        'unique_to_2_indices': unique_to_2
    }
    
    if feature_names is not None:
        comparison['common_features'] = [feature_names[i] for i in common_features]
        comparison['unique_to_1_features'] = [feature_names[i] for i in unique_to_1]
        comparison['unique_to_2_features'] = [feature_names[i] for i in unique_to_2]
    
    return comparison