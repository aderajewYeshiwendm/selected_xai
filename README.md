# XAI Feature Selection: Metaheuristics vs SHAP

## Project Overview

This project benchmarks nature-inspired metaheuristic algorithms (Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing) against SHAP baseline for Explainable AI feature selection.

**Course**: Benchmarking Nature-Inspired Metaheuristics against State-of-the-Art Baselines in Modern AI  
**Institution**: School of Information Technology and Engineering, Addis Ababa University  
**Topic**: #12 - Explainable AI (XAI): Minimal sufficient feature subsets for SHAP

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Understanding the Results](#understanding-the-results)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)

---

## Requirements

### Software Requirements
- Python 3.8 or higher
- pip package manager
- (Optional) LaTeX distribution for compiling the paper

### Hardware Requirements
- Minimum: 4GB RAM, 2 CPU cores
- Recommended: 8GB+ RAM, 4+ CPU cores
- Storage: ~1GB for dataset cache and results

---

## Installation

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <your-repository-url>
cd xai-feature-selection

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
shap>=0.42.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
```

### Step 4: Verify Installation

```bash
python -c "import sklearn, shap, numpy, pandas; print('All packages installed successfully!')"
```

---

## Project Structure

```
xai-feature-selection/
│
├── main.py                      # Main execution script
├── genetic_algorithm.py         # GA implementation
├── pso.py                       # PSO implementation
├── simulated_annealing.py       # SA implementation
├── baseline_shap.py             # SHAP baseline implementation
├── utils.py                     # Utility functions
├── visualize.py                 # Visualization functions
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── paper/
│   └── paper.tex               # LaTeX source for research paper
│
└── results/                     # Generated after running experiments
    ├── convergence_curves.png
    ├── feature_comparison.png
    └── results.json
```

---

## Quick Start

### Run Complete Experiment (30 runs)

```bash
python main.py
```

This will:
1. Load the Breast Cancer dataset
2. Run GA, PSO, SA, and SHAP for 30 independent runs
3. Perform statistical analysis (Wilcoxon tests)
4. Generate visualizations
5. Save results to JSON

**Expected runtime**: 15-20 minutes on standard hardware

### Output Files

After completion, you'll find:
- `convergence_curves.png` - Fitness evolution over iterations
- `feature_comparison.png` - Performance comparison charts
- `results.json` - Detailed numerical results

---

## Detailed Usage

### Running with Different Datasets

Edit `main.py` and change the `DATASET` variable:

```python
# In main.py, line ~130
DATASET = 'wine'  # Options: 'breast_cancer', 'wine', 'digits'
```

### Adjusting Number of Runs

```python
# In main.py, line ~131
N_RUNS = 10  # Reduce for faster testing (minimum recommended: 10)
```

### Modifying Algorithm Parameters

**Genetic Algorithm** (in `genetic_algorithm.py`):
```python
ga = GeneticAlgorithm(
    pop_size=50,          # Population size
    n_generations=100,    # Number of generations
    crossover_rate=0.8,   # Crossover probability
    mutation_rate=0.1     # Mutation probability
)
```

**PSO** (in `pso.py`):
```python
pso = ParticleSwarmOptimization(
    n_particles=50,       # Swarm size
    n_iterations=100,     # Iterations
    w=0.7,               # Inertia weight
    c1=1.5,              # Cognitive coefficient
    c2=1.5               # Social coefficient
)
```

**Simulated Annealing** (in `simulated_annealing.py`):
```python
sa = SimulatedAnnealing(
    n_iterations=100,     # Number of iterations
    initial_temp=100,     # Starting temperature
    cooling_rate=0.95     # Temperature decay
)
```

### Running Individual Algorithms

You can test individual algorithms by modifying the main loop:

```python
# In main.py, comment out unwanted algorithms
# Example: Run only GA and SHAP
for run in range(n_runs):
    # ... GA code ...
    # Comment out PSO and SA sections
    # ... SHAP code ...
```

---

## Understanding the Results

### Console Output

The program prints detailed progress:

```
================================================================================
RUN 1/30
================================================================================

[1/4] Running Genetic Algorithm...
   Best fitness: 0.0823, Features: 12, Time: 18.4s

[2/4] Running Particle Swarm Optimization...
   Best fitness: 0.0891, Features: 14, Time: 21.7s

...

================================================================================
STATISTICAL ANALYSIS
================================================================================

Summary Statistics (30 runs):
--------------------------------------------------------------------------------
Method     Mean Fitness    Best Fitness    Avg Features    Avg Time (s)
--------------------------------------------------------------------------------
GA         0.0823±0.0124  0.0687          12.3±2.1        18.4
PSO        0.0891±0.0156  0.0712          13.8±2.8        21.7
SA         0.0947±0.0183  0.0734          15.1±3.2        15.2
SHAP       0.0841±0.0089  0.0728          14.7±1.4        12.8
--------------------------------------------------------------------------------

Wilcoxon Rank-Sum Tests (vs SHAP baseline):
--------------------------------------------------------------------------------
GA vs SHAP: p-value = 0.0234 * (better)
PSO vs SHAP: p-value = 0.1876 ns (worse)
SA vs SHAP: p-value = 0.0089 ** (worse)
--------------------------------------------------------------------------------

Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant
```

### Interpreting Metrics

**Fitness Value**: Lower is better
- Fitness = (1 - accuracy) + 0.01 × (features selected / total features)
- Balances model accuracy with feature count

**Statistical Significance**:
- `***` p < 0.001: Highly significant
- `**` p < 0.01: Very significant  
- `*` p < 0.05: Significant
- `ns`: Not significant

**Best Performer**: Algorithm with lowest mean fitness and statistical significance

### Visualization Interpretation

**Convergence Curves** (`convergence_curves.png`):
- Shows how fitness improves over iterations
- Shaded area represents standard deviation
- Steeper curves = faster convergence
- Flatter lines = consistent performance

**Feature Comparison** (`feature_comparison.png`):
- Box plots show fitness distribution
- Bar charts compare average performance
- Feature count indicates model complexity
- Computation time shows efficiency

---

## Customization

### Using Your Own Dataset

1. Prepare your dataset as NumPy arrays or CSV
2. Modify the `load_dataset()` function in `main.py`:

```python
def load_custom_dataset(filepath):
    """Load custom dataset"""
    import pandas as pd
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Separate features and labels
    X = df.drop('target', axis=1).values
    y = df['target'].values
    feature_names = df.drop('target', axis=1).columns.tolist()
    
    # Split and scale
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names
```

### Changing the Base Model

To use a different classifier, modify `main.py`:

```python
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Replace Random Forest with SVM
base_model = SVC(kernel='rbf', random_state=42)

# Or use Neural Network
base_model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
```

### Adjusting Fitness Function

Modify the `alpha` parameter in `utils.py`:

```python
# More emphasis on feature reduction
fitness_func = create_fitness_function(X_train, y_train, base_model, alpha=0.05)

# More emphasis on accuracy
fitness_func = create_fitness_function(X_train, y_train, base_model, alpha=0.001)
```

---

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'shap'**
```bash
# Solution: Install SHAP
pip install shap
```

**2. Memory Error during execution**
```python
# Solution: Reduce population/swarm size
pop_size = 20  # Instead of 50
n_runs = 10    # Instead of 30
```

**3. SHAP computation is very slow**
```python
# Solution: Use TreeExplainer (only for tree-based models)
# This is already optimized in baseline_shap.py
# Or reduce dataset size for SHAP computation
```

**4. Convergence curves look strange**
```bash
# Check if all runs completed successfully
# Verify fitness values are in reasonable range (0-1)
# Ensure at least one feature is always selected
```

**5. Statistical tests show no significance**
```python
# Increase number of runs for more statistical power
N_RUNS = 50  # More runs = better statistics
```

### Performance Optimization

**Speed up experiments**:
1. Reduce cross-validation folds: `cv_folds=3` (instead of 5)
2. Decrease max iterations: `max_iterations=50`
3. Use smaller datasets for testing
4. Parallelize runs (requires code modification)

**Improve solution quality**:
1. Increase population/swarm size: `pop_size=100`
2. More iterations: `max_iterations=200`
3. Fine-tune algorithm parameters
4. Run more independent trials: `N_RUNS=50`

---

## Compiling the Research Paper

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- LNCS document class (included in most distributions)

### Compilation Steps

```bash
cd paper/
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or use your favorite LaTeX editor (Overleaf, TeXstudio, etc.)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourname2025xai,
  title={Benchmarking Nature-Inspired Metaheuristics Against SHAP for Explainable AI Feature Selection},
  author={Your Name and Team Members},
  booktitle={Course Project Report},
  year={2025},
  institution={Addis Ababa University}
}
```

---

## License

This project is created for educational purposes as part of a university course project.

---

## Contact

For questions or issues:
- Email: your.email@example.edu.et
- Course: Benchmarking Nature-Inspired Metaheuristics
- Institution: Addis Ababa University

---

## Acknowledgments

- SHAP library authors for the baseline implementation
- Scikit-learn community for machine learning tools
- Course instructors for project guidance

---

## Additional Resources

**Learn More**:
- SHAP documentation: https://shap.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/
- Metaheuristic algorithms: See references in the paper

**Datasets**:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
- Scikit-learn datasets: https://scikit-learn.org/stable/datasets.html

---

