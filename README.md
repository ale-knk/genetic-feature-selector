# Genetic Feature Selector

A powerful and flexible genetic algorithm-based feature selection tool for machine learning pipelines. This tool helps identify the most relevant features for your machine learning models while optimizing for model performance.

## Features

- 🧬 Genetic Algorithm-based feature selection
- 📊 Multiple initialization strategies for better exploration
- 🏆 Elitism to preserve best solutions
- 📈 Progress tracking with tqdm
- 📊 Visualization of algorithm evolution
- 🔄 Cross-validation support
- 🛠️ Customizable ML pipeline configuration
- 📝 Detailed results and history logging

## Installation

```bash
# Clone the repository
git clone https://github.com/ale-knk/genetic-feature-selector.git
cd genetic-feature-selector

# Install the package
pip install -e .
```

## Quick Start

The project comes with example data and is ready to use right after cloning. To run with the example data:

```bash
./run.sh
```

This will:
1. Use the example dataset in `data/`
2. Use the default pipeline configuration
3. Save results in a timestamped directory under `results/`

To use your own data:

1. Prepare your data in CSV format
2. Create a pipeline configuration file (YAML)
3. Run the feature selector:

```bash
./run.sh --input data/your_data.csv --target target_column
```

## Configuration

The tool can be configured through command-line arguments or the run script:

```bash
./run.sh \
    --input data/data.csv \
    --target target_column \
    --config pipeline_config.yaml \
    --output-dir results \
    --pop-size 50 \
    --generations 50 \
    --cv 5 \
    --crossover-rate 0.6 \
    --mutation-rate 0.05 \
    --elite-size 2
```

### Parameters

- `--input`: Input CSV file path
- `--target`: Target column name
- `--config`: Pipeline configuration file (YAML)
- `--output-dir`: Output directory for results
- `--pop-size`: Population size (default: 50)
- `--generations`: Number of generations (default: 50)
- `--cv`: Cross-validation folds (default: 5)
- `--crossover-rate`: Crossover probability (default: 0.6)
- `--mutation-rate`: Mutation probability (default: 0.05)
- `--elite-size`: Number of elite individuals to preserve (default: 2)

## ML Pipeline Configuration

Create a YAML file to define your machine learning pipeline. Example:

```yaml
steps:
  - name: StandardScaler
  - name: RandomForestClassifier
    params:
      n_estimators: 100
      random_state: 42
```

## Output

The tool generates several output files in the specified directory:

- `results.json`: Selected features and performance metrics
- `history.json`: Evolution history of the genetic algorithm
- `evolution_plots.png`: Visualization of the algorithm's progress

## Project Structure

```
genetic-feature-selector/
├── genetic_feature_selector/
│   ├── __init__.py
│   ├── cli.py           # Command-line interface
│   ├── fitness.py       # Fitness evaluation functions
│   ├── ga.py           # Genetic algorithm implementation
│   ├── selector.py     # Feature selector class
│   ├── utils.py        # Utility functions
│   └── visualization.py # Plotting functions
├── data/               # Dataset directory
├── run.sh             # Main run script
├── pipeline_config.yaml # Pipeline configuration
├── config/            # Additional configurations
├── tests/             # Test suite
└── results/           # Output directory
```

## How It Works

1. **Initialization**: Creates a diverse initial population using multiple strategies:
   - All features selected
   - Single feature individuals
   - Fixed number of features
   - Random individuals

2. **Evolution**:
   - Evaluates fitness using cross-validation
   - Preserves elite individuals
   - Performs crossover and mutation
   - Tracks progress and history

3. **Selection**:
   - Returns the best feature subset found
   - Provides detailed performance metrics
   - Generates visualization of the process

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
