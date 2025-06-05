import click
import numpy as np
import json
import os
from pathlib import Path

from .utils import load_data, binary_to_features, load_pipeline_config
from .selector import FeatureSelector
from .visualization import plot_all_metrics

@click.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(exists=True),
    required=True,
    help="Ruta al archivo CSV de entrada",
)
@click.option(
    "--target",
    "-t",
    "target_col",
    required=True,
    help="Nombre de la columna objetivo",
)
@click.option(
    "--config",
    "-c",
    "config_file",
    type=click.Path(exists=True),
    required=True,
    help="Ruta al archivo YAML de configuración del pipeline",
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Directorio donde guardar los resultados",
)
@click.option(
    "--pop-size",
    default=50,
    show_default=True,
    type=int,
    help="Tamaño de la población",
)
@click.option(
    "--generations",
    default=20,
    show_default=True,
    type=int,
    help="Número de generaciones",
)
@click.option(
    "--cv",
    default=5,
    show_default=True,
    type=int,
    help="Número de folds para cross-validation",
)
@click.option(
    "--crossover-rate",
    default=0.8,
    show_default=True,
    type=float,
    help="Tasa de cruce",
)
@click.option(
    "--mutation-rate",
    default=0.01,
    show_default=True,
    type=float,
    help="Tasa de mutación",
)
@click.option(
    "--elite-size",
    default=2,
    show_default=True,
    type=int,
    help="Número de mejores individuos a mantener en cada generación",
)
def main(input_file, target_col, config_file, output_dir, pop_size, generations, cv, crossover_rate, mutation_rate, elite_size):
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data and pipeline configuration
    X, y, feature_names = load_data(input_file, target_col)
    pipeline = load_pipeline_config(config_file)
    
    # Initialize and run feature selector
    fs = FeatureSelector(
        population_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        cv=cv,
        estimator=pipeline,
        elite_size=elite_size,
    )
    selected_indices, best_score = fs.fit(X, y, feature_names=feature_names)
    
    # Get selected features
    selected_features = binary_to_features(
        [1 if i in selected_indices else 0 for i in range(len(feature_names))],
        feature_names,
    )
    
    # Save results
    results = {
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'best_score': best_score,
        'parameters': {
            'population_size': pop_size,
            'generations': generations,
            'cv': cv,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'elite_size': elite_size
        }
    }
    
    # Save results to JSON
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save history to JSON
    with open(output_path / 'history.json', 'w') as f:
        json.dump(fs.history, f, indent=2)
    
    # Generate and save plots
    plot_all_metrics(
        fs.history,
        feature_names=feature_names,
        save_path=str(output_path / 'evolution_plots.png')
    )
    
    # Print results
    click.echo(f"Best features: {selected_features}")
    click.echo(f"Best score: {best_score:.4f}")
    click.echo(f"\nResults saved in: {output_path}")

if __name__ == "__main__":
    main()
