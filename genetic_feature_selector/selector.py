from typing import List, Tuple
import numpy as np

from .ga import GeneticAlgorithm
from .fitness import evaluate_fitness

class FeatureSelector:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        cv: int = 5,
        estimator=None,
        scoring: str = "accuracy",
        elite_size: int = 2,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.cv = cv
        self.estimator = estimator
        self.scoring = scoring
        self.elite_size = elite_size
        self.history = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> Tuple[List[int], float]:
        n_features = X.shape[1]

        def fitness_wrapper(genome):
            return evaluate_fitness(
                genome,
                X,
                y,
                estimator=self.estimator,
                cv=self.cv,
                scoring=self.scoring,
            )

        ga = GeneticAlgorithm(
            genome_length=n_features,
            population_size=self.population_size,
            generations=self.generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            fitness_func=fitness_wrapper,
            elite_size=self.elite_size,
        )

        best_genome, best_score, history = ga.run()
        self.history = history

        selected_indices = [i for i, gene in enumerate(best_genome) if gene == 1]
        return selected_indices, best_score
