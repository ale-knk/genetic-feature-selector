import random
import numpy as np
from tqdm import tqdm

class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None

    def evaluate(self, fitness_func):
        self.fitness = fitness_func(self.genome)
        return self.fitness

    @staticmethod
    def crossover(parent1, parent2):
        point = random.randint(1, len(parent1.genome) - 1)
        child1_genome = parent1.genome[:point] + parent2.genome[point:]
        child2_genome = parent2.genome[:point] + parent1.genome[point:]
        return Individual(child1_genome), Individual(child2_genome)

    def mutate(self, mutation_rate):
        new_genome = [
            gene if random.random() > mutation_rate else 1 - gene
            for gene in self.genome
        ]
        self.genome = new_genome

class GeneticAlgorithm:
    def __init__(
        self,
        genome_length,
        population_size=50,
        generations=20,
        crossover_rate=0.8,
        mutation_rate=0.01,
        fitness_func=None,
        elite_size=2,  # Number of best individuals to keep
    ):
        self.genome_length = genome_length
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fitness_func = fitness_func
        self.elite_size = elite_size
        self.population = []

    def _initialize_population(self):
        """Initialize population with a mix of strategies"""
        self.population = []
        
        # 1. All features selected
        self.population.append(Individual([1] * self.genome_length))
        
        # 2. Single feature individuals
        for i in range(min(5, self.genome_length)):  # At most 5 single-feature individuals
            genome = [0] * self.genome_length
            genome[i] = 1
            self.population.append(Individual(genome))
        
        # 3. Random individuals with fixed number of features
        n_features_list = [3, 5, 7]  # Different numbers of features to try
        for n_features in n_features_list:
            if n_features < self.genome_length:
                for _ in range(2):  # Two individuals for each n_features
                    genome = [0] * self.genome_length
                    # Randomly select n_features positions to set to 1
                    positions = random.sample(range(self.genome_length), n_features)
                    for pos in positions:
                        genome[pos] = 1
                    self.population.append(Individual(genome))
        
        # 4. Fill the rest with completely random individuals
        while len(self.population) < self.population_size:
            genome = [random.randint(0, 1) for _ in range(self.genome_length)]
            self.population.append(Individual(genome))

    def _select_parents(self):
        # Ensure all fitness values are positive for selection
        min_fitness = min(ind.fitness for ind in self.population)
        if min_fitness < 0:
            # Shift all fitness values to be positive
            adjusted_fitness = [ind.fitness - min_fitness + 1e-10 for ind in self.population]
        else:
            adjusted_fitness = [ind.fitness + 1e-10 for ind in self.population]
            
        total_fitness = sum(adjusted_fitness)
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind, fitness in zip(self.population, adjusted_fitness):
            current += fitness
            if current >= pick:
                return ind
        return self.population[-1]

    def run(self):
        self._initialize_population()
        for ind in self.population:
            ind.evaluate(self.fitness_func)

        # Initialize history
        history = {
            'best_genomes': [],
            'best_fitnesses': []
        }

        for _ in tqdm(range(self.generations), desc="Genetic Algorithm Progress"):
            # Sort population by fitness
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Keep elite individuals
            elite = self.population[:self.elite_size]
            
            # Create new population
            new_population = elite.copy()  # Start with elite individuals
            
            # Generate rest of the population
            while len(new_population) < self.population_size:
                parent1 = self._select_parents()
                parent2 = self._select_parents()
                if random.random() < self.crossover_rate:
                    child1, child2 = Individual.crossover(parent1, parent2)
                else:
                    child1 = Individual(parent1.genome.copy())
                    child2 = Individual(parent2.genome.copy())

                child1.mutate(self.mutation_rate)
                child2.mutate(self.mutation_rate)

                child1.evaluate(self.fitness_func)
                child2.evaluate(self.fitness_func)

                new_population.extend([child1, child2])

            # Update population
            self.population = sorted(
                new_population, key=lambda ind: ind.fitness, reverse=True
            )[: self.population_size]

            # Record best individual
            best = max(self.population, key=lambda ind: ind.fitness)
            history['best_genomes'].append(best.genome)
            history['best_fitnesses'].append(best.fitness)

        best = max(self.population, key=lambda ind: ind.fitness)
        return best.genome, best.fitness, history
