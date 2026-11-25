from dataclasses import dataclass
import math
import random
from typing import List, Sequence, Tuple

from .objective import DIMENSION, LOWER_BOUND, UPPER_BOUND, stiblinski_tang

Individual = Tuple[float, ...]

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    evaluation_budget: int = 50_000
    tournament_size: int = 3
    crossover_probability: float = 0.9
    mutation_probability: float = 1.0 / DIMENSION
    mutation_sigma: float = 0.2
    random_reset_probability: float = 0.0
    elite_count: int = 1

    def validate(self) -> None:
        if self.population_size < 30:
            raise ValueError("Размер популяции должен быть не менее 30")
        if self.evaluation_budget < self.population_size:
            raise ValueError("Бюджет должен покрывать начальную популяцию")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("Размер турнира должен быть от 1 до размера популяции")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("Вероятность кроссовера должна быть в [0, 1]")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("Вероятность мутации должна быть в [0, 1]")
        if not math.isfinite(self.mutation_sigma) or self.mutation_sigma < 0.0:
            raise ValueError("Масштаб мутации должен быть конечным и неотрицательным")
        if not 0.0 <= self.random_reset_probability <= 1.0:
            raise ValueError("Вероятность сброса должна быть в [0, 1]")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("Число элитных особей должно быть от 1 до размера популяции минус 1")

@dataclass
class RunResult:
    method: str
    seed: int
    best_value: float
    best_vector: Individual
    evaluations: int
    history: List[Tuple[int, float]]

def _random_individual(rng: random.Random) -> Individual:
    return tuple(rng.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIMENSION))

def _tournament(
    population: Sequence[Individual],
    values: Sequence[float],
    size: int,
    rng: random.Random,
) -> Individual:
    indices = rng.sample(range(len(population)), size)
    winner = min(indices, key=lambda index: values[index])
    return population[winner]

def _crossover(
    first: Individual,
    second: Individual,
    probability: float,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    if rng.random() >= probability:
        return tuple(first), tuple(second)

    child_a = []
    child_b = []
    for x, y in zip(first, second):
        alpha = rng.random()
        child_a.append(alpha * x + (1.0 - alpha) * y)
        child_b.append((1.0 - alpha) * x + alpha * y)
    return tuple(child_a), tuple(child_b)

def _mutate(
    individual: Individual,
    probability: float,
    sigma: float,
    rng: random.Random,
    random_reset_probability: float = 0.0,
) -> Individual:
    result = []
    for coordinate in individual:
        if rng.random() < probability:
            if (
                random_reset_probability > 0.0
                and rng.random() < random_reset_probability
            ):

                coordinate = rng.uniform(LOWER_BOUND, UPPER_BOUND)
            else:
                coordinate += rng.gauss(0.0, sigma)

        result.append(min(UPPER_BOUND, max(LOWER_BOUND, coordinate)))
    return tuple(result)

def run_ga(config: GAConfig, seed: int) -> RunResult:
    config.validate()
    rng = random.Random(seed)
    population = [_random_individual(rng) for _ in range(config.population_size)]
    values = [stiblinski_tang(individual) for individual in population]
    evaluations = len(population)

    best_index = min(range(len(population)), key=values.__getitem__)
    best_vector = population[best_index]
    best_value = values[best_index]
    history = [(evaluations, best_value)]

    while evaluations < config.evaluation_budget:
        ranked = sorted(range(len(population)), key=values.__getitem__)
        elites = [population[index] for index in ranked[: config.elite_count]]
        elite_values = [values[index] for index in ranked[: config.elite_count]]
        offspring = []

        while len(offspring) < config.population_size - config.elite_count:
            parent_a = _tournament(
                population, values, config.tournament_size, rng
            )
            parent_b = _tournament(
                population, values, config.tournament_size, rng
            )
            child_a, child_b = _crossover(
                parent_a, parent_b, config.crossover_probability, rng
            )
            offspring.append(
                _mutate(
                    child_a,
                    config.mutation_probability,
                    config.mutation_sigma,
                    rng,
                    config.random_reset_probability,
                )
            )
            if len(offspring) < config.population_size - config.elite_count:
                offspring.append(
                    _mutate(
                        child_b,
                        config.mutation_probability,
                        config.mutation_sigma,
                        rng,
                        config.random_reset_probability,
                    )
                )

        remaining = config.evaluation_budget - evaluations
        evaluated_offspring = offspring[:remaining]
        offspring_values = [stiblinski_tang(child) for child in evaluated_offspring]
        evaluations += len(evaluated_offspring)

        if offspring_values:
            generation_best = min(range(len(offspring_values)), key=offspring_values.__getitem__)
            if offspring_values[generation_best] < best_value:
                best_value = offspring_values[generation_best]
                best_vector = evaluated_offspring[generation_best]

        history.append((evaluations, best_value))
        if len(evaluated_offspring) < len(offspring):
            break

        population = elites + offspring
        values = elite_values + offspring_values

    return RunResult(
        method=(
            f"Hybrid_GA_reset_{config.random_reset_probability:g}"
            if config.random_reset_probability > 0.0
            else f"GA_sigma_{config.mutation_sigma:g}"
        ),
        seed=seed,
        best_value=best_value,
        best_vector=best_vector,
        evaluations=evaluations,
        history=history,
    )

def run_random_search(evaluation_budget: int, seed: int) -> RunResult:
    if evaluation_budget <= 0:
        raise ValueError("Бюджет вычислений должен быть положительным")
    rng = random.Random(seed)
    best_vector = _random_individual(rng)
    best_value = stiblinski_tang(best_vector)
    history = [(1, best_value)]

    checkpoint_step = max(1, evaluation_budget // 500)
    for evaluation in range(2, evaluation_budget + 1):
        candidate = _random_individual(rng)
        value = stiblinski_tang(candidate)
        if value < best_value:
            best_value = value
            best_vector = candidate
        if evaluation % checkpoint_step == 0 or evaluation == evaluation_budget:
            history.append((evaluation, best_value))

    return RunResult(
        method="Random_search",
        seed=seed,
        best_value=best_value,
        best_vector=best_vector,
        evaluations=evaluation_budget,
        history=history,
    )
