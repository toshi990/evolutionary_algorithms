import math
import random
import time
from dataclasses import dataclass

from .pareto import environmental_selection, hypervolume, pareto_archive, ranks_and_distances


@dataclass(frozen=True)
class Settings:
    population_size: int = 60
    evaluation_budget: int = 4200
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    tournament_size: int = 2
    weighted_elite_size: int = 2

    def __post_init__(self):
        if any(type(value) is not int for value in (self.population_size, self.evaluation_budget,
                                                    self.tournament_size, self.weighted_elite_size)):
            raise ValueError("Размеры и лимиты должны быть целыми.")
        if self.population_size < 4 or self.evaluation_budget < self.population_size or self.tournament_size < 2:
            raise ValueError("Популяция >= 4, бюджет >= популяции, турнир >= 2.")
        if not 0 <= self.weighted_elite_size < self.population_size:
            raise ValueError("Число элитных особей должно быть меньше популяции.")
        if any(not 0 <= p <= 1 for p in (self.crossover_probability, self.mutation_probability)):
            raise ValueError("Вероятности должны принадлежать [0, 1].")


@dataclass
class RunResult:
    method: str
    seed: int
    evaluations: int
    archive: list
    population: list
    history: list[dict]
    elapsed_seconds: float


def offspring(parent_a, parent_b, sizes, rng, settings):
    child = parent_a
    if rng.random() < settings.crossover_probability:
        child = tuple(a if rng.random() < 0.5 else b for a, b in zip(parent_a, parent_b))
    genes = list(child)
    for index, size in enumerate(sizes):
        if rng.random() < settings.mutation_probability:
            new_value = rng.randrange(size - 1)
            genes[index] = new_value + int(new_value >= genes[index])
    return tuple(genes)


def archive_checkpoint(problem, archive, new_solutions, evaluations, history, reference):
    archive = pareto_archive([*archive, *new_solutions])
    history.append({"evaluations": evaluations, "archive_size": len(archive),
                    "hypervolume": hypervolume([problem.normalize(s) for s in archive], reference)})
    return archive


def run_nsga2(problem, settings, seed, reference=(1.1, 1.1, 1.1)):
    started, rng = time.perf_counter(), random.Random(seed)
    population = [problem.evaluate(problem.random_genome(rng)) for _ in range(settings.population_size)]
    evaluations, history = len(population), []
    archive = archive_checkpoint(problem, [], population, evaluations, history, reference)
    while evaluations < settings.evaluation_budget:
        ranks, distances = ranks_and_distances(population)

        def tournament():
            contestants = [rng.randrange(len(population)) for _ in range(settings.tournament_size)]
            winner = min(contestants, key=lambda i: (ranks[i], -distances[i]))
            return population[winner].genome

        children = []
        size = min(settings.population_size, settings.evaluation_budget - evaluations)
        for _ in range(size):
            genome = offspring(tournament(), tournament(), problem.sizes, rng, settings)
            children.append(problem.evaluate(genome))
        evaluations += len(children)
        population = environmental_selection(population + children, settings.population_size)
        archive = archive_checkpoint(problem, archive, children, evaluations, history, reference)
    return RunResult("nsga2", seed, evaluations, archive, population, history, time.perf_counter() - started)


def validate_weights(profiles):
    if not profiles:
        raise ValueError("Нужен хотя бы один набор весов.")
    for weights in profiles:
        if (len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-10)):
            raise ValueError("Три неотрицательных конечных веса должны давать сумму 1.")


def run_weighted(problem, settings, seed, profiles, reference=(1.1, 1.1, 1.1)):
    validate_weights(profiles)
    if settings.evaluation_budget < len(profiles) * settings.population_size:
        raise ValueError("Бюджет взвешенной суммы должен покрывать все начальные популяции.")
    started = time.perf_counter()
    base, remainder = divmod(settings.evaluation_budget, len(profiles))
    evaluations, history, archive, population = 0, [], [], []
    for profile_index, weights in enumerate(profiles):
        rng = random.Random(seed * 1000 + profile_index)
        budget = base + int(profile_index < remainder)
        population = [problem.evaluate(problem.random_genome(rng)) for _ in range(settings.population_size)]
        spent = len(population)
        evaluations += spent
        archive = archive_checkpoint(problem, archive, population, evaluations, history, reference)

        def score(solution):
            return sum(w * value for w, value in zip(weights, problem.normalize(solution)))

        while spent < budget:
            candidates = list(sorted(population, key=score)[:settings.weighted_elite_size])
            new_solutions = []

            def tournament():
                return min((rng.choice(population) for _ in range(settings.tournament_size)), key=score).genome

            size = min(settings.population_size, budget - spent)
            for _ in range(size):
                genome = offspring(tournament(), tournament(), problem.sizes, rng, settings)
                new_solutions.append(problem.evaluate(genome))
            spent += size
            evaluations += size
            population = sorted(candidates + new_solutions, key=score)[:settings.population_size]
            archive = archive_checkpoint(problem, archive, new_solutions, evaluations, history, reference)
    return RunResult("weighted", seed, evaluations, archive, population, history, time.perf_counter() - started)
