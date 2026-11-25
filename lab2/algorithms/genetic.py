import math
import random
import time
from dataclasses import asdict, dataclass

from .operators import flip_mutation, repair, swap_mutation, uniform_crossover
from .problem import random_feasible


@dataclass(frozen=True)
class Settings:
    population_size: int = 80
    evaluation_budget: int = 4000
    tournament_size: int = 3
    elite_size: int = 2
    crossover_probability: float = 0.9
    mutation_probability: float = 0.35
    penalty: float = 37.0

    def __post_init__(self):
        counts = (self.population_size, self.evaluation_budget, self.tournament_size, self.elite_size)
        if any(type(x) is not int for x in counts):
            raise ValueError("Численные лимиты должны быть целыми.")
        if self.population_size < 2 or self.evaluation_budget < self.population_size:
            raise ValueError("Популяция >= 2, бюджет >= размер популяции.")
        if not 0 <= self.elite_size < self.population_size or self.tournament_size < 1:
            raise ValueError("Некорректный элитизм или размер турнира.")
        if not 0 <= self.crossover_probability <= 1 or not 0 <= self.mutation_probability <= 1:
            raise ValueError("Вероятности должны принадлежать [0, 1].")
        if not math.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("Штраф должен быть конечным положительным числом.")


@dataclass(frozen=True)
class Individual:
    genome: tuple[int, ...]
    coverage: int
    feasible: bool
    fitness: float


@dataclass
class RunResult:
    method: str
    seed: int
    best_genome: tuple[int, ...]
    best_coverage: int
    evaluations: int
    feasible_evaluations: int
    elapsed_seconds: float
    history: list[dict]

    def to_dict(self, problem):
        result = asdict(self)
        result["solution"] = problem.decode(self.best_genome)
        return result


METHODS = {
    "penalty_flip": ("penalty", "flip"),
    "repair_flip": ("repair", "flip"),
    "repair_swap": ("repair", "swap"),
}


def run_ga(problem, settings, seed, method="repair_swap"):
    if method not in METHODS:
        raise ValueError("Неизвестная конфигурация ГА.")
    strategy, mutation_name = METHODS[method]
    mutation = flip_mutation if mutation_name == "flip" else swap_mutation
    rng = random.Random(seed)
    started = time.perf_counter()
    evaluations, feasible_evaluations, best = 0, 0, None

    def evaluate(genome):
        nonlocal evaluations, feasible_evaluations, best
        value = problem.coverage(genome)
        violation = max(0, sum(genome) - problem.k)
        individual = Individual(genome, value, violation == 0,
                                value - settings.penalty * violation)
        evaluations += 1
        feasible_evaluations += int(individual.feasible)
        if individual.feasible and (best is None or value > best.coverage):
            best = individual
        return individual

    def rank(individual):
        return individual.fitness, individual.coverage

    def select(population):
        return max((rng.choice(population) for _ in range(settings.tournament_size)), key=rank)

    population = [evaluate(random_feasible(problem, rng)) for _ in range(settings.population_size)]
    history = []
    generation = 0

    def log():
        history.append({"generation": generation, "evaluations": evaluations,
                        "best_coverage": best.coverage,
                        "mean_population_coverage": sum(x.coverage for x in population) / len(population),
                        "feasible_population_fraction": sum(x.feasible for x in population) / len(population)})

    log()
    while evaluations < settings.evaluation_budget:
        children = sorted(population, key=rank, reverse=True)[:settings.elite_size]
        while len(children) < settings.population_size and evaluations < settings.evaluation_budget:
            a, b = select(population).genome, select(population).genome
            child = uniform_crossover(a, b, rng) if rng.random() < settings.crossover_probability else a
            if rng.random() < settings.mutation_probability:
                child = mutation(child, rng)
            if strategy == "repair":
                child = repair(child, problem.k, rng)
            children.append(evaluate(child))
        population = children
        generation += 1
        log()
    return RunResult(method, seed, best.genome, best.coverage, evaluations,
                     feasible_evaluations, time.perf_counter() - started, history)


def random_search(problem, settings, seed):
    rng = random.Random(seed)
    started = time.perf_counter()
    best_genome, best_value, history = None, -1, []
    step = settings.population_size - settings.elite_size
    checkpoints = {settings.population_size, settings.evaluation_budget}
    checkpoints.update(range(settings.population_size + step, settings.evaluation_budget, step))
    for evaluation in range(1, settings.evaluation_budget + 1):
        genome = random_feasible(problem, rng)
        value = problem.coverage(genome)
        if value > best_value:
            best_genome, best_value = genome, value
        if evaluation in checkpoints:
            history.append({"generation": len(history), "evaluations": evaluation,
                            "best_coverage": best_value,
                            "mean_population_coverage": "", "feasible_population_fraction": 1.0})
    return RunResult("random", seed, best_genome, best_value, settings.evaluation_budget,
                     settings.evaluation_budget, time.perf_counter() - started, history)
