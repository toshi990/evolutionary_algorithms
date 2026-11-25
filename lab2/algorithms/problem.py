import itertools
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Problem:
    n: int
    k: int
    edges: tuple[tuple[int, int], ...]
    positions: tuple[tuple[float, float], ...] = ()
    neighborhoods: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.n) is not int or type(self.k) is not int or not 1 <= self.k <= self.n:
            raise ValueError("Требуется 1 <= k <= n, n и k должны быть целыми.")
        masks = [1 << i for i in range(self.n)]
        seen = set()
        for u, v in self.edges:
            if type(u) is not int or type(v) is not int or not (0 <= u < self.n and 0 <= v < self.n) or u == v:
                raise ValueError("Некорректное ребро графа.")
            edge = tuple(sorted((u, v)))
            if edge in seen:
                raise ValueError("Повторяющееся ребро графа.")
            seen.add(edge)
            masks[u] |= 1 << v
            masks[v] |= 1 << u
        if self.positions and (len(self.positions) != self.n or any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in self.positions)):
            raise ValueError("Координаты должны содержать n конечных пар чисел.")
        object.__setattr__(self, "neighborhoods", tuple(masks))

    def validate_genome(self, genome):
        if len(genome) != self.n or any(type(x) is not int or x not in (0, 1) for x in genome):
            raise ValueError("Хромосома должна содержать n целых нулей и единиц.")

    def covered_mask(self, genome):
        mask = 0
        for bit, neighborhood in zip(genome, self.neighborhoods):
            if bit:
                mask |= neighborhood
        return mask

    def coverage(self, genome):
        return self.covered_mask(genome).bit_count()

    def feasible(self, genome):
        self.validate_genome(genome)
        return sum(genome) <= self.k

    def decode(self, genome):
        self.validate_genome(genome)
        mask = self.covered_mask(genome)
        return {"stations": [i + 1 for i, bit in enumerate(genome) if bit],
                "covered": [i + 1 for i in range(self.n) if mask & (1 << i)],
                "uncovered": [i + 1 for i in range(self.n) if not mask & (1 << i)],
                "coverage": mask.bit_count(), "station_count": sum(genome),
                "feasible": sum(genome) <= self.k, "genome": list(genome)}


def random_feasible(problem, rng):
    selected = set(rng.sample(range(problem.n), problem.k))
    return tuple(int(i in selected) for i in range(problem.n))


def generate_problem(settings):
    n, k = settings["vertices"], settings["max_stations"]
    probability = settings["chord_probability"]
    if type(n) is not int or n < 3 or not 0 <= probability <= 1:
        raise ValueError("Нужно не менее трёх вершин и вероятность из [0, 1].")
    rng = random.Random(settings["seed"])
    edges = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges and rng.random() < probability:
                edges.add((u, v))
    positions = tuple((round(math.cos(2 * math.pi * i / n), 6),
                       round(math.sin(2 * math.pi * i / n), 6)) for i in range(n))
    return Problem(n, k, tuple(sorted(edges)), positions)


def save_problem(problem, settings, path):
    payload = {"description": "Неориентированный граф; пункт покрывает себя и соседей, радиус 1.",
               "generation": settings, "n": problem.n, "k": problem.k,
               "edges": problem.edges, "positions": problem.positions,
               "indexing": "Рёбра: 0..n-1; номера вершин в отчёте: 1..n."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_problem(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Problem(data["n"], data["k"], tuple(tuple(e) for e in data["edges"]),
                   tuple(tuple(p) for p in data.get("positions", [])))


def exact_optimum(problem):
    best_value, best_selection, count = -1, (), 0
    for selected in itertools.combinations(range(problem.n), problem.k):
        mask = 0
        for vertex in selected:
            mask |= problem.neighborhoods[vertex]
        value = mask.bit_count()
        count += 1
        if value > best_value:
            best_value, best_selection = value, selected
    return {"coverage": best_value, "stations": [i + 1 for i in best_selection],
            "combinations": count}


def solution_examples(problem):
    choices = [(0,), tuple(range(problem.k)), tuple(range(problem.k + 1)), tuple(range(problem.n))]
    examples = []
    for selected in choices:
        genome = tuple(int(i in selected) for i in range(problem.n))
        examples.append(problem.decode(genome))
    return examples
