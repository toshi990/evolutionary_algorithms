import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Item:
    name: str
    utility: int
    mass_g: int
    reliability_permille: int


@dataclass(frozen=True)
class Solution:
    genome: tuple[int, ...]
    costs: tuple[int, int, int]


class Problem:
    def __init__(self, categories):
        self.categories = tuple((name, tuple(items)) for name, items in categories)
        if not self.categories or len({name for name, _ in self.categories}) != len(self.categories):
            raise ValueError("Нужны непустые категории с уникальными названиями.")
        for name, items in self.categories:
            if not name or len(items) < 2:
                raise ValueError("В каждой категории нужно не менее двух альтернатив.")
            for item in items:
                if (any(type(value) is not int for value in (item.utility, item.mass_g, item.reliability_permille))
                        or item.utility <= 0 or item.mass_g <= 0 or not 1 <= item.reliability_permille <= 1000):
                    raise ValueError("Полезность и масса положительны; надёжность — целое 1..1000.")
        self.sizes = tuple(len(items) for _, items in self.categories)
        self.denominator = 1000 ** len(self.categories)
        self.utility_bounds = self._sum_bounds("utility")
        self.mass_bounds = self._sum_bounds("mass_g")
        self.reliability_bounds = tuple(math.prod(fn(item.reliability_permille for item in items)
                                                   for _, items in self.categories) for fn in (min, max))

    def _sum_bounds(self, field):
        return tuple(sum(fn(getattr(item, field) for item in items) for _, items in self.categories) for fn in (min, max))

    def validate_genome(self, genome):
        if len(genome) != len(self.sizes) or any(type(gene) is not int or not 0 <= gene < size for gene, size in zip(genome, self.sizes)):
            raise ValueError("Нужно выбрать ровно один допустимый вариант из каждой категории.")

    def random_genome(self, rng):
        return tuple(rng.randrange(size) for size in self.sizes)

    def evaluate(self, genome):
        self.validate_genome(genome)
        selected = [items[gene] for (_, items), gene in zip(self.categories, genome)]
        return Solution(tuple(genome), (-sum(item.utility for item in selected),
                                       sum(item.mass_g for item in selected),
                                       -math.prod(item.reliability_permille for item in selected)))

    def normalize(self, solution):
        utility, mass, reliability = solution.costs
        ulo, uhi = self.utility_bounds
        mlo, mhi = self.mass_bounds
        rlo, rhi = self.reliability_bounds
        return ((uhi + utility) / (uhi - ulo) if uhi != ulo else 0.0,
                (mass - mlo) / (mhi - mlo) if mhi != mlo else 0.0,
                (rhi + reliability) / (rhi - rlo) if rhi != rlo else 0.0)

    def decode(self, solution):
        return {"genome": list(solution.genome), "utility": -solution.costs[0],
                "mass_g": solution.costs[1], "reliability_numerator": -solution.costs[2],
                "reliability": -solution.costs[2] / self.denominator,
                "normalized_costs": list(self.normalize(solution)),
                "items": [{"category": category, "name": items[gene].name,
                           "utility": items[gene].utility, "mass_g": items[gene].mass_g,
                           "reliability_permille": items[gene].reliability_permille}
                          for (category, items), gene in zip(self.categories, solution.genome)]}

    @property
    def search_space_size(self):
        return math.prod(self.sizes)


def generate_data(seed):
    rng = random.Random(seed)
    bases = [("Палатка", 2000, 48), ("Спальный мешок", 1200, 40), ("Горелка", 650, 30),
             ("Фильтр воды", 350, 44), ("Навигация", 220, 40), ("Связь", 450, 45),
             ("Аптечка", 550, 38), ("Освещение", 180, 25), ("Питание электроники", 650, 32),
             ("Защитная одежда", 1100, 42)]
    categories = []
    for category, mass, utility in bases:
        items = [Item(f"{category}: облегчённый вариант", round(utility * rng.uniform(0.72, 0.84)),
                      round(mass * rng.uniform(0.58, 0.72)), rng.randint(900, 945)),
                 Item(f"{category}: функциональный вариант", round(utility * rng.uniform(1.10, 1.24)),
                      round(mass * rng.uniform(0.92, 1.08)), rng.randint(945, 973)),
                 Item(f"{category}: надёжный вариант", round(utility * rng.uniform(0.88, 1.04)),
                      round(mass * rng.uniform(1.25, 1.47)), rng.randint(983, 997))]
        categories.append((category, tuple(items)))
    return Problem(categories)


def save_data(problem, seed, path):
    payload = {"seed": seed, "synthetic": True,
               "assumptions": ["Ровно один предмет из каждой обязательной категории.",
                               "Надёжность комплекта — вероятность исправности всех предметов при независимых отказах."],
               "categories": [{"name": name, "items": [item.__dict__ for item in items]} for name, items in problem.categories]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_data(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Problem([(category["name"], tuple(Item(**item) for item in category["items"])) for category in payload["categories"]])


def enumerate_solutions(problem):
    return (problem.evaluate(genome) for genome in itertools.product(*(range(size) for size in problem.sizes)))
