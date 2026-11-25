import itertools
import math
import random
import unittest
from unittest.mock import patch

from algorithms.genetic import Settings, offspring, run_nsga2, run_weighted
from algorithms.pareto import crowding_distance, dominates, environmental_selection, hypervolume, nondominated_sort, pareto_archive
from algorithms.plots import convergence_data
from algorithms.problem import Item, Problem, Solution, enumerate_solutions, generate_data


class ProblemTests(unittest.TestCase):
    def test_exact_values_and_normalization(self):
        problem = Problem([("A", (Item("a", 10, 100, 900), Item("b", 20, 200, 990))),
                           ("B", (Item("c", 30, 300, 950), Item("d", 40, 400, 980)))])
        solution = problem.evaluate((0, 1))
        self.assertEqual(solution.costs, (-50, 500, -900 * 980))
        self.assertAlmostEqual(problem.decode(solution)["reliability"], 0.882)
        self.assertEqual(problem.search_space_size, 4)
        for candidate in enumerate_solutions(problem):
            self.assertTrue(all(0 <= value <= 1 for value in problem.normalize(candidate)))

    def test_input_validation(self):
        problem = generate_data(14)
        for genome in [(0,), (3,) * 10, (0.5,) * 10]:
            with self.assertRaises(ValueError):
                problem.evaluate(genome)
        with self.assertRaises(ValueError):
            Problem([("A", (Item("bad", 1, 1, 1001), Item("ok", 2, 2, 999)))])

    def test_generation_and_variation(self):
        problem = generate_data(20261403)
        self.assertEqual(problem.categories, generate_data(20261403).categories)
        self.assertEqual(problem.search_space_size, 59049)
        self.assertEqual(sum(problem.sizes), 30)
        rng = random.Random(3)
        settings = Settings(mutation_probability=1.0, crossover_probability=0.0)
        parent = problem.random_genome(rng)
        for _ in range(100):
            child = offspring(parent, parent, problem.sizes, rng, settings)
            problem.validate_genome(child)
            self.assertTrue(all(a != b for a, b in zip(parent, child)))


class ParetoTests(unittest.TestCase):
    def test_strict_dominance_and_layers(self):
        self.assertTrue(dominates((1, 1, 1), (1, 2, 1)))
        self.assertFalse(dominates((1, 1, 1), (1, 1, 1)))
        self.assertFalse(dominates((1, 3, 1), (3, 1, 1)))
        costs = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (1, 1, 1)]
        population = [Solution((i,), cost) for i, cost in enumerate(costs)]
        self.assertEqual(nondominated_sort(population), [[0], [1, 3], [2]])

    def test_sort_and_archive_against_direct_definition(self):
        rng = random.Random(44)
        for _ in range(50):
            population = [Solution((i,), tuple(rng.randrange(8) for _ in range(3))) for i in range(35)]
            expected = {solution.costs for solution in population
                        if not any(dominates(other.costs, solution.costs) for other in population)}
            self.assertEqual({s.costs for s in pareto_archive(population)}, expected)
            fronts = nondominated_sort(population)
            self.assertEqual({population[i].costs for i in fronts[0]}, expected)
            flattened = [i for front in fronts for i in front]
            self.assertEqual(sorted(flattened), list(range(len(population))))

    def test_crowding_normalization_and_constants(self):
        population = [Solution((i,), cost) for i, cost in enumerate([(0, 2, 7), (1, 1, 7), (2, 0, 7)])]
        distance = crowding_distance(population, [0, 1, 2])
        self.assertTrue(math.isinf(distance[0]) and math.isinf(distance[2]))
        self.assertAlmostEqual(distance[1], 2.0)
        equal = [Solution((i,), (1, 1, 1)) for i in range(4)]
        self.assertTrue(all(value == 0 for value in crowding_distance(equal, list(range(4))).values()))
        self.assertEqual(len(environmental_selection(population, 2)), 2)

    def test_hypervolume_known_cases(self):
        self.assertEqual(hypervolume([], (1, 1, 1)), 0)
        self.assertEqual(hypervolume([(0, 0, 0)], (1, 1, 1)), 1)
        self.assertAlmostEqual(hypervolume([(0.5, 0, 0), (0, 0.5, 0)], (1, 1, 1)), 0.75)
        self.assertEqual(hypervolume([(0, 0, 0), (0, 0, 0), (1, 1, 1)], (2, 2, 2)), 8)
        with self.assertRaises(ValueError):
            hypervolume([(2, 0, 0)], (1, 1, 1))

    def test_hypervolume_against_independent_grid_volume(self):
        rng = random.Random(19)
        for _ in range(80):
            points = [tuple(rng.randrange(5) for _ in range(3)) for _ in range(8)]
            expected = sum(any(all(p[i] <= cell[i] for i in range(3)) for p in points)
                           for cell in itertools.product(range(5), repeat=3))
            self.assertEqual(hypervolume(points, (5, 5, 5)), expected)


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.problem = generate_data(14)
        self.settings = Settings(population_size=8, evaluation_budget=71)
        self.profiles = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.6, 0.2, 0.2),
                         (0.2, 0.6, 0.2), (0.2, 0.2, 0.6), (1/3, 1/3, 1/3)]

    def run_method(self, name):
        if name == "nsga2":
            return run_nsga2(self.problem, self.settings, 100)
        return run_weighted(self.problem, self.settings, 100, self.profiles)

    def test_actual_calls_budget_and_partial_generations(self):
        for method in ("nsga2", "weighted"):
            with patch.object(self.problem, "evaluate", wraps=self.problem.evaluate) as evaluate:
                result = self.run_method(method)
                self.assertEqual(evaluate.call_count, 71)
            self.assertEqual(result.evaluations, 71)
            self.assertEqual(result.history[-1]["evaluations"], 71)
            self.assertTrue(all(a["hypervolume"] <= b["hypervolume"] + 1e-12 for a, b in zip(result.history, result.history[1:])))
            self.assertTrue(all(not dominates(a.costs, b.costs) for a in result.archive for b in result.archive))

    def test_reproducible_and_global_rng_unchanged(self):
        original = random.getstate()
        for method in ("nsga2", "weighted"):
            a, b = self.run_method(method), self.run_method(method)
            self.assertEqual(a.archive, b.archive)
            self.assertEqual(a.history, b.history)
        self.assertEqual(original, random.getstate())

    def test_history_alignment_uses_past_values(self):
        a, b = self.run_method("nsga2"), self.run_method("weighted")
        grid, values = convergence_data([a, b], 1.0)
        self.assertEqual(grid[-1], 71)
        for method, run in [("nsga2", a), ("weighted", b)]:
            for i, budget in enumerate(grid):
                past = [point for point in run.history if point["evaluations"] <= budget][-1]
                self.assertEqual(values[method]["mean"][i], 100 * past["hypervolume"])

    def test_invalid_settings_and_weights(self):
        for kwargs in [dict(population_size=1), dict(evaluation_budget=10), dict(mutation_probability=2), dict(tournament_size=1)]:
            with self.assertRaises(ValueError):
                Settings(**kwargs)
        with self.assertRaises(ValueError):
            run_weighted(self.problem, self.settings, 1, [(0.5, 0.5, 0.5)])
        with self.assertRaises(ValueError):
            run_weighted(self.problem, self.settings, 1, [(1, 0, 0)] * 10)


if __name__ == "__main__":
    unittest.main()
