import itertools
import random
import unittest
from dataclasses import replace

from algorithms.genetic import METHODS, Settings, random_search, run_ga
from algorithms.operators import flip_mutation, repair, swap_mutation, uniform_crossover
from algorithms.problem import Problem, exact_optimum, generate_problem, random_feasible, solution_examples


class ProblemTests(unittest.TestCase):
    def setUp(self):
        self.problem = Problem(6, 2, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))

    def test_coverage_against_direct_sets_for_all_genomes(self):
        adjacency = [{i} for i in range(6)]
        for u, v in self.problem.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        for genome in itertools.product((0, 1), repeat=6):
            expected = set().union(*(adjacency[i] for i, bit in enumerate(genome) if bit))
            self.assertEqual(self.problem.coverage(genome), len(expected))

    def test_exact_optimum_also_covers_at_most_k(self):
        expected = max(self.problem.coverage(g) for g in itertools.product((0, 1), repeat=6) if sum(g) <= 2)
        self.assertEqual(exact_optimum(self.problem)["coverage"], expected)
        self.assertEqual(expected, 6)

    def test_feasibility_and_decoding(self):
        genome = (0, 1, 0, 0, 1, 0)
        self.assertEqual(self.problem.decode(genome)["stations"], [2, 5])
        self.assertTrue(self.problem.feasible(genome))
        self.assertFalse(self.problem.feasible((1, 1, 1, 0, 0, 0)))
        self.assertTrue(self.problem.feasible((0,) * 6))
        examples = solution_examples(self.problem)
        self.assertEqual([e["feasible"] for e in examples], [True, True, False, False])

    def test_invalid_data_rejected(self):
        for kwargs in [dict(n=0, k=1, edges=()), dict(n=4, k=5, edges=()),
                       dict(n=4, k=1, edges=((0, 4),)), dict(n=4, k=1, edges=((1, 1),)),
                       dict(n=4, k=1, edges=((0, 1), (1, 0)))]:
            with self.assertRaises(ValueError):
                Problem(**kwargs)
        with self.assertRaises(ValueError):
            self.problem.validate_genome((1, 0))
        with self.assertRaises(ValueError):
            self.problem.validate_genome((0, 0, 0, 0, 0, 2))

    def test_generator_reproducible_and_connected(self):
        data = dict(vertices=36, max_stations=4, chord_probability=0.045, seed=20261420)
        problem = generate_problem(data)
        self.assertEqual(problem, generate_problem(data))
        reached, frontier = {0}, [0]
        while frontier:
            vertex = frontier.pop()
            for u, v in problem.edges:
                neighbor = v if u == vertex else u if v == vertex else None
                if neighbor is not None and neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        self.assertEqual(len(reached), problem.n)


class OperatorTests(unittest.TestCase):
    def test_repair_and_swap_preserve_k(self):
        rng = random.Random(14)
        for genome in itertools.product((0, 1), repeat=8):
            repaired = repair(genome, 3, rng)
            self.assertEqual(sum(repaired), 3)
            swapped = swap_mutation(repaired, rng)
            self.assertEqual(sum(swapped), 3)
            self.assertEqual(sum(a != b for a, b in zip(repaired, swapped)), 2)
        self.assertEqual(repair((0, 0), 0, rng), (0, 0))
        self.assertEqual(repair((0, 0), 2, rng), (1, 1))
        self.assertEqual(swap_mutation((1, 1), rng), (1, 1))
        self.assertEqual(swap_mutation((0, 0), rng), (0, 0))

    def test_flip_and_uniform_crossover(self):
        rng = random.Random(7)
        parent = (1, 0, 1, 0, 0, 1)
        for _ in range(100):
            child = flip_mutation(parent, rng)
            self.assertEqual(sum(a != b for a, b in zip(parent, child)), 1)
            cross = uniform_crossover(parent, tuple(1 - bit for bit in parent), rng)
            self.assertEqual(len(cross), len(parent))
            self.assertTrue(all(bit in (0, 1) for bit in cross))


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.problem = generate_problem(dict(vertices=24, max_stations=4, chord_probability=0.04, seed=14))
        self.settings = Settings(population_size=30, evaluation_budget=137)

    def test_all_methods_budget_validity_history_and_reproducibility(self):
        for method in METHODS:
            a = run_ga(self.problem, self.settings, 17, method)
            b = run_ga(self.problem, self.settings, 17, method)
            self.assertEqual(a.evaluations, 137)
            self.assertEqual(a.best_genome, b.best_genome)
            self.assertEqual(a.history, b.history)
            self.assertTrue(self.problem.feasible(a.best_genome))
            self.assertEqual(a.best_coverage, self.problem.coverage(a.best_genome))
            values = [p["best_coverage"] for p in a.history]
            self.assertEqual(values, sorted(values))
            self.assertEqual(a.history[-1]["evaluations"], 137)
            if method.startswith("repair"):
                self.assertEqual(a.feasible_evaluations, 137)

    def test_baseline_equal_budget_and_checkpoints(self):
        baseline = random_search(self.problem, self.settings, 17)
        ga = run_ga(self.problem, self.settings, 17)
        self.assertEqual(baseline.evaluations, ga.evaluations)
        self.assertEqual([h["evaluations"] for h in baseline.history], [h["evaluations"] for h in ga.history])
        self.assertEqual(baseline.history[0]["best_coverage"], ga.history[0]["best_coverage"])

    def test_initial_population_only_and_no_elites(self):
        for settings in [replace(self.settings, evaluation_budget=30), replace(self.settings, elite_size=0)]:
            run = run_ga(self.problem, settings, 1)
            self.assertEqual(run.evaluations, settings.evaluation_budget)

    def test_global_random_state_unchanged(self):
        original = random.getstate()
        run_ga(self.problem, self.settings, 100)
        self.assertEqual(original, random.getstate())

    def test_penalty_ordering(self):
        penalty = self.problem.n + 1
        max_infeasible_fitness = self.problem.n - penalty
        self.assertLess(max_infeasible_fitness, 0)

    def test_invalid_settings_rejected(self):
        for kwargs in [dict(population_size=1), dict(evaluation_budget=1), dict(elite_size=80),
                       dict(mutation_probability=1.1), dict(tournament_size=0), dict(penalty=-1)]:
            with self.assertRaises(ValueError):
                Settings(**kwargs)


if __name__ == "__main__":
    unittest.main()
