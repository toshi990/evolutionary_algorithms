import random
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from algorithms.experiment import run_experiments
from algorithms.genetic_algorithm import GAConfig, _crossover, _mutate, run_ga, run_random_search
from algorithms.objective import (
    DIMENSION,
    GLOBAL_MINIMIZER_COORDINATE,
    GLOBAL_MINIMUM,
    stiblinski_tang,
)
from algorithms.visualizer import VisualizerState


class ObjectiveTests(unittest.TestCase):
    def test_zero_vector(self):
        self.assertEqual(stiblinski_tang([0.0] * DIMENSION), 0.0)

    def test_known_minimum(self):
        vector = [GLOBAL_MINIMIZER_COORDINATE] * DIMENSION
        self.assertAlmostEqual(stiblinski_tang(vector), GLOBAL_MINIMUM, places=4)

    def test_wrong_dimension(self):
        with self.assertRaises(ValueError):
            stiblinski_tang([0.0])


class OperatorTests(unittest.TestCase):
    def test_arithmetic_crossover_stays_between_parents(self):
        first = (-4.0,) * DIMENSION
        second = (2.0,) * DIMENSION
        children = _crossover(first, second, 1.0, random.Random(1))
        for child in children:
            self.assertTrue(all(-4.0 <= value <= 2.0 for value in child))

    def test_mutation_respects_bounds(self):
        individual = (5.0,) * DIMENSION
        mutated = _mutate(individual, 1.0, 100.0, random.Random(1))
        self.assertTrue(all(-5.0 <= value <= 5.0 for value in mutated))

    def test_random_reset_mutation_is_reproducible_and_bounded(self):
        individual = (2.75,) * DIMENSION
        first = _mutate(individual, 1.0, 0.8, random.Random(7), 1.0)
        second = _mutate(individual, 1.0, 0.8, random.Random(7), 1.0)
        self.assertEqual(first, second)
        self.assertTrue(all(-5.0 <= value <= 5.0 for value in first))
        self.assertNotEqual(first, individual)


class AlgorithmTests(unittest.TestCase):
    def test_actual_function_calls_and_monotonic_history(self):
        config = GAConfig(population_size=30, evaluation_budget=500)
        for method in (lambda: run_ga(config, 42), lambda: run_random_search(500, 42)):
            with patch("algorithms.genetic_algorithm.stiblinski_tang", wraps=stiblinski_tang) as objective:
                result = method()
            self.assertEqual(objective.call_count, 500)
            self.assertEqual(result.evaluations, 500)
            self.assertEqual(result.history[-1][0], 500)
            self.assertAlmostEqual(stiblinski_tang(result.best_vector), result.best_value)
            self.assertTrue(all(
                previous[1] >= current[1]
                for previous, current in zip(result.history, result.history[1:])
            ))

    def test_reproducibility_and_budget(self):
        config = GAConfig(population_size=30, evaluation_budget=500)
        first = run_ga(config, seed=42)
        second = run_ga(config, seed=42)
        self.assertEqual(first.best_value, second.best_value)
        self.assertEqual(first.best_vector, second.best_vector)
        self.assertEqual(first.evaluations, 500)

    def test_visualizer_exposes_tournament_and_generation(self):
        state = VisualizerState(GAConfig(population_size=30), seed=14)
        state.step()
        self.assertEqual(state.stage, "tournament_a")
        self.assertEqual(len(state.tournament_a), 3)
        expected_winner = min(state.tournament_a, key=state.values.__getitem__)
        self.assertEqual(state.winner_a, expected_winner)

        while state.generation == 0:
            state.step()
        self.assertEqual(state.generation, 1)
        self.assertEqual(len(state.population), 30)
        self.assertEqual(state.evaluations, 59)


class ExperimentTests(unittest.TestCase):
    def test_generation_data_and_partial_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            grouped = run_experiments(
                output, [0, 1], GAConfig(population_size=30, evaluation_budget=60)
            )
            self.assertEqual(len(grouped), 3)
            for method, runs in grouped.items():
                self.assertTrue(all(run.evaluations == 60 for run in runs))
                if method == "Random_search":
                    continue
                with (output / f"generations_{method}.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual([int(row["evaluations"]) for row in rows], [30, 59, 60])
                self.assertEqual([int(row["generation"]) for row in rows], [0, 1, 2])
                self.assertAlmostEqual(
                    float(rows[-1]["mean"]), sum(run.best_value for run in runs) / len(runs)
                )
                self.assertIn("Поколение", (output / f"generations_{method}.svg").read_text())

    def test_visualize_command_opens_html_launcher(self):
        from main import main

        with patch("sys.argv", ["main.py", "visualize"]), patch("visualize.main") as launch:
            main()
        launch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
