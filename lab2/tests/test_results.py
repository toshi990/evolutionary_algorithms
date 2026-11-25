import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from algorithms.experiments import save_results, verify_results
from algorithms.genetic import METHODS, Settings, random_search, run_ga
from algorithms.problem import Problem, save_problem


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.problem = Problem(6, 2, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
        self.settings = Settings(population_size=10, evaluation_budget=37)

    def test_real_objective_call_budget(self):
        original = Problem.coverage
        for method in [*METHODS, 'random']:
            with patch.object(Problem, 'coverage', autospec=True, side_effect=original) as objective:
                result = (random_search(self.problem, self.settings, 14) if method == 'random'
                          else run_ga(self.problem, self.settings, 14, method))
            self.assertEqual(objective.call_count, 37)
            self.assertEqual(result.evaluations, 37)

    def test_saved_results_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = root / 'graph.json'
            save_problem(self.problem, {}, graph)
            config = {'student': {}, 'data': {}}
            result = run_ga(self.problem, self.settings, 14)
            output = root / 'results' / 'demo'
            save_results(self.problem, self.settings, config, [result], output, graph)
            self.assertEqual(verify_results(self.problem, output, graph)['checked_solutions'], 1)
            changes = {
                'solutions.json': lambda values: [],
                'summary.json': lambda values: [{**values[0], 'mean': -1}],
            }
            for filename, change in changes.items():
                path = output / filename
                original = path.read_text()
                path.write_text(json.dumps(change(json.loads(original))))
                with self.assertRaises(ValueError):
                    verify_results(self.problem, output, graph)
                path.write_text(original)
            path = output / 'runs.csv'
            original = path.read_text()
            rows = list(csv.DictReader(io.StringIO(original)))
            rows[0]['coverage'] = '-1'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                verify_results(self.problem, output, graph)
            path.write_text(original)
            path = output / 'history.csv'
            original = path.read_text()
            path.write_text(original.splitlines()[0] + '\n')
            with self.assertRaises(ValueError):
                verify_results(self.problem, output, graph)
            path.write_text(original)
            self.assertEqual(verify_results(self.problem, output, graph)['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
