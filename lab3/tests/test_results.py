import json
from pathlib import Path
import tempfile
import unittest

from algorithms.experiments import save_results
from algorithms.genetic import Settings, run_nsga2, run_weighted
from algorithms.problem import Problem, generate_data, save_data
from algorithms.verification import verify_results


class ResultTests(unittest.TestCase):
    def test_files_and_detection_of_inconsistent_results(self):
        problem = Problem(generate_data(14).categories[:2])
        settings = Settings(population_size=8, evaluation_budget=35)
        profiles = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        config = {'student': {}, 'data': {}, 'weighted_profiles': profiles,
                  'hypervolume_reference': [1.1, 1.1, 1.1]}
        results = [run_nsga2(problem, settings, 17),
                   run_weighted(problem, settings, 17, profiles)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'equipment.json'
            save_data(problem, 14, data)
            directory = root / 'results' / 'demo'
            save_results(problem, settings, config, results, directory, data)
            self.assertEqual(verify_results(problem, directory, data)['runs'], 2)
            for filename in ['summary.csv', 'history.csv', 'fronts.csv', 'exact_front.csv']:
                path = directory / filename
                original = path.read_text()
                path.write_text(original.splitlines()[0] + '\n')
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    verify_results(problem, directory, data)
                path.write_text(original)
            for filename in ['summary.json', 'representatives.json', 'runs.json']:
                path = directory / filename
                original = path.read_text()
                payload = json.loads(original)
                if filename == 'summary.json':
                    payload[0]['mean'] = -1
                elif filename == 'representatives.json':
                    payload['solutions'][0]['mass_g'] = -1
                else:
                    payload[0]['history'][0]['hypervolume'] = -1
                path.write_text(json.dumps(payload))
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    verify_results(problem, directory, data)
                path.write_text(original)
            self.assertEqual(verify_results(problem, directory, data)['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
