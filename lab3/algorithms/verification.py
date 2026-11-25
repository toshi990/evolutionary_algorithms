import csv
import hashlib
import json
import math
from pathlib import Path

from .experiments import (choose_representatives, reference_front, solution_row,
                          summarize, write_json)
from .genetic import RunResult, Settings, validate_weights
from .pareto import hypervolume, pareto_archive
from .problem import Solution


def check_table(path, expected):
    with path.open(encoding='utf-8', newline='') as stream:
        actual = list(csv.DictReader(stream))
    if actual != [{key: str(value) for key, value in row.items()} for row in expected]:
        raise ValueError(f'Таблица {path.name} не совпадает с сохранёнными результатами.')


def verify_results(problem, directory, data_path):
    directory = Path(directory)
    saved = json.loads((directory / 'runs.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
    reference = json.loads((directory / 'reference.json').read_text(encoding='utf-8'))
    settings = Settings(**config['algorithm'])
    validate_weights(config['weighted_profiles'])
    seeds = config['seeds']
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('Неверный набор seed.')
    if config['data_sha256'] != hashlib.sha256(Path(data_path).read_bytes()).hexdigest():
        raise ValueError('Данные изменились после эксперимента.')
    exact, exact_hv = reference_front(problem, config['hypervolume_reference'])
    if (reference['combinations'] != problem.search_space_size or len(exact) != reference['front_size']
            or reference['point'] != config['hypervolume_reference']
            or not math.isclose(exact_hv, reference['hypervolume'], rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('Эталонный фронт не соответствует данным.')
    identities, restored, total_points = set(), [], 0
    for run in saved:
        identity = run['method'], run['seed']
        if identity in identities or identity[0] not in ('nsga2', 'weighted'):
            raise ValueError('Дублирование или неизвестный метод в серии.')
        identities.add(identity)
        if run['evaluations'] != settings.evaluation_budget:
            raise ValueError('Нарушен бюджет вычислений.')
        if not math.isfinite(run['elapsed_seconds']) or run['elapsed_seconds'] < 0:
            raise ValueError('Неверное время работы.')
        archive = []
        for payload in run['archive']:
            genome = tuple(payload['genome'])
            problem.validate_genome(genome)
            items = [alternatives[gene] for (_, alternatives), gene in zip(problem.categories, genome)]
            expected = (-sum(item.utility for item in items), sum(item.mass_g for item in items),
                        -math.prod(item.reliability_permille for item in items))
            if tuple(payload['costs']) != expected:
                raise ValueError('Критерии комплекта пересчитаны с другим результатом.')
            archive.append(Solution(genome, expected))
        if not archive or len(pareto_archive(archive)) != len(archive):
            raise ValueError('Архив пуст либо содержит дубликаты или доминируемые точки.')
        hv = hypervolume([problem.normalize(solution) for solution in archive], config['hypervolume_reference'])
        history = run['history']
        previous_evaluations, previous_hv = 0, 0.0
        for point in history:
            evaluations, value = point['evaluations'], point['hypervolume']
            if (type(evaluations) is not int or not previous_evaluations < evaluations <= settings.evaluation_budget
                    or not math.isfinite(value) or not 0 <= value <= exact_hv + 1e-12
                    or value < previous_hv - 1e-12 or type(point['archive_size']) is not int
                    or not 1 <= point['archive_size'] <= evaluations):
                raise ValueError('Некорректная траектория поиска.')
            previous_evaluations, previous_hv = evaluations, value
        if (not history or history[0]['evaluations'] != settings.population_size
                or previous_evaluations != run['evaluations'] or history[-1]['archive_size'] != len(archive)
                or not math.isclose(previous_hv, hv, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Траектория не совпадает с началом или итогом запуска.')
        restored.append(RunResult(run['method'], run['seed'], run['evaluations'], archive, [], history,
                                  run['elapsed_seconds']))
        total_points += len(archive)
    expected_identities = {(method, seed) for method in ('nsga2', 'weighted') for seed in seeds}
    if identities != expected_identities:
        raise ValueError('В серии пропущен метод или seed.')
    summary, rows = summarize(restored, exact, exact_hv)
    if summary != json.loads((directory / 'summary.json').read_text(encoding='utf-8')):
        raise ValueError('Сводная статистика не совпадает с результатами запусков.')
    check_table(directory / 'runs.csv', rows)
    check_table(directory / 'summary.csv', summary)
    check_table(directory / 'history.csv', [
        {'method': run.method, 'seed': run.seed, **point,
         'hv_percent_of_exact': 100 * point['hypervolume'] / exact_hv}
        for run in restored for point in run.history])
    check_table(directory / 'fronts.csv', [solution_row(problem, solution, method=run.method, seed=run.seed)
                                         for run in restored for solution in run.archive])
    check_table(directory / 'exact_front.csv', [solution_row(problem, solution) for solution in exact])
    demonstration = max((run for run in restored if run.method == 'nsga2'),
                        key=lambda run: run.history[-1]['hypervolume'])
    kits = json.loads((directory / 'representatives.json').read_text(encoding='utf-8'))
    if ((kits['method'], kits['seed']) != (demonstration.method, demonstration.seed)
            or kits['solutions'] != choose_representatives(problem, demonstration.archive)):
        raise ValueError('Характерные комплекты не совпадают с выбранным архивом.')
    report = {'status': 'ok', 'runs': len(saved), 'checked_archive_points': total_points,
              'evaluation_budget_per_run': settings.evaluation_budget,
              'exact_front_size': len(exact), 'exact_hypervolume': exact_hv,
              'checks': ['Хеш исходных данных', 'Допустимость и критерии каждого комплекта',
                         'Бюджеты и траектории', 'Недоминируемость архивов',
                         'Гиперобъём', 'Все методы и seed', 'Статистика и таблицы',
                         'Точный фронт', 'Характерные комплекты']}
    write_json(directory / 'verification.json', report)
    return report
