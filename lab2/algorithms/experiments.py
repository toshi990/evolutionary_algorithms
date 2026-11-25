import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev

from .genetic import METHODS, RunResult, random_search, run_ga
from .problem import exact_optimum, solution_examples
from .visualization import LABELS, write_visuals
from .report import write_report


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results, optimum):
    summary = []
    for method in dict.fromkeys(r.method for r in results):
        selected = [r for r in results if r.method == method]
        values = [r.best_coverage for r in selected]
        summary.append({"method": method, "runs": len(values), "best": max(values),
                        "mean": round(mean(values), 4), "median": median(values),
                        "std": round(stdev(values), 4) if len(values) > 1 else 0,
                        "worst": min(values), "optimal_runs": values.count(optimum),
                        "mean_gap": round(mean(optimum - value for value in values), 4),
                        "feasible_fraction": round(mean(r.feasible_evaluations / r.evaluations for r in selected), 6),
                        "mean_seconds": round(mean(r.elapsed_seconds for r in selected), 6)})
    return summary


def save_results(problem, settings, config, results, directory, graph_path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference = exact_optimum(problem)
    summary = summarize(results, reference["coverage"])
    rows, histories, solutions = [], [], []
    for result in results:
        decoded = problem.decode(result.best_genome)
        rows.append({"method": result.method, "seed": result.seed, "coverage": result.best_coverage,
                     "stations": json.dumps(decoded["stations"]), "evaluations": result.evaluations,
                     "feasible_evaluations": result.feasible_evaluations,
                     "feasible_fraction": result.feasible_evaluations / result.evaluations,
                     "seconds": result.elapsed_seconds})
        histories.extend({"method": result.method, "seed": result.seed, **point} for point in result.history)
        payload = result.to_dict(problem)
        payload.pop("history")
        solutions.append(payload)
    best = max(results, key=lambda r: r.best_coverage)
    write_csv(directory / "runs.csv", rows)
    write_csv(directory / "history.csv", histories)
    write_json(directory / "summary.json", summary)
    write_json(directory / "solutions.json", solutions)
    write_json(directory / "best_solution.json", best.to_dict(problem))
    write_json(directory / "exact_reference.json", reference)
    write_json(directory / "examples.json", solution_examples(problem))
    write_json(directory / "configuration.json", {"student": config["student"], "data": config["data"],
               "algorithm": asdict(settings), "seeds": sorted(set(r.seed for r in results)),
               "methods": list(dict.fromkeys(r.method for r in results)),
               "graph_sha256": hashlib.sha256(Path(graph_path).read_bytes()).hexdigest()})
    write_report(problem, settings, config, results, summary, reference, directory)
    write_visuals(problem, settings, results, summary, reference, directory)
    return summary


def experiment(problem, settings, config, directory, graph_path):
    seeds = config["experiments"]["seeds"]
    if len(seeds) < 20 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Для полной серии необходимо не менее 20 различных целых seed.")
    results = []
    for method in [*METHODS, "random"]:
        for seed in seeds:
            result = random_search(problem, settings, seed) if method == "random" else run_ga(problem, settings, seed, method)
            results.append(result)
        values = [r.best_coverage for r in results if r.method == method]
        print(f"{LABELS[method]}: {len(values)} запусков; среднее {mean(values):.2f}; лучшее {max(values)}", flush=True)
    save_results(problem, settings, config, results, directory, graph_path)
    return results


def verify_results(problem, directory, graph_path):
    directory = Path(directory)
    saved = json.loads((directory / 'solutions.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
    if config['graph_sha256'] != hashlib.sha256(Path(graph_path).read_bytes()).hexdigest():
        raise ValueError('Входные данные изменились после проведения эксперимента.')
    pairs = [(item['method'], item['seed']) for item in saved]
    expected = {(method, seed) for method in config['methods'] for seed in config['seeds']}
    if not saved or len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError('Состав серии не совпадает с сохранёнными методами и seed.')
    budget = config['algorithm']['evaluation_budget']
    adjacency = [{i} for i in range(problem.n)]
    for u, v in problem.edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    reference = exact_optimum(problem)
    original = json.loads((directory / 'exact_reference.json').read_text(encoding='utf-8'))
    if reference != original:
        raise ValueError('Несоответствие точному оптимуму.')
    with (directory / 'runs.csv').open(encoding='utf-8', newline='') as stream:
        runs = list(csv.DictReader(stream))
    if len(runs) != len(saved) or {(row['method'], int(row['seed'])) for row in runs} != expected:
        raise ValueError('Состав runs.csv не совпадает с серией.')
    run_map = {(row['method'], int(row['seed'])): row for row in runs}
    with (directory / 'history.csv').open(encoding='utf-8', newline='') as stream:
        history = list(csv.DictReader(stream))
    histories = {pair: [] for pair in pairs}
    for row in history:
        pair = row['method'], int(row['seed'])
        if pair not in histories:
            raise ValueError('Неизвестный запуск в history.csv.')
        histories[pair].append(row)
    restored = []
    for result in saved:
        genome = result['best_genome']
        problem.validate_genome(genome)
        selected = [i for i, bit in enumerate(genome) if bit]
        covered = set().union(*(adjacency[i] for i in selected))
        if len(selected) > problem.k or len(covered) != result['best_coverage']:
            raise ValueError('Неверное покрытие или нарушение лимита в сохранённом решении.')
        if result['solution'] != problem.decode(genome):
            raise ValueError('Расшифровка хромосомы не совпадает с решением.')
        if result['evaluations'] != budget or not 0 <= result['feasible_evaluations'] <= budget:
            raise ValueError('Нарушен бюджет вычислений.')
        if result['best_coverage'] > reference['coverage']:
            raise ValueError('Покрытие больше точного оптимума.')
        pair = result['method'], result['seed']
        row = run_map[pair]
        if (int(row['coverage']) != result['best_coverage'] or
                json.loads(row['stations']) != result['solution']['stations'] or
                int(row['evaluations']) != budget or
                int(row['feasible_evaluations']) != result['feasible_evaluations'] or
                float(row['feasible_fraction']) != result['feasible_evaluations'] / budget or
                float(row['seconds']) != result['elapsed_seconds']):
            raise ValueError('Итоги runs.csv не совпадают с сохранённым решением.')
        points = histories[pair]
        previous_evaluations, previous_best = 0, -1
        for generation, point in enumerate(points):
            evaluations, value = int(point['evaluations']), int(point['best_coverage'])
            if (int(point['generation']) != generation or
                    not previous_evaluations < evaluations <= budget or
                    not previous_best <= value <= reference['coverage']):
                raise ValueError('Некорректная траектория поиска.')
            previous_evaluations, previous_best = evaluations, value
        if (not points or int(points[0]['evaluations']) != config['algorithm']['population_size'] or
                previous_evaluations != budget or previous_best != result['best_coverage']):
            raise ValueError('Траектория не совпадает с началом или итогом запуска.')
        restored.append(RunResult(**{key: result[key] for key in RunResult.__dataclass_fields__ if key != 'history'}, history=[]))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if summary != summarize(restored, reference['coverage']):
        raise ValueError('Сводная статистика не совпадает с результатами запусков.')
    best = json.loads((directory / 'best_solution.json').read_text(encoding='utf-8'))
    candidate = {key: value for key, value in best.items() if key != 'history'}
    if candidate not in saved or best['best_coverage'] != max(item['best_coverage'] for item in saved):
        raise ValueError('Неверное лучшее решение серии.')
    report = {'status': 'ok', 'checked_solutions': len(saved), 'exact_optimum': reference['coverage'],
              'evaluation_budget': budget,
              'checks': ['Состав серии и таблицы совпадают', 'Покрытие пересчитано через множества',
                         'Все решения допустимы', 'Бюджеты и траектории проверены',
                         'Статистика пересчитана', 'Оптимум проверен полным перебором',
                         'Хеш данных совпадает']}
    write_json(directory / 'verification.json', report)
    return report
