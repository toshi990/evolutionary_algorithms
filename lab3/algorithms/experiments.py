import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev

from .genetic import run_nsga2, run_weighted
from .pareto import hypervolume, pareto_archive
from .problem import enumerate_solutions
from .report import write_report


LABELS = {"nsga2": "NSGA-II", "weighted": "Взвешенная сумма"}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Пустая таблица результатов.")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_front(problem, point):
    if problem.search_space_size > 1_000_000:
        raise ValueError("Точный перебор ограничен миллионом комплектов; уменьшите экземпляр.")
    front = pareto_archive(enumerate_solutions(problem))
    return front, hypervolume([problem.normalize(solution) for solution in front], point)


def run_series(problem, settings, config, progress=True):
    seeds = config["experiments"]["seeds"]
    if len(seeds) < 20 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Для эксперимента нужны как минимум 20 различных целых seed.")
    results = []
    for method in ("nsga2", "weighted"):
        for index, seed in enumerate(seeds):
            result = (run_nsga2(problem, settings, seed, config["hypervolume_reference"]) if method == "nsga2"
                      else run_weighted(problem, settings, seed, config["weighted_profiles"], config["hypervolume_reference"]))
            results.append(result)
            if progress and (index + 1) % 5 == 0:
                print(f"{LABELS[method]}: {index + 1}/{len(seeds)} запусков", flush=True)
    return results


def summarize(results, reference, reference_hv):
    exact_costs = {solution.costs for solution in reference}
    summary, rows = [], []
    for run in results:
        hv = run.history[-1]["hypervolume"]
        exact_hits = sum(solution.costs in exact_costs for solution in run.archive)
        rows.append({"method": run.method, "seed": run.seed, "evaluations": run.evaluations,
                     "hypervolume": hv, "hv_percent_of_exact": 100 * hv / reference_hv,
                     "archive_size": len(run.archive), "globally_pareto_points": exact_hits,
                     "exact_front_recall_percent": 100 * exact_hits / len(reference),
                     "seconds": run.elapsed_seconds})
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        values = [row["hv_percent_of_exact"] for row in selected]
        summary.append({"method": method, "runs": len(values), "best": max(values), "mean": mean(values),
                        "median": median(values), "std": stdev(values) if len(values) > 1 else 0.0,
                        "worst": min(values), "mean_archive_size": mean(row["archive_size"] for row in selected),
                        "mean_exact_front_recall_percent": mean(row["exact_front_recall_percent"] for row in selected),
                        "mean_seconds": mean(row["seconds"] for row in selected)})
    return summary, rows


def choose_representatives(problem, archive):
    criteria = [("Лёгкий", lambda s: (s.costs[1], s.costs)),
                ("Полезный", lambda s: (s.costs[0], s.costs)),
                ("Надёжный", lambda s: (s.costs[2], s.costs)),
                ("Сбалансированный", lambda s: (sum(x * x for x in problem.normalize(s)), s.costs))]
    chosen, used = [], set()
    for name, key in criteria:
        candidates = [solution for solution in archive if solution.costs not in used]
        if candidates:
            solution = min(candidates, key=key)
            used.add(solution.costs)
            chosen.append({"label": name, **problem.decode(solution)})
    return chosen


def solution_row(problem, solution, **metadata):
    decoded = problem.decode(solution)
    return {**metadata, "genome": " ".join(map(str, solution.genome)), "utility": decoded["utility"],
            "mass_g": decoded["mass_g"], "reliability": decoded["reliability"],
            "reliability_numerator": decoded["reliability_numerator"]}


def save_results(problem, settings, config, results, directory, data_path):
    from .plots import write_figures, write_index
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exact, exact_hv = reference_front(problem, config["hypervolume_reference"])
    summary, rows = summarize(results, exact, exact_hv)
    candidate_runs = [run for run in results if run.method == "nsga2"] or results
    demonstration = max(candidate_runs, key=lambda run: run.history[-1]["hypervolume"])
    representatives = choose_representatives(problem, demonstration.archive)
    write_json(directory / "runs.json", [{"method": run.method, "seed": run.seed,
               "evaluations": run.evaluations, "elapsed_seconds": run.elapsed_seconds,
               "archive": [asdict(solution) for solution in run.archive], "history": run.history} for run in results])
    write_csv(directory / "runs.csv", rows)
    write_csv(directory / "summary.csv", summary)
    write_csv(directory / "history.csv", [{"method": run.method, "seed": run.seed, **point,
               "hv_percent_of_exact": 100 * point["hypervolume"] / exact_hv} for run in results for point in run.history])
    write_csv(directory / "fronts.csv", [solution_row(problem, solution, method=run.method, seed=run.seed)
                                         for run in results for solution in run.archive])
    write_csv(directory / "exact_front.csv", [solution_row(problem, solution) for solution in exact])
    write_json(directory / "summary.json", summary)
    write_json(directory / "representatives.json", {"method": demonstration.method, "seed": demonstration.seed,
               "selection": "Запуск NSGA-II с наибольшим гиперобъёмом; четыре различных комплекта.",
               "solutions": representatives})
    write_json(directory / "reference.json", {"combinations": problem.search_space_size, "front_size": len(exact),
               "hypervolume": exact_hv, "point": config["hypervolume_reference"]})
    write_json(directory / "configuration.json", {"student": config["student"], "data": config["data"],
               "algorithm": asdict(settings), "weighted_profiles": config["weighted_profiles"],
               "hypervolume_reference": config["hypervolume_reference"],
               "seeds": sorted({run.seed for run in results}),
               "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest()})
    write_figures(problem, results, exact, exact_hv, directory)
    write_index(problem, summary, representatives, len(exact), directory)
    write_report(problem, settings, config, results, summary, representatives, exact, exact_hv, demonstration, directory)
    return summary
