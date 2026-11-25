import csv
from dataclasses import replace
from pathlib import Path
import statistics
from typing import Dict, Iterable, List, Sequence, Tuple

from .genetic_algorithm import GAConfig, RunResult, run_ga, run_random_search
from .objective import GLOBAL_MINIMUM


def _summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "best": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "worst": max(values),
    }


def _write_runs(path: Path, runs: Sequence[RunResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["method", "seed", "best_f", "error", "evaluations"]
            + [f"x{i}" for i in range(1, 8)]
        )
        for run in runs:
            writer.writerow(
                [
                    run.method,
                    run.seed,
                    f"{run.best_value:.12f}",
                    f"{run.best_value - GLOBAL_MINIMUM:.12f}",
                    run.evaluations,
                ]
                + [f"{coordinate:.12f}" for coordinate in run.best_vector]
            )


def _write_summary(path: Path, grouped: Dict[str, List[RunResult]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "best", "mean", "median", "std", "worst"])
        for method, runs in grouped.items():
            stats = _summary([run.best_value for run in runs])
            writer.writerow([method] + [f"{stats[name]:.12f}" for name in stats])


def _value_at(history: Sequence[Tuple[int, float]], evaluation: int) -> float:
    value = history[0][1]
    for current_evaluation, current_value in history:
        if current_evaluation > evaluation:
            break
        value = current_value
    return value


def _aggregate_histories(
    runs: Sequence[RunResult], points: int = 250
) -> List[Tuple[int, float, float, float]]:
    budget = runs[0].evaluations
    start = max(run.history[0][0] for run in runs)
    evaluations = [
        start + round(index * (budget - start) / (points - 1))
        for index in range(points)
    ]
    aggregated = []
    for evaluation in evaluations:
        values = [_value_at(run.history, evaluation) for run in runs]
        aggregated.append(
            (evaluation, min(values), statistics.fmean(values), max(values))
        )
    return aggregated


def _svg_plot(
    path: Path,
    title: str,
    series: Sequence[Tuple[str, Sequence[Tuple[int, float]], str]],
    x_label: str = "Вычисления функции",
    y_label: str = "Лучшее f(x)",
) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 90, 30, 65, 75
    all_points = [point for _, points, _ in series for point in points]
    x_min, x_max = min(p[0] for p in all_points), max(p[0] for p in all_points)
    y_min, y_max = min(p[1] for p in all_points), max(p[1] for p in all_points)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    y_padding = 0.05 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    def sx(x: float) -> float:
        return left + (x - x_min) / max(1, x_max - x_min) * (width - left - right)

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * (height - top - bottom)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">{title}</text>',
    ]
    for tick in range(6):
        x = x_min + tick * (x_max - x_min) / 5
        px = sx(x)
        lines.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{px:.1f}" y="{height-bottom+25}" text-anchor="middle" font-family="sans-serif" font-size="13">{x:.0f}</text>')
        y = y_min + tick * (y_max - y_min) / 5
        py = sy(y)
        lines.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-10}" y="{py+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{y:.1f}</text>')
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
            f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="15">{x_label}</text>',
            f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="15">{y_label}</text>',
        ]
    )
    for index, (label, points, color) in enumerate(series):
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_x = left + index * 210
        lines.append(f'<line x1="{legend_x}" y1="51" x2="{legend_x+28}" y2="51" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+35}" y="56" font-family="sans-serif" font-size="14">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiments(
    output_directory: Path,
    seeds: Iterable[int],
    base_config: GAConfig,
    sigma_a: float = 0.2,
    sigma_b: float = 0.8,
) -> Dict[str, List[RunResult]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[RunResult]] = {}
    seed_list = list(seeds)

    for sigma in (sigma_a, sigma_b):
        config = replace(base_config, mutation_sigma=sigma)
        runs = [run_ga(config, seed) for seed in seed_list]
        grouped[runs[0].method] = runs
    random_runs = [
        run_random_search(base_config.evaluation_budget, seed) for seed in seed_list
    ]
    grouped[random_runs[0].method] = random_runs

    all_runs = [run for runs in grouped.values() for run in runs]
    _write_runs(output_directory / "runs.csv", all_runs)
    _write_summary(output_directory / "summary.csv", grouped)

    colors = {"min": "#16a34a", "mean": "#2563eb", "max": "#dc2626"}
    for method, runs in grouped.items():
        if method == "Random_search":
            continue
        generations = []
        for generation, checkpoints in enumerate(zip(*(run.history for run in runs))):
            values = [value for _, value in checkpoints]
            generations.append((
                generation, checkpoints[0][0], min(values),
                statistics.fmean(values), max(values),
            ))
        with (output_directory / f"generations_{method}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["generation", "evaluations", "min", "mean", "max"])
            writer.writerows(generations)
        _svg_plot(
            output_directory / f"generations_{method}.svg",
            f"Сходимость по поколениям: σ = {runs[0].method.split('_')[-1]}",
            [
                (name, [(row[0], row[position]) for row in generations], color)
                for position, (name, color) in enumerate(colors.items(), start=2)
            ],
            x_label="Поколение",
        )
        aggregated = _aggregate_histories(runs)
        _svg_plot(
            output_directory / f"convergence_{method}.svg",
            f"Сходимость: σ = {runs[0].method.split('_')[-1]}",
            [
                (name, [(row[0], row[position]) for row in aggregated], color)
                for position, (name, color) in enumerate(colors.items(), start=1)
            ],
        )

    comparison = []
    palette = ["#2563eb", "#ea580c", "#16a34a"]
    for (method, runs), color in zip(grouped.items(), palette):
        aggregated = _aggregate_histories(runs)
        label = "Случайный поиск" if method == "Random_search" else f"ГА, σ = {method.split('_')[-1]}"
        comparison.append((label, [(row[0], row[2]) for row in aggregated], color))
    _svg_plot(
        output_directory / "methods_comparison.svg",
        f"Среднее лучшее значение по {len(seed_list)} запускам",
        comparison,
    )
    return grouped
