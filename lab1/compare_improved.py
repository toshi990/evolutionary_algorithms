import csv
from pathlib import Path
import statistics

from algorithms.genetic_algorithm import GAConfig, run_ga
from algorithms.objective import GLOBAL_MINIMUM


def main() -> None:
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    configurations = {
        "baseline": GAConfig(
            population_size=100,
            evaluation_budget=50_000,
            mutation_sigma=0.8,
        ),
        "hybrid_reset_0.05": GAConfig(
            population_size=100,
            evaluation_budget=50_000,
            mutation_sigma=0.8,
            random_reset_probability=0.05,
        ),
    }
    rows = []
    for name, config in configurations.items():
        for seed in range(20):
            result = run_ga(config, seed)
            rows.append((name, seed, result.best_value, result.best_value - GLOBAL_MINIMUM,
                         result.evaluations, *result.best_vector))

    with (output / "improved_runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "seed", "best_f", "error", "evaluations"]
                        + [f"x{i}" for i in range(1, 8)])
        writer.writerows(rows)

    with (output / "improved_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "best", "mean", "median", "std", "worst", "success_error_below_0.01"])
        for name in configurations:
            values = [row[2] for row in rows if row[0] == name]
            success = sum(value - GLOBAL_MINIMUM < 0.01 for value in values)
            writer.writerow([
                name,
                min(values),
                statistics.fmean(values),
                statistics.median(values),
                statistics.stdev(values),
                max(values),
                success,
            ])
            print(
                f"{name}: среднее={statistics.fmean(values):.9f}, "
                f"медиана={statistics.median(values):.9f}, ошибка < 0.01: {success}/20"
            )


if __name__ == "__main__":
    main()
