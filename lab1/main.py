import argparse
from pathlib import Path

from algorithms.experiment import run_experiments
from algorithms.genetic_algorithm import GAConfig, run_ga
from algorithms.objective import GLOBAL_MINIMUM


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ЛР1, вариант 14: функция Стыблинского—Танга")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="один запуск ГА")
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--population", type=int, default=100)
    single.add_argument("--budget", type=int, default=50_000)
    single.add_argument("--sigma", type=float, default=0.2)

    experiment = subparsers.add_parser("experiment", help="серия сравнительных запусков")
    experiment.add_argument("--runs", type=int, default=20)
    experiment.add_argument("--population", type=int, default=100)
    experiment.add_argument("--budget", type=int, default=50_000)
    experiment.add_argument("--sigma-a", type=float, default=0.2)
    experiment.add_argument("--sigma-b", type=float, default=0.8)
    experiment.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    improved = subparsers.add_parser(
        "improved", help="ГА с гибридной мутацией"
    )
    improved.add_argument("--seed", type=int, default=0)
    improved.add_argument("--population", type=int, default=100)
    improved.add_argument("--budget", type=int, default=50_000)
    improved.add_argument("--sigma", type=float, default=0.8)
    improved.add_argument("--reset-probability", type=float, default=0.05)
    subparsers.add_parser("visualize", help="открыть визуализацию")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "visualize":
        from visualize import main as launch_visualizer

        launch_visualizer()
        return

    config = GAConfig(
        population_size=args.population,
        evaluation_budget=args.budget,
        mutation_sigma=getattr(args, "sigma", 0.2),
        random_reset_probability=getattr(args, "reset_probability", 0.0),
    )
    try:
        config.validate()
    except ValueError as Ошибка:
        parser.error(str(error))

    if args.command in ("single", "improved"):
        result = run_ga(config, args.seed)
        print(f"Seed: {result.seed}")
        print(f"Найденное f(x): {result.best_value:.12f}")
        print(f"Известный минимум: {GLOBAL_MINIMUM:.12f}")
        print(f"Ошибка: {result.best_value - GLOBAL_MINIMUM:.12f}")
        print("Координаты:", " ".join(f"{x:.9f}" for x in result.best_vector))
        print(f"Вычислений функции: {result.evaluations}")
        return

    if args.runs < 20:
        parser.error("Для серии укажите не менее 20 запусков")
    from dataclasses import replace

    try:
        replace(config, mutation_sigma=args.sigma_a).validate()
        replace(config, mutation_sigma=args.sigma_b).validate()
    except ValueError as Ошибка:
        parser.error(str(error))
    if args.sigma_a == args.sigma_b:
        parser.error("Укажите два разных масштаба мутации")
    grouped = run_experiments(
        args.output,
        range(args.runs),
        config,
        sigma_a=args.sigma_a,
        sigma_b=args.sigma_b,
    )
    print(f"Результаты: {args.output.resolve()}")
    for method, runs in grouped.items():
        mean = sum(run.best_value for run in runs) / len(runs)
        print(f"{method}: среднее лучшее f(x) = {mean:.9f}")


if __name__ == "__main__":
    main()
