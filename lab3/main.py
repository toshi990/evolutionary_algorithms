import argparse
import json
import webbrowser
from pathlib import Path

from algorithms.genetic import Settings, run_nsga2, run_weighted, validate_weights
from algorithms.experiments import LABELS, run_series, save_results
from algorithms.problem import load_data
from algorithms.verification import verify_results


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="ЛР3, вариант 14, трек A: оборудование для экспедиции.")
    parser.add_argument("action", nargs="?", default="demo", choices=("demo", "experiment", "view", "check"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--budget", type=int, help="Общее число оценок на метод.")
    parser.add_argument("--open", action="store_true", help="Открыть страницу результатов в браузере.")
    args = parser.parse_args()
    try:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        student = config["student"]
        variant = 1 + ((17 * student["N"] + 7 * student["Q"] + student["Y"]) % 20)
        if variant != 14 or student["variant"] != 14 or student["track"] != "A":
            raise ValueError("Проект реализует вариант 14 трека A.")
        options = dict(config["algorithm"])
        if args.budget is not None:
            options["evaluation_budget"] = args.budget
        settings = Settings(**options)
        validate_weights(config["weighted_profiles"])
        if settings.evaluation_budget < len(config["weighted_profiles"]) * settings.population_size:
            raise ValueError("Недостаточно бюджета для начальных популяций всех профилей весов.")
        data_path = ROOT / "data" / "equipment.json"
        problem = load_data(data_path)
        output = ROOT / "results" / ("demo" if args.action == "demo" else "experiment")
        if args.action == "check":
            checked = verify_results(problem, output, data_path)
            print(f"Проверено запусков: {checked['runs']}; точек точного фронта: {checked['exact_front_size']}.")
            return
        if args.action in ("demo", "experiment"):
            if args.action == "demo":
                point = config["hypervolume_reference"]
                results = [run_nsga2(problem, settings, args.seed, point),
                           run_weighted(problem, settings, args.seed, config["weighted_profiles"], point)]
            else:
                results = run_series(problem, settings, config, progress=False)
            summary = save_results(problem, settings, config, results, output, data_path)
            print(f"Вариант 14, трек A: {len(problem.categories)} категорий, {sum(problem.sizes)} предметов.")
            for row in summary:
                print(f"{LABELS[row['method']]}: число запусков — {row['runs']}; средний HV/HV* {row['mean']:.3f}%.")
            representatives = json.loads((output / "representatives.json").read_text(encoding="utf-8"))["solutions"]
            for solution in representatives:
                print(f"{solution['label']}: полезность {solution['utility']}; масса {solution['mass_g']/1000:.3f} кг; надёжность {100*solution['reliability']:.2f}%.")
            if args.action == "experiment":
                checked = verify_results(problem, output, data_path)
                print(f"Проверено запусков: {checked['runs']}; точек точного фронта: {checked['exact_front_size']}.")
        page = output / "index.html"
        if not page.exists():
            raise ValueError("Сначала выполните python main.py experiment.")
        print(f"Результаты: {page}")
        if args.open or args.action == "view":
            webbrowser.open(page.as_uri())
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
