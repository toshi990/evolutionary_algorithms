import argparse
import json
import webbrowser
from pathlib import Path

from algorithms.experiments import experiment, save_results, verify_results
from algorithms.genetic import METHODS, Settings, run_ga
from algorithms.problem import load_problem, solution_examples
from algorithms.visualization import LABELS


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="ЛР2, вариант 14: наблюдательные пункты на графе.")
    parser.add_argument("action", nargs="?", default="demo", choices=["demo", "experiment", "examples", "check", "view"])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--method", choices=list(METHODS), default="repair_swap")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--open", action="store_true", help="Открыть результат в браузере.")
    args = parser.parse_args()
    try:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        student = config["student"]
        variant = 1 + ((17 * student["N"] + 7 * student["Q"] + student["Y"]) % 20)
        if variant != 14 or student["variant"] != variant:
            parser.error("Проект реализует вариант 14; проверьте параметры студента.")
        algorithm = dict(config["algorithm"])
        if args.budget is not None:
            algorithm["evaluation_budget"] = args.budget
        settings = Settings(**algorithm)
        graph_path = ROOT / "data" / "graph.json"
        problem = load_problem(graph_path)
        if problem.n < 20:
            raise ValueError("Граф должен содержать не менее 20 вершин.")
        output = ROOT / "results" / ("demo" if args.action == "demo" else "experiment")
        if args.action == "demo":
            result = run_ga(problem, settings, args.seed, args.method)
            save_results(problem, settings, config, [result], output, graph_path)
            solution = problem.decode(result.best_genome)
            print(f"Вариант {variant}: {problem.n} вершин, {len(problem.edges)} рёбер, лимит {problem.k} пункта.")
            print(f"{LABELS[result.method]}; seed: {result.seed}; вычислений: {result.evaluations}.")
            print(f"Пункты: {solution['stations']}; покрытие: {solution['coverage']}/{problem.n}.")
            print(f"Без покрытия: {solution['uncovered']}.")
        elif args.action == "experiment":
            experiment(problem, settings, config, output, graph_path)
            checked = verify_results(problem, output, graph_path)
            print(f"Проверено решений: {checked['checked_solutions']}; точный оптимум: {checked['exact_optimum']}.")
        elif args.action == "examples":
            for item in solution_examples(problem):
                status = "допустимо" if item["feasible"] else "превышен лимит пунктов"
                print(f"Пункты: {item['stations']}; покрытие: {item['coverage']}/{problem.n}; {status}.")
            return
        elif args.action == "check":
            checked = verify_results(problem, output, graph_path)
            print(f"Проверено решений: {checked['checked_solutions']}; точный оптимум: {checked['exact_optimum']}.")
            return
        if args.action in ("demo", "experiment", "view"):
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
