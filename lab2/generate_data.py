import json
from pathlib import Path

from algorithms.problem import generate_problem, save_problem


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    problem = generate_problem(config["data"])
    destination = root / "data" / "graph.json"
    save_problem(problem, config["data"], destination)
    print(f"Сохранён граф: {problem.n} вершин, {len(problem.edges)} рёбер. {destination}")
