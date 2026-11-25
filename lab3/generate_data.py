import json
from pathlib import Path

from algorithms.problem import generate_data, save_data


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    seed = config["data"]["seed"]
    problem = generate_data(seed)
    target = root / "data" / "equipment.json"
    save_data(problem, seed, target)
    print(f"Сохранено {len(problem.categories)} категорий, {sum(problem.sizes)} предметов; seed={seed}.")
