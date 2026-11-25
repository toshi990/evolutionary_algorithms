from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
import sys
from typing import List, Optional, Sequence, Tuple

if __package__ in (None, ""):

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from algorithms.genetic_algorithm import (
        GAConfig,
        Individual,
        _crossover,
        _mutate,
        _random_individual,
    )
    from algorithms.objective import GLOBAL_MINIMUM, stiblinski_tang
else:
    from .genetic_algorithm import (
        GAConfig,
        Individual,
        _crossover,
        _mutate,
        _random_individual,
    )
    from .objective import GLOBAL_MINIMUM, stiblinski_tang


@dataclass
class VisualizerState:
    config: GAConfig
    seed: int
    rng: random.Random = field(init=False)
    population: List[Individual] = field(init=False)
    values: List[float] = field(init=False)
    generation: int = 0
    evaluations: int = 0
    stage: str = "initialization"
    tournament_a: List[int] = field(default_factory=list)
    tournament_b: List[int] = field(default_factory=list)
    winner_a: Optional[int] = None
    winner_b: Optional[int] = None
    children_before_mutation: Optional[Tuple[Individual, Individual]] = None
    children_after_mutation: Optional[Tuple[Individual, Individual]] = None
    offspring: List[Individual] = field(default_factory=list)
    history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.population = [
            _random_individual(self.rng) for _ in range(self.config.population_size)
        ]
        self.values = [stiblinski_tang(item) for item in self.population]
        self.evaluations = len(self.population)
        self.history = [min(self.values)]

    @property
    def best_value(self) -> float:
        return min(self.values)

    @property
    def best_index(self) -> int:
        return min(range(len(self.values)), key=self.values.__getitem__)

    def _select_tournament(self) -> Tuple[List[int], int]:
        contestants = self.rng.sample(
            range(len(self.population)), self.config.tournament_size
        )
        winner = min(contestants, key=self.values.__getitem__)
        return contestants, winner

    def step(self) -> None:
        if self.stage in ("initialization", "generation_complete"):
            self.tournament_a, self.winner_a = self._select_tournament()
            self.tournament_b = []
            self.winner_b = None
            self.children_before_mutation = None
            self.children_after_mutation = None
            self.stage = "tournament_a"
            return

        if self.stage == "tournament_a":
            self.tournament_b, self.winner_b = self._select_tournament()
            self.stage = "tournament_b"
            return

        if self.stage == "tournament_b":
            first = self.population[self.winner_a]
            second = self.population[self.winner_b]
            self.children_before_mutation = _crossover(
                first, second, self.config.crossover_probability, self.rng
            )
            self.stage = "crossover"
            return

        if self.stage == "crossover":
            assert self.children_before_mutation is not None
            self.children_after_mutation = tuple(
                _mutate(
                    child,
                    self.config.mutation_probability,
                    self.config.mutation_sigma,
                    self.rng,
                )
                for child in self.children_before_mutation
            )
            self.stage = "mutation"
            return

        if self.stage == "mutation":
            assert self.children_after_mutation is not None
            required = self.config.population_size - self.config.elite_count
            for child in self.children_after_mutation:
                if len(self.offspring) < required:
                    self.offspring.append(child)
            if len(self.offspring) >= required:
                ranked = sorted(range(len(self.population)), key=self.values.__getitem__)
                elites = [
                    self.population[index]
                    for index in ranked[: self.config.elite_count]
                ]
                elite_values = [
                    self.values[index]
                    for index in ranked[: self.config.elite_count]
                ]
                offspring_values = [stiblinski_tang(child) for child in self.offspring]
                self.evaluations += len(self.offspring)
                self.population = elites + self.offspring
                self.values = elite_values + offspring_values
                self.offspring = []
                self.generation += 1
                self.history.append(self.best_value)
                self.stage = "generation_complete"
            else:
                self.stage = "initialization"


class GeneticAlgorithmVisualizer:
    STAGE_TEXT = {
        "initialization": "Готов к турниру: нажмите «Шаг»",
        "tournament_a": "ТУРНИР 1: жёлтые особи участвуют, зелёная победила",
        "tournament_b": "ТУРНИР 2: выбран второй родитель",
        "crossover": "КРОССОВЕР: координаты родителей смешаны",
        "mutation": "МУТАЦИЯ: отдельные координаты потомков изменены",
        "generation_complete": "ЗАМЕНА: потомки образовали новое поколение",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ЛР1 — как работает генетический алгоритм")
        self.root.geometry("1180x780")
        self.root.minsize(960, 680)
        self.running = False
        self.after_id: Optional[str] = None

        self.seed_var = tk.IntVar(value=14)
        self.sigma_var = tk.DoubleVar(value=0.8)
        self.speed_var = tk.IntVar(value=600)
        self.status_var = tk.StringVar()
        self.details_var = tk.StringVar()

        self._build_controls()
        self._build_content()
        self.reset()

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.root, padding=(12, 10))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Seed:").pack(side=tk.LEFT)
        ttk.Spinbox(controls, from_=0, to=999999, width=7, textvariable=self.seed_var).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(controls, text="σ мутации:").pack(side=tk.LEFT)
        ttk.Spinbox(controls, from_=0.05, to=2.0, increment=0.05, width=6, textvariable=self.sigma_var).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(controls, text="Скорость, мс:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=100, to=1500, variable=self.speed_var, length=130).pack(side=tk.LEFT, padx=(4, 14))
        self.play_button = ttk.Button(controls, text="▶ Запустить", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Шаг", command=self.next_step).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Поколение", command=self.next_generation).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Сброс", command=self.reset).pack(side=tk.LEFT, padx=3)

    def _build_content(self) -> None:
        ttk.Label(self.root, textvariable=self.status_var, font=("TkDefaultFont", 15, "bold"), anchor=tk.CENTER).pack(fill=tk.X, padx=12, pady=(4, 8))
        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=12)

        population_frame = ttk.Labelframe(content, text="Популяция: строки — особи, столбцы — 7 генов", padding=8)
        process_frame = ttk.Labelframe(content, text="Что происходит сейчас", padding=8)
        content.add(population_frame, weight=3)
        content.add(process_frame, weight=2)

        self.population_canvas = tk.Canvas(population_frame, bg="white", highlightthickness=0)
        self.population_canvas.pack(fill=tk.BOTH, expand=True)

        self.process_canvas = tk.Canvas(process_frame, bg="white", highlightthickness=0, height=440)
        self.process_canvas.pack(fill=tk.BOTH, expand=True)
        ttk.Label(process_frame, textvariable=self.details_var, wraplength=390, justify=tk.LEFT).pack(fill=tk.X, pady=(8, 0))

        legend = ttk.Frame(self.root, padding=(12, 8))
        legend.pack(fill=tk.X)
        for color, text in (("#fde68a", "участник турнира"), ("#86efac", "победитель / родитель"), ("#ddd6fe", "потомок"), ("#fca5a5", "изменённый мутацией ген")):
            sample = tk.Canvas(legend, width=18, height=18, highlightthickness=0)
            sample.create_rectangle(1, 1, 17, 17, fill=color, outline="#374151")
            sample.pack(side=tk.LEFT, padx=(10, 4))
            ttk.Label(legend, text=text).pack(side=tk.LEFT)

    def reset(self) -> None:
        self.pause()
        config = GAConfig(population_size=30, mutation_sigma=float(self.sigma_var.get()))
        self.model = VisualizerState(config=config, seed=int(self.seed_var.get()))
        self._redraw()

    def pause(self) -> None:
        self.running = False
        self.play_button.configure(text="▶ Запустить")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def toggle_play(self) -> None:
        self.running = not self.running
        self.play_button.configure(text="⏸ Пауза" if self.running else "▶ Запустить")
        if self.running:
            self._auto_step()
        elif self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _auto_step(self) -> None:
        if not self.running:
            return
        self.next_step()
        self.after_id = self.root.after(int(self.speed_var.get()), self._auto_step)

    def next_step(self) -> None:
        self.model.step()
        self._redraw()

    def next_generation(self) -> None:
        start_generation = self.model.generation
        while self.model.generation == start_generation:
            self.model.step()
        self._redraw()

    @staticmethod
    def _gene_color(value: float) -> str:

        ratio = (value + 5.0) / 10.0
        if ratio < 0.5:
            amount = int(255 * ratio * 2)
            return f"#{amount:02x}{amount:02x}ff"
        amount = int(255 * (2.0 - ratio * 2.0))
        return f"#ff{amount:02x}{amount:02x}"

    def _redraw(self) -> None:
        stage = self.model.stage
        self.status_var.set(
            f"Поколение {self.model.generation}  •  {self.STAGE_TEXT[stage]}"
        )
        self._draw_population()
        self._draw_process()

    def _draw_population(self) -> None:
        canvas = self.population_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 600)
        row_height = 19
        label_width = 44
        fitness_width = 105
        gene_width = (width - label_width - fitness_width - 12) / 7
        active = set(self.model.tournament_a if self.model.stage == "tournament_a" else self.model.tournament_b if self.model.stage == "tournament_b" else [])
        winner = self.model.winner_a if self.model.stage == "tournament_a" else self.model.winner_b if self.model.stage == "tournament_b" else None

        for index, (individual, value) in enumerate(zip(self.model.population, self.model.values)):
            y = 8 + index * row_height
            row_fill = "#86efac" if index == winner else "#fde68a" if index in active else "white"
            canvas.create_rectangle(2, y, width - 2, y + row_height - 2, fill=row_fill, outline="#e5e7eb")
            canvas.create_text(7, y + 8, text=f"#{index + 1}", anchor=tk.W, font=("TkDefaultFont", 9))
            for gene_index, gene in enumerate(individual):
                x = label_width + gene_index * gene_width
                canvas.create_rectangle(x, y + 2, x + gene_width - 2, y + row_height - 4, fill=self._gene_color(gene), outline="")
                canvas.create_text(x + gene_width / 2 - 1, y + 8, text=f"{gene:.1f}", font=("TkDefaultFont", 8))
            canvas.create_text(width - fitness_width + 3, y + 8, text=f"f={value:.2f}", anchor=tk.W, font=("TkDefaultFont", 9, "bold" if index == self.model.best_index else "normal"))

    def _draw_individual(self, canvas: tk.Canvas, individual: Individual, x: int, y: int, title: str, mutated_from: Optional[Individual] = None) -> None:
        canvas.create_text(x, y, text=title, anchor=tk.W, font=("TkDefaultFont", 11, "bold"))
        for index, gene in enumerate(individual):
            left = x + index * 48
            changed = mutated_from is not None and abs(gene - mutated_from[index]) > 1e-12
            fill = "#fca5a5" if changed else "#ddd6fe"
            canvas.create_rectangle(left, y + 16, left + 44, y + 44, fill=fill, outline="#6b7280")
            canvas.create_text(left + 22, y + 30, text=f"{gene:.1f}", font=("TkDefaultFont", 8))

    def _draw_process(self) -> None:
        canvas = self.process_canvas
        canvas.delete("all")
        model = self.model
        canvas.create_text(12, 15, text=f"Лучшее f(x): {model.best_value:.6f}", anchor=tk.W, font=("TkDefaultFont", 13, "bold"), fill="#166534")
        canvas.create_text(12, 39, text=f"Теоретический минимум: {GLOBAL_MINIMUM:.6f}", anchor=tk.W)
        canvas.create_text(12, 59, text=f"Вычислений функции: {model.evaluations}  •  Потомков готово: {len(model.offspring)}/29", anchor=tk.W)

        if model.stage in ("tournament_a", "tournament_b"):
            contestants = model.tournament_a if model.stage == "tournament_a" else model.tournament_b
            winner = model.winner_a if model.stage == "tournament_a" else model.winner_b
            canvas.create_text(12, 100, text="Случайно выбраны три участника:", anchor=tk.W, font=("TkDefaultFont", 12, "bold"))
            for row, index in enumerate(contestants):
                is_winner = index == winner
                canvas.create_rectangle(12, 125 + row * 42, 370, 158 + row * 42, fill="#86efac" if is_winner else "#fde68a", outline="#374151")
                canvas.create_text(24, 141 + row * 42, text=f"Особь #{index + 1}: f(x) = {model.values[index]:.6f}" + ("  ← минимум, победитель" if is_winner else ""), anchor=tk.W, font=("TkDefaultFont", 10, "bold" if is_winner else "normal"))
            self.details_var.set("Турнир не создаёт вероятности из fitness. Он напрямую сравнивает три значения и выбирает наименьшее.")
        elif model.stage == "crossover" and model.children_before_mutation:
            self._draw_individual(canvas, model.population[model.winner_a], 12, 100, "Родитель 1")
            self._draw_individual(canvas, model.population[model.winner_b], 12, 165, "Родитель 2")
            self._draw_individual(canvas, model.children_before_mutation[0], 12, 245, "Потомок 1")
            self._draw_individual(canvas, model.children_before_mutation[1], 12, 310, "Потомок 2")
            self.details_var.set("Арифметический кроссовер смешивает соответствующие координаты двух победителей турниров.")
        elif model.stage == "mutation" and model.children_after_mutation and model.children_before_mutation:
            self._draw_individual(canvas, model.children_after_mutation[0], 12, 115, "Потомок 1 после мутации", model.children_before_mutation[0])
            self._draw_individual(canvas, model.children_after_mutation[1], 12, 200, "Потомок 2 после мутации", model.children_before_mutation[1])
            self.details_var.set("Красным отмечены гены, к которым добавлено гауссовское случайное изменение. Значения обрезаются до диапазона [−5, 5].")
        else:
            self._draw_history(canvas)
            self.details_var.set("Нажимайте «Шаг», чтобы увидеть каждую операцию, или «Поколение», чтобы быстро сформировать следующее поколение.")

    def _draw_history(self, canvas: tk.Canvas) -> None:
        x0, y0, width, height = 35, 115, 345, 250
        canvas.create_text(x0, y0 - 20, text="Лучшее значение по поколениям", anchor=tk.W, font=("TkDefaultFont", 12, "bold"))
        canvas.create_rectangle(x0, y0, x0 + width, y0 + height, outline="#9ca3af")
        values = self.model.history
        low = min(values + [GLOBAL_MINIMUM])
        high = max(values)
        span = max(high - low, 1.0)
        points = []
        for index, value in enumerate(values):
            px = x0 + (index / max(1, len(values) - 1)) * width
            py = y0 + (high - value) / span * height
            points.extend((px, py))
        if len(points) >= 4:
            canvas.create_line(*points, fill="#2563eb", width=3)
        elif points:
            canvas.create_oval(points[0] - 3, points[1] - 3, points[0] + 3, points[1] + 3, fill="#2563eb", outline="")
        target_y = y0 + (high - GLOBAL_MINIMUM) / span * height
        canvas.create_line(x0, target_y, x0 + width, target_y, fill="#16a34a", dash=(5, 4))
        canvas.create_text(x0 + width, target_y - 8, text="глобальный минимум", anchor=tk.E, fill="#166534")


def launch_visualizer() -> None:
    global tk, ttk
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass
    GeneticAlgorithmVisualizer(root)


    root.update_idletasks()
    root.deiconify()
    root.lift()
    try:
        root.attributes("-topmost", True)
        root.after(800, lambda: root.attributes("-topmost", False))
        root.focus_force()
    except tk.TclError:
        pass
    root.mainloop()


if __name__ == "__main__":
    launch_visualizer()
