import bisect
import html
import json
import os
import tempfile
from pathlib import Path
from statistics import mean


LABELS = {"nsga2": "NSGA-II", "weighted": "Взвешенная сумма"}
COLORS = {"nsga2": "#087e8b", "weighted": "#9b4d96"}


def convergence_data(results, exact_hv):
    grid = sorted({row["evaluations"] for run in results for row in run.history})
    grid = [value for value in grid if value >= max(run.history[0]["evaluations"] for run in results)]
    groups = {}
    for method in dict.fromkeys(run.method for run in results):
        selected = [run for run in results if run.method == method]
        series = []
        for run in selected:
            checkpoints = [point["evaluations"] for point in run.history]
            series.append([100 * run.history[bisect.bisect_right(checkpoints, value) - 1]["hypervolume"] / exact_hv for value in grid])
        groups[method] = {"mean": [mean(values) for values in zip(*series)],
                          "min": [min(values) for values in zip(*series)],
                          "max": [max(values) for values in zip(*series)]}
    return grid, groups


def best_runs(results):
    return {method: max((run for run in results if run.method == method), key=lambda run: run.history[-1]["hypervolume"])
            for method in dict.fromkeys(run.method for run in results)}


def write_figures(problem, results, exact, exact_hv, directory):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "evolutionary-lab3-matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        write_svg_fallback(problem, results, exact, exact_hv, directory)
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18,
                         "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white"})

    def save(fig, name):
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    selected = best_runs(results)
    projections = [("Масса, кг", "Полезность, баллы", lambda s: (s.costs[1] / 1000, -s.costs[0])),
                   ("Масса, кг", "Надёжность, %", lambda s: (s.costs[1] / 1000, -100 * s.costs[2] / problem.denominator)),
                   ("Полезность, баллы", "Надёжность, %", lambda s: (-s.costs[0], -100 * s.costs[2] / problem.denominator))]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    for ax, (xlabel, ylabel, transform) in zip(axes, projections):
        x, y = zip(*(transform(solution) for solution in exact))
        ax.scatter(x, y, s=17, color="#b8c0ca", alpha=0.5, label=f"Точный фронт ({len(exact)})", linewidths=0)
        for method, run in selected.items():
            x, y = zip(*(transform(solution) for solution in run.archive))
            ax.scatter(x, y, s=12, color=COLORS[method], alpha=0.6, label=f"{LABELS[method]}, seed {run.seed}", linewidths=0)
        ax.set(xlabel=xlabel, ylabel=ylabel)
    axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("Три проекции трёхмерного фронта Парето", fontsize=14)
    save(fig, "pareto")

    grid, groups = convergence_data(results, exact_hv)
    fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
    for method, values in groups.items():
        ax.fill_between(grid, values["min"], values["max"], color=COLORS[method], alpha=0.12)
        ax.plot(grid, values["mean"], color=COLORS[method], linewidth=2.1, label=LABELS[method])
    ax.axhline(100, color="#344054", linestyle="--", linewidth=1, label="Точный гиперобъём")
    ax.set(xlabel="Число оценок трёх критериев", ylabel="Доля точного гиперобъёма, %",
           title="Сходимость: среднее и диапазон min–max", xlim=(0, max(grid)))
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "convergence")

    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    methods = list(selected)
    data = [[100 * run.history[-1]["hypervolume"] / exact_hv for run in results if run.method == method] for method in methods]
    boxes = ax.boxplot(data, tick_labels=[LABELS[method] for method in methods], patch_artist=True, showmeans=True)
    for box, method in zip(boxes["boxes"], methods):
        box.set_facecolor(COLORS[method])
        box.set_alpha(0.25)
    for index, values in enumerate(data, 1):
        offsets = [0.055 * ((i % 5) - 2) for i in range(len(values))]
        ax.scatter([index + offset for offset in offsets], values, s=15, color=COLORS[methods[index-1]], alpha=0.6)
    ax.set(ylabel="Доля точного гиперобъёма, %", title="Итоговое качество независимых запусков")
    save(fig, "quality")

    write_flow(directory)


def svg(width, height, body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" fill="#172c3d">{body}</g></svg>'


def write_svg_fallback(problem, results, exact, exact_hv, directory):
    selected = best_runs(results)
    projections = [(lambda s: (s.costs[1] / 1000, -s.costs[0]), "Масса, кг", "Полезность"),
                   (lambda s: (s.costs[1] / 1000, -100*s.costs[2]/problem.denominator), "Масса, кг", "Надёжность, %"),
                   (lambda s: (-s.costs[0], -100*s.costs[2]/problem.denominator), "Полезность", "Надёжность, %")]
    body = '<text x="20" y="25" font-size="18">Три проекции фронта Парето</text>'
    for panel, (transform, xlabel, ylabel) in enumerate(projections):
        points = [transform(solution) for solution in exact]
        lo_x, hi_x = min(p[0] for p in points), max(p[0] for p in points)
        lo_y, hi_y = min(p[1] for p in points), max(p[1] for p in points)
        x = lambda value: panel*410+55+320*(value-lo_x)/max(1e-12, hi_x-lo_x)
        y = lambda value: 340-260*(value-lo_y)/max(1e-12, hi_y-lo_y)
        body += f'<path d="M{panel*410+55} 65 V340 H{panel*410+385}" fill="none" stroke="#8899aa"/>'
        for value in range(5):
            tx, ty = lo_x+(hi_x-lo_x)*value/4, lo_y+(hi_y-lo_y)*value/4
            body += f'<text x="{x(tx):.1f}" y="359" text-anchor="middle" font-size="11">{tx:.1f}</text><text x="{panel*410+50}" y="{y(ty)+4:.1f}" text-anchor="end" font-size="11">{ty:.1f}</text>'
        body += f'<text x="{panel*410+200}" y="388" text-anchor="middle" font-size="13">{xlabel}</text><text x="{panel*410+50}" y="54" font-size="13">{ylabel}</text>'
        for archive, color in [(exact, "#b8c0ca"), *[(run.archive, COLORS[method]) for method, run in selected.items()]]:
            for solution in archive:
                a, b = transform(solution)
                body += f'<circle cx="{x(a):.2f}" cy="{y(b):.2f}" r="2.1" fill="{color}" opacity="0.6"/>'
    body += '<text x="30" y="420" font-size="12">Серый — точный фронт; бирюзовый — NSGA-II; фиолетовый — взвешенная сумма.</text>'
    (directory / "pareto.svg").write_text(svg(1240, 440, body), encoding="utf-8")
    grid, groups = convergence_data(results, exact_hv)
    low = min(min(values["min"]) for values in groups.values()) - 1
    x = lambda value: 70+760*value/max(grid)
    y = lambda value: 370-300*(value-low)/(101-low)
    body = '<text x="25" y="25" font-size="18">Сходимость: среднее и диапазон min–max</text>'
    for tick in range(6):
        value = low+(101-low)*tick/5
        body += f'<path d="M70 {y(value):.1f} H830" stroke="#dde5ec"/><text x="60" y="{y(value)+4:.1f}" text-anchor="end" font-size="11">{value:.1f}</text>'
    body += f'<path d="M70 {y(100):.1f} H830" stroke="#344054" stroke-dasharray="6 4"/>'
    for tick in range(5):
        value = round(max(grid) * tick / 4)
        body += f'<text x="{x(value):.1f}" y="392" text-anchor="middle" font-size="11">{value}</text>'
    for method, values in groups.items():
        coordinate = lambda curve: " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(grid, curve))
        polygon = coordinate(values["min"])+" "+" ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in reversed(list(zip(grid, values["max"]))))
        body += f'<polygon points="{polygon}" fill="{COLORS[method]}" opacity="0.12"/><polyline points="{coordinate(values["mean"])}" fill="none" stroke="{COLORS[method]}" stroke-width="2"/>'
    for index, method in enumerate(groups):
        body += f'<text x="80" y="{450+25*index}" fill="{COLORS[method]}" font-size="13">{LABELS[method]}</text>'
    body += f'<text x="340" y="418" font-size="13">Число оценок критериев</text><text x="70" y="55" font-size="12">Доля точного гиперобъёма, %</text>'
    (directory / "convergence.svg").write_text(svg(900, 500, body), encoding="utf-8")
    methods = list(selected)
    values_by_method = {method: [100 * run.history[-1]["hypervolume"] / exact_hv
                                for run in results if run.method == method] for method in methods}
    all_values = [value for values in values_by_method.values() for value in values]
    lo, hi = min(all_values) - 0.3, max(all_values) + 0.3
    py = lambda value: 330 - 260 * (value - lo) / (hi - lo)
    body = '<text x="25" y="28" font-size="18">Итоговое качество запусков</text>'
    for tick in range(6):
        value = lo + tick * (hi - lo) / 5
        body += f'<path d="M75 {py(value):.1f} H760" stroke="#e0e6eb"/><text x="65" y="{py(value)+4:.1f}" text-anchor="end" font-size="12">{value:.2f}</text>'
    for index, method in enumerate(methods):
        px = 260 + index * 300
        for run_index, value in enumerate(values_by_method[method]):
            offset = 11 * ((run_index % 5) - 2)
            body += f'<circle cx="{px+offset}" cy="{py(value):.1f}" r="4" fill="{COLORS[method]}" opacity="0.65"/>'
        average = py(mean(values_by_method[method]))
        body += f'<path d="M{px-45} {average:.1f} H{px+45}" stroke="{COLORS[method]}" stroke-width="3"/><text x="{px}" y="365" text-anchor="middle" font-size="14">{LABELS[method]}</text>'
    body += '<text x="75" y="53" font-size="12">Доля точного гиперобъёма, %</text><text x="100" y="404" font-size="12">Точки — отдельные запуски; горизонтальный отрезок — среднее.</text>'
    (directory / "quality.svg").write_text(svg(820, 430, body), encoding="utf-8")
    write_flow(directory)
    for name in ("pareto", "convergence", "quality", "flow"):
        (directory / f"{name}.png").unlink(missing_ok=True)


def write_flow(directory):
    body = '<text x="25" y="28" font-size="19">NSGA-II с внешним архивом</text>'
    body += '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10" fill="#496976"/></marker></defs>'
    labels = [(55, "Случайные допустимые комплекты"),
              (120, "Оценка критериев; обновление архива"),
              (320, "Сортировка фронтов и расстояние до скопления"),
              (385, "Турнир, кроссовер, категориальная мутация"),
              (450, "Оценка потомков; обновление архива"),
              (515, "Отбор из родителей и потомков")]
    for yy, label in labels:
        body += f'<rect x="65" y="{yy}" width="465" height="42" rx="5" fill="#edf7f7" stroke="#8fbcc0"/><text x="297" y="{yy+26}" text-anchor="middle" font-size="13">{label}</text>'
    body += '<path d="M297 97 V120 M297 162 V185 M297 362 V385 M297 427 V450 M297 492 V515" fill="none" stroke="#496976" marker-end="url(#arrow)"/>'
    body += '<path d="M297 185 L447 232 L297 279 L147 232 Z" fill="#edf7f7" stroke="#8fbcc0"/><text x="297" y="237" text-anchor="middle" font-size="13">Бюджет исчерпан?</text>'
    body += '<path d="M297 279 V320" stroke="#496976" marker-end="url(#arrow)"/><text x="308" y="303" font-size="12">нет</text>'
    body += '<path d="M447 232 H655 V278" fill="none" stroke="#496976" marker-end="url(#arrow)"/><text x="480" y="222" font-size="12">да</text><rect x="563" y="278" width="184" height="65" rx="5" fill="#edf7f7" stroke="#8fbcc0"/><text x="655" y="304" text-anchor="middle" font-size="13">Итоговый архив</text><text x="655" y="326" text-anchor="middle" font-size="13">компромиссов</text>'
    body += '<path d="M297 557 V585 H30 V232 H147" fill="none" stroke="#496976" marker-end="url(#arrow)"/>'
    (directory / "flow.svg").write_text(svg(780, 610, body), encoding="utf-8")


def write_index(problem, summary, representatives, exact_size, directory):
    rows = "".join(f'<tr><td>{LABELS[s["method"]]}</td><td>{s["runs"]}</td><td>{s["mean"]:.3f}%</td><td>{s["worst"]:.3f}%</td><td>{s["std"]:.3f}</td><td>{s["mean_archive_size"]:.0f}</td></tr>' for s in summary)
    options = "".join(f'<option value="{i}">{html.escape(r["label"])}</option>' for i, r in enumerate(representatives))
    data = json.dumps(representatives, ensure_ascii=False).replace("<", "\\u003c")
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ЛР3 — оборудование экспедиции</title>
<style>body{{margin:0;background:#f1f5f6;color:#17333e;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1150px;margin:auto;padding:32px}}section{{background:white;padding:25px;border-radius:14px;margin:20px 0}}h1{{font-size:30px}}h2{{font-size:22px}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce7eb;text-align:left}}select{{font:inherit;padding:8px;border:1px solid #97b8bd;border-radius:8px}}a{{color:#087e8b}}.scroll{{overflow:auto}}.number{{font-size:24px;font-weight:700}}</style>
<main><h1>Оборудование для экспедиции</h1><p>ЛР3 · вариант 14 · трек A · Б24-517</p><section><p>Выбираем по одному предмету из десяти категорий. Ищем большую полезность и надёжность при меньшей массе.</p><p>Точный перебор {problem.search_space_size:,} комплектов дал <b>{exact_size}</b> различных компромиссов. Среди них нельзя улучшить один критерий без ухудшения другого.</p></section>
<section><h2>Сравнение методов</h2><p>Каждый полный запуск получает одинаковое суммарное число оценок критериев. HV/HV* — доля точного гиперобъёма, а не доля найденных комплектов.</p><div class="scroll"><table><tr><th>Метод</th><th>Запусков</th><th>Среднее HV/HV*</th><th>Худшее</th><th>σ, п.п.</th><th>Средний архив</th></tr>{rows}</table></div></section>
<section><h2>Компромиссы по трём критериям</h2><img src="pareto.svg" alt="Три попарные проекции фронта Парето"><p>Третий критерий скрыт в каждой проекции. Визуально худшая по двум осям точка может выигрывать по третьей.</p></section>
<section><h2>Изменение качества поиска</h2><img src="convergence.svg" alt="Средний гиперобъём и min–max по запускам"></section>
<section><h2>Примеры комплектов</h2><select id="kit">{options}</select><p id="metrics" class="number"></p><div class="scroll"><table><thead><tr><th>Предмет</th><th>Полезность</th><th>Масса, г</th><th>Надёжность</th></tr></thead><tbody id="items"></tbody></table></div><p>Вероятности и полезность синтетические. Надёжность комплекта считается произведением вероятностей исправности всех предметов.</p></section>
<section><h2>Материалы</h2><a href="report.md">Отчёт</a> · <a href="runs.csv">Серия запусков</a> · <a href="fronts.csv">Найденные фронты</a> · <a href="exact_front.csv">Точный фронт</a> · <a href="configuration.json">Параметры</a></section></main>
<script>const kits={data}; const select=document.getElementById('kit'); function showKit(){{const kit=kits[Number(select.value)];document.getElementById('metrics').textContent=kit.utility+' баллов · '+(kit.mass_g/1000).toFixed(3)+' кг · '+(100*kit.reliability).toFixed(2)+'%';const target=document.getElementById('items');target.replaceChildren();for(const item of kit.items){{const row=document.createElement('tr');for(const value of [item.name,item.utility,item.mass_g,(item.reliability_permille/1000).toFixed(3)]){{const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell);}}target.appendChild(row);}}}}select.addEventListener('change',showKit);showKit();</script></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")
