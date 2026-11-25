import html
import math
from pathlib import Path
from statistics import mean


LABELS = {"penalty_flip": "Штраф + инверсия бита", "repair_flip": "Ремонт + инверсия бита",
          "repair_swap": "Ремонт + обмен", "random": "Случайный допустимый поиск"}
COLORS = {"penalty_flip": "#b54708", "repair_flip": "#0969a2", "repair_swap": "#13805e", "random": "#7f56a5"}


def svg_document(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
            '<g font-family="Arial, sans-serif" fill="#172c3d">' + body + '</g></svg>')


def graph_svg(problem, genome):
    covered = problem.covered_mask(genome)
    positions = problem.positions or tuple((math.cos(2 * math.pi * i / problem.n),
                                           math.sin(2 * math.pi * i / problem.n)) for i in range(problem.n))
    points = [(300 + 245 * x, 310 + 245 * y) for x, y in positions]
    lines = [f'<text x="28" y="30" font-size="19">Пункты наблюдения и покрытие графа</text>']
    for u, v in problem.edges:
        x1, y1 = points[u]
        x2, y2 = points[v]
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#c6d1db" stroke-width="1.1"/>')
    for i, (x, y) in enumerate(points):
        color = "#174a72" if genome[i] else ("#bee4cf" if covered & (1 << i) else "#f3d4d3")
        text_color = "white" if genome[i] else "#172c3d"
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{color}" stroke="white" stroke-width="2"/>')
        lines.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="11" fill="{text_color}">{i+1}</text>')
    for y, color, title in [(130, "#174a72", "Пункт наблюдения"), (175, "#bee4cf", "Покрытая вершина"), (220, "#f3d4d3", "Нет покрытия")]:
        lines.append(f'<circle cx="635" cy="{y}" r="11" fill="{color}"/><text x="657" y="{y+5}" font-size="14">{title}</text>')
    lines.append(f'<text x="620" y="295" font-size="17">Пунктов: {sum(genome)} из {problem.k}</text>')
    lines.append(f'<text x="620" y="328" font-size="17">Покрытие: {problem.coverage(genome)} / {problem.n}</text>')
    lines.append('<text x="620" y="375" font-size="13">Покрывается сама вершина</text><text x="620" y="395" font-size="13">и её непосредственные соседи.</text>')
    return svg_document(890, 610, "".join(lines))


def convergence_svg(results, optimum):
    groups = {}
    for result in results:
        groups.setdefault(result.method, []).append(result)
    rows = [point for result in results for point in result.history]
    left, right, top, bottom = 72, 870, 70, 415
    ymax, ymin = optimum + 1, max(0, min(r["best_coverage"] for r in rows) - 1)
    budget = max(r["evaluations"] for r in rows)
    x = lambda value: left + (right - left) * value / budget
    y = lambda value: bottom - (bottom - top) * (value - ymin) / max(1, ymax - ymin)
    body = ['<text x="26" y="30" font-size="19">Сходимость: среднее и диапазон min–max по запускам</text>']
    for tick in range(ymin, ymax + 1):
        body.append(f'<line x1="{left}" y1="{y(tick):.1f}" x2="{right}" y2="{y(tick):.1f}" stroke="#e3eaf0"/><text x="56" y="{y(tick)+5:.1f}" text-anchor="end" font-size="12">{tick}</text>')
    for tick in range(0, budget + 1, max(1, budget // 4)):
        body.append(f'<text x="{x(tick):.1f}" y="440" text-anchor="middle" font-size="12">{tick}</text>')
    body.append(f'<line x1="{left}" y1="{y(optimum):.1f}" x2="{right}" y2="{y(optimum):.1f}" stroke="#222" stroke-dasharray="7 5"/>')
    body.append('<text x="630" y="57" font-size="12">Пунктир — точный оптимум</text>')
    for method, runs in groups.items():
        checkpoints = [p["evaluations"] for p in runs[0].history]
        maps = [{p["evaluations"]: p["best_coverage"] for p in r.history} for r in runs]
        values = [[m[checkpoint] for m in maps] for checkpoint in checkpoints]
        coordinates = lambda ys: " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(checkpoints, ys))
        lower = [min(v) for v in values]
        upper = [max(v) for v in values]
        polygon = coordinates(lower) + " " + " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in reversed(list(zip(checkpoints, upper))))
        color = COLORS[method]
        body.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.09"/>')
        body.append(f'<polyline points="{coordinates([mean(v) for v in values])}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    body.append('<text x="350" y="475" font-size="14">Число вычислений целевой функции</text>')
    body.append('<text x="12" y="58" font-size="12">Покрытие</text>')
    for i, method in enumerate(groups):
        yy = 512 + i * 25
        body.append(f'<line x1="80" y1="{yy}" x2="110" y2="{yy}" stroke="{COLORS[method]}" stroke-width="3"/><text x="122" y="{yy+4}" font-size="13">{LABELS[method]}</text>')
    return svg_document(920, 625, "".join(body))


def flow_svg(population_size):
    body = ['<text x="30" y="30" font-size="19">Цикл генетического алгоритма</text>']
    labels = [(55, f"Инициализация: {population_size} допустимых особей"),
              (121, "Оценка покрытия и сохранение лучшего"),
              (322, "Турнирный отбор и перенос элиты"),
              (388, "Равномерный кроссовер и мутация"),
              (454, "Ремонт, если выбран; оценка потомков")]
    for yy, label in labels:
        body.append(f'<rect x="70" y="{yy}" width="440" height="42" rx="6" fill="#eef4f8" stroke="#9eb7cb"/><text x="290" y="{yy+26}" text-anchor="middle" font-size="13">{label}</text>')
    body.append('<path d="M290 97 V121 M290 163 V190 M290 364 V388 M290 430 V454" fill="none" stroke="#607b90" marker-end="url(#arrow)"/>')
    body.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10" fill="#607b90"/></marker></defs>')
    body.append('<path d="M290 190 L440 238 L290 286 L140 238 Z" fill="#eef4f8" stroke="#9eb7cb"/><text x="290" y="243" text-anchor="middle" font-size="13">Бюджет исчерпан?</text>')
    body.append('<path d="M290 286 V322" fill="none" stroke="#607b90" marker-end="url(#arrow)"/><text x="300" y="309" font-size="12">нет</text>')
    body.append('<path d="M440 238 H650 V278" fill="none" stroke="#607b90" marker-end="url(#arrow)"/><text x="470" y="228" font-size="12">да</text>')
    body.append('<rect x="560" y="278" width="180" height="62" rx="6" fill="#eef4f8" stroke="#9eb7cb"/><text x="650" y="303" text-anchor="middle" font-size="13">Лучшее допустимое</text><text x="650" y="324" text-anchor="middle" font-size="13">решение</text>')
    body.append('<path d="M290 496 V520 H35 V238 H140" fill="none" stroke="#607b90" marker-end="url(#arrow)"/>')
    return svg_document(780, 545, "".join(body))

def write_visuals(problem, settings, results, summary, reference, directory):
    directory = Path(directory)
    best = max(results, key=lambda result: result.best_coverage)
    (directory / "graph.svg").write_text(graph_svg(problem, best.best_genome), encoding="utf-8")
    (directory / "convergence.svg").write_text(convergence_svg(results, reference["coverage"]), encoding="utf-8")
    (directory / "flow.svg").write_text(flow_svg(settings.population_size), encoding="utf-8")
    table = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in [LABELS[s["method"]], s["runs"], s["best"], s["mean"], s["median"], s["std"], s["worst"], s["optimal_runs"]]) + "</tr>" for s in summary)
    solution = problem.decode(best.best_genome)
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ЛР2 — наблюдательные пункты</title><style>body{{font:16px/1.6 system-ui,sans-serif;color:#172c3d;background:#f2f5f8;margin:0}}main{{max-width:1050px;margin:auto;padding:32px}}section{{background:white;border-radius:14px;padding:24px;margin:18px 0}}h1{{font-size:29px}}h2{{font-size:21px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dde5eb;text-align:left}}.scroll{{overflow:auto}}a{{color:#0969a2}}.metric{{font-size:28px;font-weight:700}}</style>
<main><h1>Расстановка наблюдательных пунктов</h1><p>Лабораторная работа №2 · вариант 14 · Б24-517</p>
<section><span class="metric">{best.best_coverage} / {problem.n}</span><p>Лучшее покрытие. Точный оптимум: {reference["coverage"]}. Пункты: {solution["stations"]}. Лимит: {problem.k}.</p><p>Покрытие означает наблюдение самой вершины и соседей на расстоянии одного ребра.</p></section>
<section><h2>Сравнение методов</h2><p>Каждый запуск получает одинаковый бюджет вычислений функции. Столбец «Оптимум» показывает число запусков, достигших точного оптимума.</p><div class="scroll"><table><tr><th>Метод</th><th>Запусков</th><th>Лучшее</th><th>Среднее</th><th>Медиана</th><th>σ</th><th>Худшее</th><th>Оптимум</th></tr>{table}</table></div></section>
<section><h2>Динамика поиска</h2><img src="convergence.svg" alt="Среднее лучшее покрытие и диапазон по запускам"></section>
<section><h2>Лучшее найденное решение</h2><img src="graph.svg" alt="Выбранные пункты и покрытые вершины"><p>Без покрытия: {solution["uncovered"]}.</p></section>
<section><h2>Материалы</h2><p><a href="report.md">Отчёт</a> · <a href="runs.csv">Все запуски</a> · <a href="history.csv">Траектории</a> · <a href="best_solution.json">Лучшее решение</a> · <a href="examples.json">Примеры ограничений</a></p></section></main></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")
