import bisect
import itertools
import math


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated_sort(population):
    n = len(population)
    dominated, counts = [[] for _ in range(n)], [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = population[i].costs, population[j].costs
            if dominates(a, b):
                dominated[i].append(j)
                counts[j] += 1
            elif dominates(b, a):
                dominated[j].append(i)
                counts[i] += 1
    fronts = [[i for i, count in enumerate(counts) if count == 0]] if n else []
    while fronts and fronts[-1]:
        next_front = []
        for i in fronts[-1]:
            for j in dominated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        else:
            break
    return fronts


def crowding_distance(population, front):
    distances = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for axis in range(3):
        ordered = sorted(front, key=lambda i: population[i].costs[axis])
        lo, hi = population[ordered[0]].costs[axis], population[ordered[-1]].costs[axis]
        if lo == hi:
            continue
        distances[ordered[0]] = distances[ordered[-1]] = math.inf
        for position in range(1, len(ordered) - 1):
            before = population[ordered[position - 1]].costs[axis]
            after = population[ordered[position + 1]].costs[axis]
            distances[ordered[position]] += (after - before) / (hi - lo)
    return distances


def ranks_and_distances(population):
    ranks, distances = {}, {}
    for rank, front in enumerate(nondominated_sort(population)):
        ranks.update({i: rank for i in front})
        distances.update(crowding_distance(population, front))
    return ranks, distances


def environmental_selection(population, size):
    result = []
    for front in nondominated_sort(population):
        if len(result) + len(front) <= size:
            result.extend(population[i] for i in front)
        else:
            distances = crowding_distance(population, front)
            selected = sorted(front, key=lambda i: distances[i], reverse=True)[:size - len(result)]
            result.extend(population[i] for i in selected)
            break
    return result


def pareto_archive(solutions):
    unique = {}
    for solution in solutions:
        unique.setdefault(solution.costs, solution)
    ordered = sorted(unique.values(), key=lambda s: (s.costs, s.genome))
    y_values = sorted({s.costs[1] for s in ordered})
    tree = [math.inf] * (len(y_values) + 1)
    front = []
    for solution in ordered:
        index = bisect.bisect_left(y_values, solution.costs[1]) + 1
        query, best = index, math.inf
        while query > 0:
            best = min(best, tree[query])
            query -= query & -query
        if best <= solution.costs[2]:
            continue
        front.append(solution)
        while index < len(tree):
            tree[index] = min(tree[index], solution.costs[2])
            index += index & -index
    return front


def hypervolume(points, reference=(1.1, 1.1, 1.1)):
    if len(reference) != 3 or not all(math.isfinite(v) for v in reference):
        raise ValueError("Опорная точка должна содержать три конечных числа.")
    points = sorted(set(tuple(point) for point in points))
    if any(len(p) != 3 or not all(math.isfinite(v) for v in p) or any(v > r for v, r in zip(p, reference)) for p in points):
        raise ValueError("Точки должны лежать не хуже опорной точки по всем осям.")
    groups = [(x, list(group)) for x, group in itertools.groupby(points, key=lambda p: p[0])]
    skyline, volume = [], 0.0
    for group_index, (x, group) in enumerate(groups):
        for _, y, z in group:
            index = bisect.bisect_left(skyline, (y, -math.inf))
            if index and skyline[index - 1][1] <= z:
                continue
            if index < len(skyline) and skyline[index][0] == y and skyline[index][1] <= z:
                continue
            while index < len(skyline) and skyline[index][1] >= z:
                skyline.pop(index)
            skyline.insert(index, (y, z))
        area, previous_z = 0.0, reference[2]
        for y, z in skyline:
            area += (reference[1] - y) * (previous_z - z)
            previous_z = z
        next_x = groups[group_index + 1][0] if group_index + 1 < len(groups) else reference[0]
        volume += (next_x - x) * area
    return volume
