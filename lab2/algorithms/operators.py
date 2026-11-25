def uniform_crossover(parent_a, parent_b, rng):
    return tuple(a if rng.random() < 0.5 else b for a, b in zip(parent_a, parent_b))


def flip_mutation(genome, rng):
    result = list(genome)
    i = rng.randrange(len(result))
    result[i] = 1 - result[i]
    return tuple(result)


def swap_mutation(genome, rng):
    selected = [i for i, bit in enumerate(genome) if bit]
    empty = [i for i, bit in enumerate(genome) if not bit]
    if not selected or not empty:
        return genome
    result = list(genome)
    result[rng.choice(selected)] = 0
    result[rng.choice(empty)] = 1
    return tuple(result)


def repair(genome, k, rng):
    if not 0 <= k <= len(genome):
        raise ValueError("Недопустимое число пунктов.")
    result = list(genome)
    selected = [i for i, bit in enumerate(result) if bit]
    empty = [i for i, bit in enumerate(result) if not bit]
    if len(selected) > k:
        for i in rng.sample(selected, len(selected) - k):
            result[i] = 0
    elif len(selected) < k:
        for i in rng.sample(empty, k - len(selected)):
            result[i] = 1
    return tuple(result)
