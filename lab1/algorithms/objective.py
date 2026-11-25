from typing import Sequence

VARIANT = 14
DIMENSION = 5 + VARIANT % 6
LOWER_BOUND = -5.0
UPPER_BOUND = 5.0


GLOBAL_MINIMIZER_COORDINATE = -2.903534027771178
GLOBAL_MINIMUM = 0.5 * DIMENSION * (
    GLOBAL_MINIMIZER_COORDINATE ** 4
    - 16.0 * GLOBAL_MINIMIZER_COORDINATE ** 2
    + 5.0 * GLOBAL_MINIMIZER_COORDINATE
)


def stiblinski_tang(vector: Sequence[float]) -> float:
    if len(vector) != DIMENSION:
        raise ValueError(f"Ожидалось координат: {DIMENSION}, получено: {len(vector)}")
    return 0.5 * sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x for x in vector)
