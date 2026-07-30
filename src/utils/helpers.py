from collections.abc import Iterable


def as_list(x):
    if isinstance(x, str):
        return [x]

    if isinstance(x, Iterable):
        return list(x)

    raise TypeError("features must be str or iterable")