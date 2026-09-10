import numpy as np
import array_api_compat as aac


def compat(*args):
    """
    Returns the appropriate array namespace for the given inputs.

    Usage inside a function:
        lib = compat(x, y, ...)
        result = lib.exp(x)

    Falls back to numpy when all inputs are plain Python scalars (int/float/complex).
    Otherwise delegates to array_api_compat to detect numpy arrays, torch tensors,
    jax arrays, etc.
    """
    arrays = [a for a in args if not isinstance(a, (int, float, complex))]
    if not arrays:
        return np
    return aac.array_namespace(*arrays)
