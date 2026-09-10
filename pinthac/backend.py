"""
Array-library dispatch, so every model in PINTHAC is written once.

NumPy and PyTorch share almost identical function names (exp, log, sqrt, where, ...),
so nearly every correlation in this library can run on plain Python floats, numpy
arrays, or torch tensors from a single implementation, just by looking up the right
module here. That keeps one version of each physical model instead of a numpy version
and a torch version that drift apart -- which is exactly what happened to the three
separate dispatchers this module replaces (Arr_Compat.compat, Liquid_Metals.lib, and
the ad-hoc numpy calls inside HTC.py).
"""
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def lib(*args):
    """
    Return the array library that should be used for a set of inputs.

    Rule: if ANY input is a torch tensor, use torch, so autograd is preserved and the
    caller's device placement is respected. Otherwise use numpy, which also handles
    plain Python floats correctly.

    Getting the argument list right matters more than it looks. The most common bug in
    the pre-cleanup code was calling the dispatcher with the wrong things -- e.g.
    `compat(G, D)` where G and D were plain floats and the tensor arrived inside a
    property dictionary. That resolves to numpy, and numpy silently round-trips a torch
    tensor through __array_ufunc__, severing the autograd graph without raising. It only
    becomes a visible error once the tensor carries requires_grad=True. Pass every input
    that could be an array.

    Inputs:
        *args : any mix of floats, numpy arrays, or torch tensors
    Returns:
        module : either `torch` or `numpy`
    """
    if TORCH_AVAILABLE:
        for a in args:
            if isinstance(a, torch.Tensor):
                return torch
    return np


def is_torch(*args):
    """
    True if any input is a torch tensor.

    Used where a function has to make a structural choice rather than a numeric one --
    picking a code path, not a value. Never use it to branch on a *value*; use where().

    Inputs:
        *args : any mix of floats, numpy arrays, or torch tensors
    Returns:
        flag : bool
    """
    return lib(*args) is not np


def promote(x, like):
    """
    Turn a plain Python scalar into an array of the same kind as `like`.

    Why this exists: torch's ufuncs reject Python floats outright
    (`torch.exp(0.0)` -> TypeError), while numpy's accept them. So a correlation with a
    scalar default argument -- `k_NFI(T, Bu=0.0)` is the real case -- works on numpy and
    fails on torch, even though the physics is identical. One promote() call at the top
    of such a function fixes it without touching the equations below.

    Already-array inputs pass through untouched, so this is safe to apply unconditionally
    and costs nothing when the caller already passed an array.

    Inputs:
        x    : float, numpy array, or torch tensor
        like : an input whose library, dtype and device x should match
    Returns:
        x as an array of the same kind as `like`
    """
    xp = lib(like)
    if xp is np:
        return np.asarray(x, dtype=float) if np.isscalar(x) else x
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def where(cond, a, b):
    """
    Elementwise select, with the scalar promotion torch.where refuses to do itself.

    Why this exists: `np.where(T < 1000, low, high)` accepts a Python bool condition and
    float branches; `torch.where` requires all three to be tensors and raises otherwise.
    Every piecewise property model in this library hits that difference.

    This is also the replacement for `if` statements on array values. A Python `if` on a
    tensor forces a synchronization, collapses a batch to one branch, and breaks under
    vmap -- so piecewise physics is written with where(), always.

    Inputs:
        cond  : boolean array or tensor (or a Python bool, if a and b are scalars)
        a, b  : values taken where cond is True / False
    Returns:
        selected values, same kind as the array inputs
    """
    xp = lib(cond, a, b)
    if xp is np:
        return np.where(cond, a, b)

    # The branches must be promoted against a *value* tensor, never against cond:
    # cond is boolean, and promoting a float branch to bool dtype turns 1.0 and 2.0
    # into True and True.
    ref = a if isinstance(a, torch.Tensor) else (b if isinstance(b, torch.Tensor) else None)
    if ref is None:
        device = cond.device if isinstance(cond, torch.Tensor) else None
        ref = torch.zeros((), dtype=torch.get_default_dtype(), device=device)
    if not isinstance(cond, torch.Tensor):
        cond = torch.as_tensor(cond, device=ref.device)
    return torch.where(cond, promote(a, ref), promote(b, ref))


def clip(x, lo=None, hi=None):
    """
    Clamp x into [lo, hi]. Either bound may be None.

    Why this exists: the operation is `np.clip` in numpy and `torch.clamp` in torch --
    different names for the same thing, so it cannot be reached through the module that
    lib() returns.

    Clamping rather than folding matters physically in a couple of places: taking abs()
    of a temperature that a solver has pushed negative creates a spurious mirror root
    that the solver can then lock onto, whereas clamping to a floor keeps the residual
    monotonic and pushes the iterate back toward the feasible region.

    Inputs:
        x      : float, numpy array, or torch tensor
        lo, hi : lower / upper bound, or None
    Returns:
        clipped value, same kind as x
    """
    xp = lib(x)
    if xp is np:
        return np.clip(x, lo, hi)
    return torch.clamp(x, min=lo, max=hi)


def maximum(a, b):
    """
    Elementwise maximum of two values.

    Why this exists: torch.maximum needs both arguments to be tensors, so the very
    common `maximum(x, 1.0)` guard against a non-physical excursion fails on torch
    unless the scalar is promoted first.

    Inputs:
        a, b : floats, numpy arrays, or torch tensors
    Returns:
        elementwise maximum, same kind as the array inputs
    """
    xp = lib(a, b)
    if xp is np:
        return np.maximum(a, b)
    ref = a if isinstance(a, torch.Tensor) else b
    return torch.maximum(promote(a, ref), promote(b, ref))


def interp(x, xp_tab, fp_tab):
    """
    Piecewise-linear interpolation of a lookup table, differentiable under torch.

    Why this exists: torch has no `interp` at all. Several material property models are
    published as tables rather than formulas -- Zircaloy specific heat and both Zircaloy
    thermal-expansion tables are the cases here -- so without this they simply cannot run
    on a tensor.

    Implemented with searchsorted and an explicit two-point linear blend rather than by
    calling out to numpy, so the result stays on the caller's device and keeps a gradient
    with respect to x. Values outside the table are held flat at the end points, matching
    numpy's default; the table itself is treated as constant data, so no gradient flows
    back into xp_tab/fp_tab.

    Inputs:
        x      : query point(s), float / numpy array / torch tensor
        xp_tab : table abscissae, strictly increasing
        fp_tab : table ordinates, same length as xp_tab
    Returns:
        interpolated value(s), same kind as x
    """
    xl = lib(x)
    if xl is np:
        return np.interp(x, xp_tab, fp_tab)

    xt = promote(x, x)
    xs = torch.as_tensor(xp_tab, dtype=xt.dtype, device=xt.device)
    fs = torch.as_tensor(fp_tab, dtype=xt.dtype, device=xt.device)

    idx = torch.searchsorted(xs, xt.detach().reshape(-1).contiguous())
    idx = torch.clamp(idx, 1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    f0, f1 = fs[idx - 1], fs[idx]

    slope = (f1 - f0) / (x1 - x0)
    out = f0 + slope * (xt.reshape(-1) - x0)
    out = torch.where(xt.reshape(-1) < xs[0], fs[0], out)
    out = torch.where(xt.reshape(-1) > xs[-1], fs[-1], out)
    return out.reshape(xt.shape)


def zeros_like(x):
    """
    An array of zeros shaped like x, accepting a plain float for x.

    Why this exists: the piecewise property models need a zero branch to hand to where(),
    and `np.zeros_like(2.0)` returns an integer-dtype zero-dimensional array, which then
    silently truncates any float result assigned through it.

    Inputs:
        x : float, numpy array, or torch tensor
    Returns:
        zeros of the same kind, shape and float dtype as x
    """
    xp = lib(x)
    if xp is np:
        return np.zeros_like(np.asarray(x, dtype=float))
    return torch.zeros_like(x)


def promote_all(*args):
    """
    Promote every scalar argument to match whichever argument is an array.

    Why this exists: promote() matches one value against one named reference, which only
    works when you already know which of the two is the array. In a correlation you often
    do not. `k_NFI(T, Bu)` is called with a whole axial temperature field and a scalar
    burnup, and equally with a scalar temperature and a burnup sweep -- so
    `Bu = promote(Bu, T)` fixes the first call shape and breaks on the second.

    This resolves the array library once across all the arguments, exactly as lib() does,
    and lifts every scalar to it. Use it at the top of any function whose arguments can
    independently be scalars or arrays:

        T, sig, t = backend.promote_all(T, sig, t)
        xp = backend.lib(T, sig, t)

    Inputs:
        *args : any mix of floats, numpy arrays, or torch tensors
    Returns:
        tuple of the same values, with scalars lifted to the dispatched library. Returned
        unchanged when every argument is already a scalar, since numpy handles those.
    """
    xp = lib(*args)
    if xp is np:
        return args

    reference = next(a for a in args if isinstance(a, torch.Tensor))
    return tuple(promote(a, reference) for a in args)
