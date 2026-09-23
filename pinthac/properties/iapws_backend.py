"""Float / NumPy / PyTorch plumbing shared by the three IAPWS modules.

Why this module is here:
    `pinthac/backend.py` dispatches a correlation onto whichever array library its
    arguments came from, because a correlation is a handful of arithmetic lines that
    NumPy and Torch both spell the same way. The IAPWS equations of state are not like
    that. IAPWS-95 sums 56 residual terms against stacked coefficient tables, solves a
    two-by-two Newton system on the saturation line, and inverts p(rho,T) for density;
    IF97 selects backward-equation subregions elementwise. All of that is written once,
    in torch, and the float / NumPy paths are a conversion at the boundary -- so the
    dispatch these modules need is a type round-trip, not a library switch.

    Before this module existed each of iapws95.py and iapws97.py carried its own copy of
    that round-trip, and the two copies disagreed: one forced every input onto the module
    device, the other rebuilt scalars with `torch.tensor(x)` and inherited whatever
    process-wide default dtype happened to be set. Both are collected here so there is
    one answer to "what does an IAPWS function do with its arguments".

The contract every IAPWS entry point keeps, via prepare() and restore():

    1. Floats, NumPy arrays and torch tensors all work, and the result comes back as the
       same kind of thing that went in. A Python list or tuple is accepted too and is
       treated as a NumPy array -- the answer comes back as one. That is deliberate
       rather than an omission: the state dicts these modules return are meant to be
       done arithmetic on, and a dict of lists cannot be, so returning lists would hand
       the caller something that looks right and fails on the next line.
    2. A torch input keeps its own device. Nothing is moved to the module `device`
       except values that arrived with no device of their own -- floats, lists and NumPy
       arrays, which are host data and are evaluated on the host. A CPU tensor stays on
       the CPU even when a GPU is present, and a tensor on `cuda:1` is answered on
       `cuda:1`.
    3. A torch input stays differentiable. Nothing here detaches, and the restore step
       is a reshape and a cast, both of which carry the graph.
    4. The arithmetic runs in float64 regardless of what came in, and the result is cast
       back to the caller's dtype. The residual terms of IAPWS-95 carry tau exponents up
       to 50 and cancel against each other to several digits; in float32 the cancellation
       leaves noise rather than a pressure. A float32 tensor is therefore upcast on the
       way in and downcast on the way out, which is differentiable in both directions.
    5. Inputs broadcast against each other, so `helmholtz(rho_array, 500.0)` means what
       it looks like it means.

Reference:
    Not applicable -- plumbing, no physics.
"""
import numpy as np
import torch

# Where a float or NumPy input is evaluated. Torch inputs always override it with their
# own device, so this is a default and never a destination.
#
# It is the CPU, deliberately, even on a machine with a GPU. pinthac/backend.py already
# states the library's rule -- a tensor input selects torch and keeps its device, anything
# else selects NumPy -- and NumPy means the host. Following it here is also much faster
# for the calls that actually arrive as NumPy. The single-channel solvers invert density
# for five to a few hundred points at a time, hundreds of times per solve; on this
# machine a batch of five costs 1.0 s for twenty rho_Tp calls on the CPU and 22.7 s on
# the GPU, because sixty Newton iterations over five elements is nothing but kernel
# launch latency. An earlier version of these modules forced every input onto the GPU,
# and ml/pinn.py carried a monkey-patch to undo it.
#
# The crossover is around a hundred thousand points: at one million states the same work
# is 3.0 s on the CPU and 0.5 s on the GPU. A caller who wants that puts their data on
# the GPU themselves -- `torch.as_tensor(x, device=iapws_backend.accelerator)` -- and
# everything from there down follows, including the coefficient tables. Setting this
# module attribute works too, and is what a caller who only has NumPy arrays should do.
device = torch.device("cpu")

# The device to put a tensor on to get the batched GPU path, for callers who want it.
# Read it rather than assuming; nothing in the library prints it, because CONTRIBUTING.md
# section 5 rule 6 forbids announcing anything at import time.
accelerator = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Everything is evaluated in this dtype. See rule 4 above for why it is not negotiable.
dtype = torch.float64


def prepare(*vals):
    """Bring a set of mixed float / NumPy / list / torch inputs to one common torch form.

    Formulation:
        Not applicable -- type handling, no physics.

    Valid range:
        Not applicable.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable.

    Inputs:
        *vals : any mix of Python floats/ints, lists, NumPy arrays and torch tensors,
                broadcastable against each other (a list counts as a NumPy array)

    Returns:
        flat  : tuple of 1-D float64 torch tensors, one per input, all the same length,
                already broadcast against each other and living on the device described
                by `state`. Torch inputs keep their autograd graph.
        state : dict describing how to undo this, to be handed to restore(). Keys are
                'kind' ('torch' / 'numpy' / 'scalar'), 'shape' (the broadcast shape),
                'device' and 'dtype' (the caller's, not the working dtype).
    """
    ref = None
    for v in vals:
        if isinstance(v, torch.Tensor):
            ref = v
            break

    if ref is not None:
        kind = 'torch'
        dev = ref.device
        # An integer tensor in means a float tensor out: there is no sensible integer
        # answer to a density, and rounding one would be worse than changing the dtype.
        out_dtype = ref.dtype if ref.dtype.is_floating_point else dtype
    elif any(isinstance(v, (np.ndarray, list, tuple)) for v in vals):
        kind, dev, out_dtype = 'numpy', device, dtype
    else:
        kind, dev, out_dtype = 'scalar', device, dtype

    tens = []
    for v in vals:
        if isinstance(v, torch.Tensor):
            # .to() is differentiable, and a no-op when the tensor is already float64 on
            # this device -- so a tensor that arrives in the working form is not copied.
            tens.append(v.to(device=dev, dtype=dtype))
        else:
            tens.append(torch.as_tensor(v, dtype=dtype, device=dev))

    shape = torch.broadcast_shapes(*[t.shape for t in tens])
    flat = tuple(t.expand(shape).reshape(-1) for t in tens)

    return flat, {'kind': kind, 'shape': shape, 'device': dev, 'dtype': out_dtype}


def restore(val, state):
    """Put a computed 1-D float64 result back into the form the caller's inputs came in.

    Formulation:
        Not applicable -- type handling, no physics.

    Valid range:
        Not applicable.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable.

    Inputs:
        val   : 1-D torch tensor of results, as long as the flattened inputs were
        state : the dict prepare() returned

    Returns:
        the same numbers as a torch tensor (on the caller's device, in the caller's
        dtype, with the autograd graph intact), a NumPy array, or a Python float --
        matching what the caller passed in
    """
    val = val.reshape(state['shape'])

    if state['kind'] == 'torch':
        return val.to(state['dtype'])
    if state['kind'] == 'numpy':
        return val.detach().cpu().numpy()
    # A scalar in gives a scalar out. reshape above made it shape (), so .item() is the
    # only cast here and it cannot be reached from a torch input -- see rule 2.
    return val.item()


_moved = {}


def on(table, ref):
    """Return a published coefficient table on the same device as `ref`.

    Formulation:
        Not applicable -- table management, no physics.

    Valid range:
        `table` must be a module- or class-level constant that lives for the life of the
        process. Do not call this on a tensor built inside a function: the cache holds
        the result forever and would leak.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable.

    Inputs:
        table : CPU torch tensor of published coefficients
        ref   : any torch tensor whose device the table should follow

    Returns:
        the same table as a tensor on ref.device (the original object when ref is
        already on the CPU, so the common case costs nothing)
    """
    key = (id(table), ref.device)
    hit = _moved.get(key)
    if hit is None:
        hit = (table, table.to(ref.device))
        _moved[key] = hit
    return hit[1]


def stack(*rows):
    """Stack the coefficient columns of one term group into a single constant table.

        The trailing axis of length 1 is what makes the sum work: each column becomes
        shape (terms, 1), which broadcasts against a state vector of shape (points,) to
        give (terms, points), and the term sum is then .sum(dim=0).

    Formulation:
        Not applicable -- table management, no physics.

    Valid range:
        Not applicable.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable.

    Inputs:
        *rows : equal-length sequences of published coefficients, one per column

    Returns:
        table : CPU float64 tensor of shape (columns, terms, 1); unpacking it along its
                first axis gives one (terms, 1) column per input row
    """
    return torch.tensor([list(r) for r in rows], dtype=dtype).unsqueeze(-1)


# Unit scaling for the property accessors. Both tables are keyed off the fact that the
# specific gas constant is carried in kJ/kg/K throughout the IAPWS modules, so an energy
# comes out of the equations in kJ/kg and a pressure in kPa; these factors take it from
# there. They live here so that iapws95.py and iapws97.py cannot drift apart on what
# `units='J'` means.
energy_units = {'J': 1000.0, 'kJ': 1.0, 'MJ': 1.0e-3}       # base: kJ/kg
pressure_units = {'Pa': 1000.0, 'kPa': 1.0, 'MPa': 1.0e-3}  # base: kPa


def sqrt(val):
    """Square root of whatever a property accessor returned.

    Formulation:
        Not applicable -- type handling, no physics.

    Valid range:
        val >= 0; a negative value gives NaN, as it does in both libraries.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable.

    Inputs:
        val : float, numpy array, or torch tensor

    Returns:
        the square root, as the same kind of object
    """
    if isinstance(val, torch.Tensor):
        return torch.sqrt(val)
    if isinstance(val, np.ndarray):
        return np.sqrt(val)
    return float(np.sqrt(val))
