"""
Monte Carlo perturbation of model outputs, for propagating model-form uncertainty.

Why this exists: every correlation and property model in this library carries a
documented uncertainty band -- roughly +/- 25 percent for Dittus-Boelter, 5 to 7 percent
for the liquid-metal heat capacities, 10 percent for UO2 conductivity below 2000 K. Those
numbers are recorded in the docstrings and range tables but, until now, nothing consumed
them. This module is the mechanism that turns them into an error bar on a full
single-channel analysis: switch perturbation on, run the same case many times, and the
spread of the answers is the propagated model-form uncertainty.

The API is deliberately minimal: a module-level on/off switch, a seed, and one function
that perturbs a value. Nothing here knows anything about which correlation it is
perturbing -- the caller supplies the relative sigma from its own documented band, so
this works identically for a property library and a correlation library.
"""
from pinthac import backend


ENABLED = False
SEED = 0

_np_rng = backend.np.random.default_rng(SEED)
_torch_generator = None


def enable(seed=0):
    """
    Switch perturbation on and reset the random streams to a known state.

    Why this model is here:
        A Monte Carlo study has to be reproducible to be worth anything -- a reviewer
        must be able to regenerate the exact same error band. Seeding through this one
        function, rather than letting callers seed the global numpy and torch generators
        themselves, keeps the library's randomness separate from whatever else in the
        process is drawing random numbers (a training loop's weight initialization, most
        obviously).

    Inputs:
        seed : integer seed for both the numpy and torch streams
    Returns:
        None
    """
    global ENABLED, SEED, _np_rng, _torch_generator
    ENABLED = True
    SEED = seed
    _np_rng = backend.np.random.default_rng(seed)
    if backend.TORCH_AVAILABLE:
        _torch_generator = backend.torch.Generator().manual_seed(seed)


def disable():
    """
    Switch perturbation off. Every perturb() call then returns its input unchanged.

    Why this model is here:
        Off is the default and must stay cheap: a deterministic run should not pay for
        random number generation it does not use, and, more importantly, a user who has
        not asked for uncertainty must never silently receive a perturbed answer.

    Returns:
        None
    """
    global ENABLED
    ENABLED = False


def perturb(value, rel_sigma):
    """
    Perturb a model output within its documented model-form uncertainty.

    Why this model is here:
        The single mechanism behind every uncertainty band this library produces. It is
        applied at the point a model returns its answer, so the perturbation propagates
        through everything downstream of that model exactly as a real modelling error
        would.

    Formulation:
        value_perturbed = value * (1 + rel_sigma * z),   z ~ N(0, 1)

        Multiplicative rather than additive, because correlation uncertainties are
        published as percentages, and because it keeps a positive quantity positive for
        any sensible sigma. One independent z is drawn per element, so a batch of cases
        gets a batch of independent perturbations rather than one shared offset -- which
        is what makes a single batched solve equivalent to many Monte Carlo trials.

        The draw is treated as a constant with respect to autograd: it is a sample of a
        modelling error, not a differentiable function of the inputs. A gradient taken
        through a perturbed result is therefore the underlying model's gradient scaled
        by the realized factor (1 + rel_sigma*z), which is the correct sensitivity for
        that particular Monte Carlo trial.

    Inputs:
        value     : model output, float / numpy array / torch tensor
        rel_sigma : relative standard deviation, dimensionless (0.25 for +/- 25 percent)

    Returns:
        the perturbed value, same type and shape as `value`; the input unchanged when
        perturbation is disabled
    """
    if not ENABLED or rel_sigma == 0.0:
        return value

    if backend.is_torch(value):
        z = backend.torch.randn(value.shape, dtype=value.dtype, device=value.device,
                                generator=_torch_generator)
        return value * (1.0 + rel_sigma * z)

    array = backend.np.asarray(value, dtype=float)
    z = _np_rng.standard_normal(array.shape)
    return value * (1.0 + rel_sigma * z)


def band(value, rel_sigma, n_samples):
    """
    Draw n_samples independent perturbations of a single value.

    Why this model is here:
        The plotting path. A figure showing an uncertainty band around a property curve
        needs many samples of the same curve, which perturb() alone cannot give -- it
        draws one perturbation per element, not many per element. This stacks the samples
        along a new leading axis so a caller can take percentiles across axis 0.

        Works regardless of the module-level switch, since asking for a band is itself an
        explicit request for perturbed values.

    Inputs:
        value      : model output, float / numpy array / torch tensor
        rel_sigma  : relative standard deviation, dimensionless
        n_samples  : number of Monte Carlo samples, integer

    Returns:
        samples : array of shape (n_samples,) + shape(value), same kind as `value`
    """
    if backend.is_torch(value):
        shape = (n_samples,) + tuple(value.shape)
        z = backend.torch.randn(shape, dtype=value.dtype, device=value.device,
                                generator=_torch_generator)
        return value.unsqueeze(0) * (1.0 + rel_sigma * z)

    array = backend.np.asarray(value, dtype=float)
    z = _np_rng.standard_normal((n_samples,) + array.shape)
    return array[None, ...] * (1.0 + rel_sigma * z)
