"""
Smoke tests for the shared foundation: backend dispatch, range checking, and the Monte
Carlo perturbation.

These three modules are what everything else stands on, so they are tested first and
tested for behaviour rather than for particular numbers: does dispatch pick the right
library, does a piecewise model survive a torch tensor, does a gradient still flow, does
a range violation warn exactly once.
"""
import warnings

import numpy as np
import pytest
import torch

from pinthac import backend, ranges, uncertainty


FLOAT = 900.0
ARRAY = np.array([700.0, 900.0, 1500.0])


def grad_tensor():
    """A float64 tensor carrying gradient, the case that catches silent numpy fallback."""
    return torch.tensor([700.0, 900.0, 1500.0], dtype=torch.float64, requires_grad=True)


# --------------------------------------------------------------------------- dispatch
def test_lib_picks_numpy_for_scalars_and_arrays():
    assert backend.lib(FLOAT) is np
    assert backend.lib(ARRAY) is np
    assert backend.lib(FLOAT, ARRAY) is np


def test_lib_picks_torch_if_any_input_is_a_tensor():
    # The mixed case is the one that matters: a tensor hiding among plain floats is
    # exactly how the pre-cleanup code silently fell back to numpy.
    assert backend.lib(grad_tensor()) is torch
    assert backend.lib(FLOAT, ARRAY, grad_tensor()) is torch
    assert backend.is_torch(FLOAT, grad_tensor()) is True
    assert backend.is_torch(FLOAT, ARRAY) is False


# ------------------------------------------------------------------------- promotion
def test_promote_leaves_arrays_alone_and_lifts_scalars():
    t = grad_tensor()
    assert backend.promote(t, t) is t
    lifted = backend.promote(0.0, t)
    assert isinstance(lifted, torch.Tensor)
    assert lifted.dtype == t.dtype and lifted.device == t.device


def test_promote_unblocks_a_scalar_default_argument():
    # torch.exp(0.0) raises; this is the k_NFI(T, Bu=0.0) failure in miniature.
    t = grad_tensor()
    assert torch.isfinite(torch.exp(-backend.promote(0.0, t))).all()


# ------------------------------------------------------------------------------ where
def test_where_keeps_float_branches_float():
    # Promoting the branches against a boolean condition would turn 1.0 and 2.0 into
    # True and True, which is how this went wrong the first time.
    out = backend.where(grad_tensor() < 1000.0, 1.0, 2.0)
    assert out.dtype is not torch.bool
    assert out.tolist() == [1.0, 1.0, 2.0]


def test_where_matches_numpy_and_carries_a_gradient():
    cond_np = ARRAY < 1000.0
    assert backend.where(cond_np, 1.0, 2.0).tolist() == [1.0, 1.0, 2.0]

    t = grad_tensor()
    out = backend.where(t < 1000.0, t * 2.0, 5.0)
    (grad,) = torch.autograd.grad(out.sum(), t)
    assert grad.tolist() == [2.0, 2.0, 0.0]


def test_where_nests_for_three_way_piecewise_models():
    t = grad_tensor()
    out = backend.where(t < 800.0, 0.0, backend.where(t < 1200.0, t / 100.0, 99.0))
    assert out.tolist() == [0.0, 9.0, 99.0]


# ---------------------------------------------------------------- clip, maximum, zeros
def test_clip_and_maximum_accept_scalar_bounds_on_torch():
    t = grad_tensor()
    assert backend.clip(t, 800.0, 1200.0).tolist() == [800.0, 900.0, 1200.0]
    assert backend.maximum(t, 1000.0).tolist() == [1000.0, 1000.0, 1500.0]
    assert backend.clip(ARRAY, 800.0, 1200.0).tolist() == [800.0, 900.0, 1200.0]


def test_zeros_like_of_a_python_float_is_floating_point():
    # np.zeros_like(2.0) is integer dtype, which silently truncates anything assigned
    # through it.
    assert backend.zeros_like(2.0).dtype == np.float64
    assert backend.zeros_like(grad_tensor()).tolist() == [0.0, 0.0, 0.0]


# ----------------------------------------------------------------------------- interp
TAB_X = [290.0, 300.0, 400.0, 640.0, 1090.0]
TAB_Y = [279.0, 281.0, 302.0, 331.0, 375.0]


def test_interp_agrees_with_numpy_including_flat_extrapolation():
    query = np.array([250.0, 295.0, 350.0, 500.0, 2000.0])
    expected = np.interp(query, TAB_X, TAB_Y)
    got = backend.interp(torch.tensor(query, dtype=torch.float64), TAB_X, TAB_Y)
    assert np.allclose(got.numpy(), expected)
    assert np.allclose(backend.interp(query, TAB_X, TAB_Y), expected)


def test_interp_is_differentiable_and_flat_outside_the_table():
    q = torch.tensor([250.0, 350.0, 2000.0], dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(backend.interp(q, TAB_X, TAB_Y).sum(), q)
    assert grad[0] == 0.0 and grad[2] == 0.0        # held flat past the ends
    assert grad[1] == pytest.approx((302.0 - 281.0) / (400.0 - 300.0))


# ----------------------------------------------------------------------------- ranges
TABLE = {"Re": (1.0e4, None), "Pr": (0.7, 160.0), "G": (None, 1000.0)}


def test_in_range_inputs_pass_silently():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ranges.check("dittus_boelter", {"Re": 5.0e4, "Pr": 2.0}, TABLE) is True
    assert caught == []


def test_a_batch_violation_warns_once_and_names_the_worst_excursion():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = ranges.check("wu", {"G": np.linspace(600.0, 2500.0, 5000)}, TABLE)
    assert ok is False
    assert len(caught) == 1                          # once per call, not per element
    assert "2500" in str(caught[0].message) and "wu" in str(caught[0].message)


def test_range_check_reads_torch_tensors_without_breaking_the_graph():
    t = torch.tensor([3.0e3, 5.0e4], dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ranges.check("dittus_boelter", {"Re": t}, TABLE) is False
    assert len(caught) == 1
    assert t.grad is None                            # inspecting must not touch autograd


def test_checking_can_be_switched_off_wholesale():
    ranges.CHECKING_ENABLED = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert ranges.check("wu", {"G": 9999.0}, TABLE) is True
        assert caught == []
    finally:
        ranges.CHECKING_ENABLED = True


# ------------------------------------------------------------------------ uncertainty
def test_perturbation_is_off_by_default_and_returns_the_input_untouched():
    uncertainty.disable()
    assert uncertainty.perturb(ARRAY, 0.25) is ARRAY


def test_perturbation_is_reproducible_from_the_seed():
    uncertainty.enable(seed=11)
    first = uncertainty.perturb(ARRAY, 0.25)
    uncertainty.enable(seed=11)
    assert np.allclose(uncertainty.perturb(ARRAY, 0.25), first)
    uncertainty.disable()


def test_each_element_gets_an_independent_draw():
    # One shared offset across a batch would make a batched solve useless as a set of
    # independent Monte Carlo trials.
    uncertainty.enable(seed=3)
    out = uncertainty.perturb(np.full(2000, 100.0), 0.2)
    ratios = out / 100.0
    assert ratios.std() == pytest.approx(0.2, rel=0.1)
    uncertainty.disable()


def test_band_recovers_the_requested_sigma():
    uncertainty.enable(seed=5)
    samples = uncertainty.band(ARRAY, 0.10, 20000)
    assert samples.shape == (20000, 3)
    assert (samples.std(axis=0) / ARRAY) == pytest.approx(0.10, rel=0.05)
    uncertainty.disable()


def test_perturbation_keeps_torch_inputs_differentiable():
    uncertainty.enable(seed=2)
    t = grad_tensor()
    out = uncertainty.perturb(t, 0.25)
    (grad,) = torch.autograd.grad(out.sum(), t)
    assert torch.isfinite(grad).all()
    # The draw is a constant, so the gradient is the realized perturbation factor.
    assert torch.allclose(grad, out / t)
    uncertainty.disable()
