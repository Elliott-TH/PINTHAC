"""Tests for torchsolve.

Run with:  python -m pytest tests -q
"""

import math
import warnings

import numpy as np
import pytest
import torch

import torchsolve as ts
from torchsolve import Status

F64 = torch.float64


def t(x, n=1):
    return torch.full((n,), float(x), dtype=F64)


# --------------------------------------------------------------------------
# basic correctness
# --------------------------------------------------------------------------
def test_bisect_cos():
    f = lambda x: torch.cos(x) - x
    r = ts.bisect(f, t(0.0), t(1.0), ftol=1e-13, rtol=1e-14)
    assert r.ok
    assert abs(float(r.root[0]) - 0.7390851332151607) < 1e-12
    # the default absolute xtol floor (2e-12) governs the interval here
    assert float(r.interval_width.max()) < 5e-12
    # ask for the machine floor explicitly and the residual test still holds
    tight = ts.bisect(f, t(0.0), t(1.0), ftol=1e-15, xtol=0.0, rtol=0.0, max_iter=200)
    assert tight.ok
    assert float(tight.interval_width.max()) < 1e-15


def test_root_at_exactly_zero_is_reachable():
    """A relative-only width tolerance can never be met at x = 0."""
    f = torch.atan
    r = ts.solve(f, bracket=(t(-1.0), t(8.0)), ftol=1e-12, max_iter=200)
    assert r.ok, r.summary()
    assert abs(float(r.root[0])) < 1e-11
    # and with xtol forced to zero it must not silently succeed either
    r0 = ts.solve(f, bracket=(t(-1.0), t(8.0)), ftol=1e-12, xtol=0.0, rtol=1e-14,
                  max_iter=60)
    assert int(r0.status[0]) == Status.MAX_ITER


@pytest.mark.parametrize("method", ["auto", "newton", "secant", "bisect"])
def test_solve_methods_agree(method):
    f = lambda x: torch.cos(x) - x
    r = ts.solve(f, bracket=(t(0.0), t(1.0)), method=method, ftol=1e-13, rtol=1e-14)
    assert r.ok, r.summary()
    assert abs(float(r.root[0]) - 0.7390851332151607) < 1e-12


def test_newton_and_secant_standalone():
    f = lambda x: x**2 - 2.0
    rn = ts.newton(f, t(1.0), ftol=1e-14, rtol=1e-14)
    rs = ts.secant(f, t(1.0), t(2.0), ftol=1e-14, rtol=1e-14)
    assert rn.ok and rs.ok
    assert abs(float(rn.root[0]) - math.sqrt(2)) < 1e-13
    assert abs(float(rs.root[0]) - math.sqrt(2)) < 1e-13


def test_newton_beats_bisect_on_evaluations():
    f = lambda x: torch.exp(x) - 5.0
    fast = ts.solve(f, bracket=(t(0.0), t(10.0)), method="newton", ftol=1e-12, rtol=1e-13)
    slow = ts.solve(f, bracket=(t(0.0), t(10.0)), method="bisect", ftol=1e-12, rtol=1e-13)
    assert fast.ok and slow.ok
    assert fast.n_fev < slow.n_fev
    assert abs(float(fast.root[0]) - math.log(5.0)) < 1e-12


def test_batched_independent_parameters():
    """Each element carries its own coefficient -- the usual closure pattern."""
    a = torch.linspace(1.0, 10.0, 257, dtype=F64)
    f = lambda x: x**2 - a
    r = ts.solve(f, bracket=(torch.zeros_like(a), torch.full_like(a, 20.0)),
                 ftol=1e-11, rtol=1e-13)
    assert r.ok, r.summary()
    assert torch.allclose(r.root, a.sqrt(), atol=1e-11)


def test_multidimensional_shape_preserved():
    a = torch.rand(4, 5, 6, dtype=F64) + 1.0
    f = lambda x: x**3 - a
    r = ts.solve(f, bracket=(torch.zeros_like(a), torch.full_like(a, 3.0)), ftol=1e-11)
    assert r.root.shape == a.shape
    assert r.ok
    assert torch.allclose(r.root, a.pow(1 / 3), atol=1e-9)


def test_analytic_derivative_path():
    f = lambda x: x**3 - 8.0
    fp = lambda x: 3 * x**2
    r = ts.solve(f, bracket=(t(0.0), t(5.0)), fprime=fp, ftol=1e-12, rtol=1e-14)
    assert r.ok and abs(float(r.root[0]) - 2.0) < 1e-12


def test_fd_derivative_path():
    f = lambda x: x**3 - 8.0
    r = ts.solve(f, bracket=(t(0.0), t(5.0)), deriv="fd", ftol=1e-12, rtol=1e-14)
    assert r.ok and abs(float(r.root[0]) - 2.0) < 1e-12


def test_non_differentiable_function_degrades_to_secant():
    """A table-lookup style residual: torch sees no graph, solver must cope."""

    def f(x):
        with torch.no_grad():
            return torch.cos(x.detach()) - x.detach()

    r = ts.solve(f, bracket=(t(0.0), t(1.0)), deriv="auto", ftol=1e-12, rtol=1e-13)
    assert r.ok, r.summary()
    assert abs(float(r.root[0]) - 0.7390851332151607) < 1e-11


# --------------------------------------------------------------------------
# the safety properties
# --------------------------------------------------------------------------
def test_solve_never_leaves_the_bracket():
    """Newton from a near-flat point would jump to a different root of sin."""
    f = torch.sin
    x0 = t(1.5707)  # cos(x0) ~ 1e-4, so the raw Newton step is ~1e4 long
    naive = ts.newton(f, x0, ftol=1e-12, rtol=1e-13, max_iter=200)
    guarded = ts.solve(f, x0, bracket=(t(1.0), t(4.0)), ftol=1e-12, rtol=1e-13)

    assert guarded.ok
    assert abs(float(guarded.root[0]) - math.pi) < 1e-11
    # the unguarded method found *a* root, just not the one in [1, 4]
    if naive.ok:
        assert abs(float(naive.root[0]) - math.pi) > 1.0
    assert float(guarded.lo[0]) >= 1.0 and float(guarded.hi[0]) <= 4.0


def test_no_bracket_is_an_error_not_a_guess():
    f = lambda x: x**2 + 1.0
    r = ts.bisect(f, t(-1.0), t(1.0))
    assert int(r.status[0]) == Status.NO_BRACKET
    assert torch.isnan(r.root).all()
    with pytest.raises(ts.SolverFailure):
        r.raise_if_failed("unit test")


def test_partial_failure_isolated_to_its_element():
    lo = torch.tensor([0.0, -1.0, 0.0], dtype=F64)
    hi = torch.tensor([2.0, 1.0, 2.0], dtype=F64)
    off = torch.tensor([1.0, 4.0, 2.0], dtype=F64)  # element 1 has no root in [-1, 1]
    f = lambda x: x**2 - off
    r = ts.solve(f, bracket=(lo, hi), ftol=1e-11)
    assert not r.ok
    assert int(r.status[1]) == Status.NO_BRACKET
    assert torch.isnan(r.root[1])
    assert bool(r.converged[0]) and bool(r.converged[2])
    assert abs(float(r.root[0]) - 1.0) < 1e-10


def test_nan_from_f_is_reported():
    def f(x):
        y = torch.log(x - 0.5)  # NaN below 0.5
        return y

    r = ts.bisect(f, t(0.0), t(3.0))
    assert int(r.status[0]) in (Status.NOT_FINITE, Status.NO_BRACKET)
    assert torch.isnan(r.root).all()


def test_multiple_roots_detected_and_refused():
    f = lambda x: (x - 1.0) * (x - 2.0) * (x - 3.0)
    r = ts.solve(f, bracket=(t(0.0), t(4.0)), unique_scan=33, ftol=1e-12)
    assert int(r.status[0]) == Status.MULTIPLE_ROOTS
    assert torch.isnan(r.root).all()
    # without the check it silently returns one of the three
    loose = ts.solve(f, bracket=(t(0.0), t(4.0)), ftol=1e-12, rtol=1e-13)
    assert loose.ok


def test_strict_raises():
    f = lambda x: (x - 1.0) * (x - 2.0)
    with pytest.raises(ts.SolverFailure):
        ts.solve(f, bracket=(t(0.0), t(3.0)), strict=True)


def test_residual_test_is_not_optional():
    """A near-discontinuity: the interval collapses but the residual does not."""
    f = lambda x: torch.where(x < 0.5, x - 10.0, x + 10.0)
    r = ts.bisect(f, t(0.0), t(1.0), ftol=1e-8, max_iter=200)
    assert int(r.status[0]) in (Status.STAGNATED, Status.MAX_ITER)
    assert torch.isnan(r.root).all()


def test_newton_bounds_fail_instead_of_wandering():
    f = torch.sin
    r = ts.newton(f, t(1.5707), bounds=(t(1.0), t(4.0)), on_exit="fail")
    assert int(r.status[0]) == Status.DOMAIN_LIMIT
    assert torch.isnan(r.root).all()


def test_float32_warns():
    f = lambda x: torch.cos(x) - x
    with pytest.warns(UserWarning):
        ts.bisect(f, torch.zeros(1), torch.ones(1), ftol=1e-6, rtol=1e-6)


def test_float16_rejected():
    f = lambda x: x - 1
    with pytest.raises(ValueError):
        ts.bisect(f, torch.zeros(1, dtype=torch.float16), torch.ones(1, dtype=torch.float16))


def test_tiny_residuals_do_not_underflow_the_sign_test():
    """f = 1e-200 * (x - 1): fa*fb underflows to 0.0, sign comparison does not."""
    f = lambda x: 1e-200 * (x - 1.0)
    r = ts.bisect(f, t(0.0), t(3.0), ftol=1e-210, rtol=1e-13)
    assert r.ok, r.summary()
    assert abs(float(r.root[0]) - 1.0) < 1e-12


# --------------------------------------------------------------------------
# bracketing helpers
# --------------------------------------------------------------------------
def test_expand_bracket_right_and_left():
    f = lambda x: x - 7.0
    br = ts.expand_bracket(f, t(0.0), direction="right", dx=t(0.5))
    assert br.ok
    assert float(br.lo[0]) <= 7.0 <= float(br.hi[0])

    bl = ts.expand_bracket(f, t(20.0), direction="left", dx=t(0.5))
    assert bl.ok
    assert float(bl.lo[0]) <= 7.0 <= float(bl.hi[0])


def test_expand_bracket_both_takes_nearest():
    f = lambda x: (x - 1.0) * (x - 9.0)
    br = ts.expand_bracket(f, t(3.0), direction="both", dx=t(0.25))
    assert br.ok
    assert float(br.lo[0]) <= 1.0 <= float(br.hi[0])  # 1 is nearer to 3 than 9


def test_expand_bracket_respects_domain_limit():
    f = lambda x: x - 100.0
    br = ts.expand_bracket(f, t(0.0), direction="right", dx=t(0.1), upper=t(10.0))
    assert int(br.status[0]) == Status.DOMAIN_LIMIT
    assert torch.isnan(br.lo).all()


def test_solve_without_bracket_uses_expansion():
    f = lambda x: torch.exp(x) - 3.0
    r = ts.solve(f, t(0.0), expand={"direction": "both", "dx": t(0.1)}, ftol=1e-11, rtol=1e-13)
    assert r.ok, r.summary()
    assert abs(float(r.root[0]) - math.log(3.0)) < 1e-10


def test_scan_side_selection():
    f = lambda x: (x - 1.0) * (x - 5.0)
    left = ts.bracket_scan(f, t(0.0), t(6.0), n=61, side="lo")
    right = ts.bracket_scan(f, t(0.0), t(6.0), n=61, side="hi")
    assert float(left.lo[0]) <= 1.0 <= float(left.hi[0])
    assert float(right.lo[0]) <= 5.0 <= float(right.hi[0])
    assert int(left.n_roots[0]) == 2


# --------------------------------------------------------------------------
# the motivating case: a peak, two branches, one intended answer
# --------------------------------------------------------------------------
def _peaked(x, target):
    """Stand-in for an htc correlation peaking at the pseudocritical point."""
    return 2.0 * torch.exp(-(((x - 650.0) / 20.0) ** 2)) - target


def test_branch_selection_around_a_peak():
    target = torch.full((16,), 1.0, dtype=F64)
    T_pc = torch.full((16,), 650.0, dtype=F64)
    f = lambda x: _peaked(x, target)

    lo_branch = ts.bracket_from_anchor(f, T_pc, torch.full_like(T_pc, 500.0))
    hi_branch = ts.bracket_from_anchor(f, T_pc, torch.full_like(T_pc, 800.0))
    assert lo_branch.ok and hi_branch.ok

    r_lo = ts.solve(f, bracket=lo_branch.as_tuple(), ftol=1e-12, rtol=1e-13, unique_scan=17)
    r_hi = ts.solve(f, bracket=hi_branch.as_tuple(), ftol=1e-12, rtol=1e-13, unique_scan=17)
    assert r_lo.ok and r_hi.ok

    exact = 20.0 * math.sqrt(math.log(2.0))
    assert torch.allclose(r_lo.root, T_pc - exact, atol=1e-9)
    assert torch.allclose(r_hi.root, T_pc + exact, atol=1e-9)
    assert bool((r_lo.root < T_pc).all()) and bool((r_hi.root > T_pc).all())


def test_find_extremum_locates_the_peak():
    target = torch.zeros(4, dtype=F64)
    f = lambda x: _peaked(x, target)
    x, fx = ts.find_extremum(f, torch.full((4,), 500.0, dtype=F64),
                             torch.full((4,), 800.0, dtype=F64), mode="max")
    assert torch.allclose(x, torch.full((4,), 650.0, dtype=F64), atol=1e-6)
    assert torch.allclose(fx, torch.full((4,), 2.0, dtype=F64), atol=1e-9)


def test_full_scw_style_workflow():
    """Peak of unknown location -> split into branches -> solve the one we want."""
    target = torch.linspace(0.5, 1.5, 32, dtype=F64)
    centre = torch.linspace(640.0, 660.0, 32, dtype=F64)
    f = lambda x: 2.0 * torch.exp(-(((x - centre) / 20.0) ** 2)) - target

    peak, _ = ts.find_extremum(f, torch.full_like(target, 500.0),
                               torch.full_like(target, 800.0), mode="max")
    assert torch.allclose(peak, centre, atol=1e-5)

    br = ts.bracket_from_anchor(f, peak, torch.full_like(target, 800.0), n=65)
    br.raise_if_failed("high-T branch")
    r = ts.solve(f, bracket=br.as_tuple(), ftol=1e-12, rtol=1e-13, unique_scan=17)
    r.raise_if_failed("high-T branch")

    exact = centre + 20.0 * torch.sqrt(torch.log(2.0 / target))
    assert torch.allclose(r.root, exact, atol=1e-8)
    assert bool((r.root > peak).all())


# --------------------------------------------------------------------------
# numpy path
# --------------------------------------------------------------------------
def test_numpy_fsolve_dispatch():
    a = np.linspace(1.0, 4.0, 8)
    f = lambda x: x**2 - a
    r = ts.solve(f, np.ones_like(a) * 1.5, ftol=1e-10)
    assert r.ok
    assert np.allclose(r.root, np.sqrt(a), atol=1e-10)
    assert isinstance(r.root, np.ndarray)


def test_numpy_brentq_dispatch():
    a = np.linspace(1.0, 4.0, 8)
    f = lambda x: x**2 - a
    r = ts.solve(f, bracket=(np.zeros_like(a), np.full_like(a, 5.0)), ftol=1e-10)
    assert r.ok, r.summary()
    assert np.allclose(r.root, np.sqrt(a), atol=1e-12)


def test_numpy_torch_backend_matches_torch_path():
    a = np.linspace(1.0, 4.0, 8)
    f = lambda x: x**2 - a
    rn = ts.solve_numpy(f, bracket=(np.zeros_like(a), np.full_like(a, 5.0)),
                        backend="torch", ftol=1e-12, rtol=1e-14)
    at = torch.from_numpy(a)
    rt = ts.solve(lambda x: x**2 - at, bracket=(torch.zeros_like(at), torch.full_like(at, 5.0)),
                  deriv="fd", ftol=1e-12, rtol=1e-14)
    assert rn.ok and rt.ok
    assert np.allclose(rn.root, rt.root.numpy(), rtol=0, atol=0)


def test_numpy_fsolve_can_miss_the_intended_root_but_says_so():
    """fsolve is unbracketed; we still verify the residual per element."""
    f = lambda x: np.sin(x)
    r = ts.solve_numpy(f, np.array([1.5707]), backend="fsolve", ftol=1e-10)
    # whatever it found, the residual claim is checked
    ok = r.status == int(Status.CONVERGED)
    assert np.all(np.abs(r.f_root[ok]) <= 1e-10)


# --------------------------------------------------------------------------
# device / dtype plumbing
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_end_to_end():
    dev = torch.device("cuda")
    a = torch.linspace(1.0, 10.0, 4096, dtype=F64, device=dev)
    f = lambda x: x**2 - a
    r = ts.solve(f, bracket=(torch.zeros_like(a), torch.full_like(a, 20.0)),
                 ftol=1e-10, rtol=1e-13, early_exit=False)
    assert r.root.device.type == "cuda"
    assert r.ok
    assert torch.allclose(r.root, a.sqrt(), atol=1e-10)


def test_early_exit_off_gives_same_answer():
    a = torch.linspace(1.0, 10.0, 64, dtype=F64)
    f = lambda x: x**2 - a
    kw = dict(bracket=(torch.zeros_like(a), torch.full_like(a, 20.0)), ftol=1e-11, rtol=1e-13)
    r1 = ts.solve(f, **kw, early_exit=True)
    r2 = ts.solve(f, **kw, early_exit=False)
    assert torch.equal(r1.root, r2.root)


def test_result_reporting_is_useful():
    f = lambda x: x**2 + 1.0
    r = ts.solve(f, bracket=(t(-1.0, 3), t(1.0, 3)))
    s = r.summary()
    assert "NO_BRACKET" in s and "3" in s


def test_element_independence_is_bitwise():
    """A batched solve must give the same bits as solving each element alone."""
    b = torch.tensor([4.0, 9.0, 1e-8], dtype=F64)
    lo, hi = torch.zeros_like(b), torch.full_like(b, 2e6)
    batched = ts.solve(lambda x: x**2 - b, bracket=(lo, hi), ftol=1e-6)
    for i, v in enumerate(b):
        one = ts.solve(lambda x, v=v: x**2 - v,
                       bracket=(torch.zeros(1, dtype=F64), torch.full((1,), 2e6, dtype=F64)),
                       ftol=1e-6)
        assert float(batched.root[i]) == float(one.root[0])


def test_works_inside_no_grad():
    """Autograd derivatives must survive an outer inference_mode/no_grad block."""
    with torch.no_grad():
        r = ts.solve(lambda x: torch.cos(x) - x, bracket=(t(0.0), t(1.0)), ftol=1e-13)
    assert r.ok
    assert abs(float(r.root[0]) - 0.7390851332151607) < 1e-12


def test_unattainable_ftol_is_refused_not_faked():
    """|f| <= 1e-12 is impossible for f = 1e10 * (x - 2.5) in float64."""
    f = lambda x: 1e10 * (x - 2.5)
    r = ts.solve(f, bracket=(t(0.0), t(10.0)), ftol=1e-12)
    assert not r.ok
    assert int(r.status[0]) == Status.STAGNATED
    # ftol_rel scales against the residual magnitude and does converge
    r2 = ts.solve(f, bracket=(t(0.0), t(10.0)), ftol=1e-12, ftol_rel=1e-15)
    assert r2.ok and abs(float(r2.root[0]) - 2.5) < 1e-14


def test_agrees_with_brentq_on_hard_problems():
    from scipy import optimize

    cases = [
        (torch.sin, np.sin, 1.0, 4.0),
        (lambda x: x**3 - 2 * x - 5, lambda x: x**3 - 2 * x - 5, 2.0, 3.0),
        (lambda x: 1 / (x - 2) - 10, lambda x: 1 / (x - 2) - 10, 2.001, 4.0),
        (lambda x: torch.sign(x - 1) * torch.sqrt((x - 1).abs()),
         lambda x: np.sign(x - 1) * np.sqrt(abs(x - 1)), -3.0, 9.0),
        (lambda x: x - torch.exp(-x), lambda x: x - np.exp(-x), 0.0, 5.0),
    ]
    for ft, fn_, a, b in cases:
        r = ts.solve(ft, bracket=(t(a), t(b)), ftol=1e-12, max_iter=400)
        ref = optimize.brentq(fn_, a, b, maxiter=400)
        assert r.ok, r.summary()
        assert abs(float(r.root[0]) - ref) < 1e-9 * max(1.0, abs(ref))
