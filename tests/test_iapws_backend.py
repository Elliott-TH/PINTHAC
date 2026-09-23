"""The backend contract for the IAPWS modules: types in, types out, devices, and gradients.

Why this file is separate from test_iapws_verification.py: that file proves the physics is
right against published check values, and says nothing about how the numbers are carried.
This one proves the carrying, which is the other half of what CONTRIBUTING.md section 5
asks for -- every public function takes floats, NumPy arrays and torch tensors and returns
the same kind; a torch input keeps its device and its autograd graph; nothing about the
process's global state is touched on import.

The gradient tests compare autograd against a central difference rather than against a
stored number. That is the only way to catch the failure these modules are most exposed
to: a solver that converges to the right value and returns it detached, so the value is
right and the derivative is silently zero or stale. Three of the four inversions here were
in exactly that state before this pass.
"""
import numpy as np
import pytest
import torch

from pinthac.properties import iapws97 as if97
from pinthac.properties import iapws_backend as wb
from pinthac.properties import iapws_transport as transport
from pinthac.properties.iapws95 import IAPWS95


# A representative supercritical state, the regime the single-channel solvers run in.
T_REF, P_REF = 650.0, 25.0


# ---------------------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------------------
def test_import_does_not_change_global_torch_dtype():
    """CONTRIBUTING.md section 5 rule 6, named in the rule itself.

    The IAPWS-97 module used to call torch.set_default_dtype(torch.float64) at module
    scope, which promoted every tensor and every neural network built anywhere else in the
    same process. A test asserting the default is float32 would be asserting a torch
    default rather than this library's behaviour, so the assertion is that importing the
    property modules leaves it wherever it already was.
    """
    before = torch.get_default_dtype()
    import importlib
    importlib.reload(wb)
    assert torch.get_default_dtype() == before


def test_import_does_not_allocate_on_the_gpu():
    """The coefficient tables are built on the cpu and copied to a device on first use.

    Building them on the accelerator at import instead would initialize a GPU context in
    every process that so much as imports pinthac.properties, including ones that never
    touch a tensor.
    """
    for table in (IAPWS95.C0, IAPWS95.C1, IAPWS95.C2, IAPWS95.C3, IAPWS95.C4,
                  if97.R1.Coeff, if97.R3.Coeff, if97.R4.Coeff,
                  transport.VISC.Hij, transport.COND.Aij):
        assert table.device.type == 'cpu'


def test_host_inputs_are_evaluated_on_the_host():
    """A float or NumPy input is host data and stays on the host, even with a GPU present.

    This is pinthac/backend.py's rule -- tensor in, torch and its device; anything else,
    NumPy -- and it is also 22x faster for the batch sizes the channel solvers use.
    """
    assert wb.device.type == 'cpu'


# ---------------------------------------------------------------------------------------
# Type round-trip
# ---------------------------------------------------------------------------------------
def test_scalar_in_scalar_out():
    rho = IAPWS95.rho_Tp(T_REF, P_REF)
    assert isinstance(rho, float)
    d = IAPWS95.helmholtz(rho, T_REF)
    for value in (IAPWS95.p(d), IAPWS95.h(d), IAPWS95.cp(d), IAPWS95.c(d),
                  IAPWS95.mu(d), IAPWS95.lam(d)):
        assert isinstance(value, float)
    assert isinstance(IAPWS95.T_sat(10.0), float)
    assert isinstance(IAPWS95.p_sat(500.0), float)
    assert isinstance(if97.R1.Tph(3.0, 500.0), float)
    assert isinstance(transport.SIGMA.sigma(400.0), float)
    assert isinstance(if97.region(3.0, 300.0), int)


def test_numpy_in_numpy_out():
    T = np.full(4, T_REF)
    rho = IAPWS95.rho_Tp(T, P_REF)
    assert isinstance(rho, np.ndarray) and rho.shape == (4,)
    d = IAPWS95.helmholtz(rho, T)
    for value in (IAPWS95.p(d), IAPWS95.h(d), IAPWS95.mu(d), IAPWS95.lam(d)):
        assert isinstance(value, np.ndarray) and value.shape == (4,)


def test_list_is_treated_as_numpy():
    """Lists are not in CONTRIBUTING's three-type contract, and a dict of lists cannot be
    done arithmetic on, so a list is converted and answered as a NumPy array.
    """
    rho = IAPWS95.rho_Tp([600.0, 700.0], 25.0)
    assert isinstance(rho, np.ndarray) and rho.shape == (2,)


def test_torch_in_torch_out_on_the_same_device():
    T = torch.full((4,), T_REF, dtype=torch.float64)
    rho = IAPWS95.rho_Tp(T, P_REF)
    assert isinstance(rho, torch.Tensor)
    assert rho.device == T.device
    d = IAPWS95.helmholtz(rho, T)
    for value in (IAPWS95.p(d), IAPWS95.h(d), IAPWS95.mu(d), IAPWS95.lam(d)):
        assert isinstance(value, torch.Tensor) and value.device == T.device


def test_float32_in_float32_out():
    """The arithmetic runs at float64 regardless -- the residual terms of Eq. (6) cancel to
    several digits and float32 leaves noise instead of a pressure -- but the answer comes
    back in the caller's dtype so a float32 network is not silently promoted.
    """
    T = torch.tensor([T_REF], dtype=torch.float32)
    rho = IAPWS95.rho_Tp(T, 25.0)
    assert rho.dtype == torch.float32
    assert float(rho) == pytest.approx(float(IAPWS95.rho_Tp(T_REF, 25.0)), rel=1e-6)


def test_shape_is_preserved_and_inputs_broadcast():
    T = np.full((2, 3), T_REF)
    assert IAPWS95.rho_Tp(T, P_REF).shape == (2, 3)

    # A scalar pressure against an array temperature is the call every channel solver
    # makes; it must mean what it looks like it means.
    T_row = np.array([600.0, 650.0, 700.0])
    one_by_one = np.array([IAPWS95.rho_Tp(float(t), P_REF) for t in T_row])
    assert IAPWS95.rho_Tp(T_row, P_REF) == pytest.approx(one_by_one, rel=1e-12)


NUMPY_TORCH_CASES = [
    ("rho_Tp",  lambda x: IAPWS95.rho_Tp(x, 25.0),          np.array([400.0, 500.0, 600.0])),
    ("T_sat",   lambda x: IAPWS95.T_sat(x),                 np.array([1.0, 10.0, 20.0])),
    ("T_hp",    lambda x: IAPWS95.T_hp(x, 25.0),            np.array([2000.0, 2500.0, 3000.0])),
    ("R1.Tph",  lambda x: if97.R1.Tph(3.0, x),              np.array([200.0, 500.0, 1200.0])),
    ("R2.Tph",  lambda x: if97.R2.Tph(x, 3000.0),           np.array([0.5, 5.0, 60.0])),
    ("R3.rho",  lambda x: if97.R3.rho_pT(25.5837018, x),    np.array([650.0, 660.0, 670.0])),
    ("sigma",   lambda x: transport.SIGMA.sigma(x),         np.array([300.0, 450.0, 600.0])),
    ("mu",      lambda x: transport.VISC.mu(998.0, x),      np.array([300.0, 350.0, 400.0])),
]


@pytest.mark.parametrize("name,call,x", NUMPY_TORCH_CASES, ids=[c[0] for c in NUMPY_TORCH_CASES])
def test_numpy_and_torch_agree(name, call, x):
    """The same numbers whichever way they are carried.

    Worth having because the two paths are not the same code below the entry point --
    NumPy input is converted once and evaluated on the host, torch input keeps its own
    device and dtype and skips the conversion. A table that landed on one path only, or a
    branch selected by input type rather than by value, shows up here and nowhere else.
    """
    got_np = call(x)
    got_torch = call(torch.tensor(x)).numpy()
    assert got_torch == pytest.approx(got_np, rel=1e-12)


# ---------------------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------------------
def central_difference(fn, x, eps):
    """d fn/dx by central difference, at detached x."""
    return (fn(x.detach() + eps) - fn(x.detach() - eps)) / (2.0 * eps)


def test_rho_Tp_is_differentiable_in_both_arguments():
    """The most important gradient in the library: the DeepONet surrogate is trained
    through this call, so a detached density makes every downstream gradient wrong rather
    than merely inaccurate.
    """
    T = torch.tensor([500.0, 650.0, 800.0], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([25.0, 25.0, 25.0], dtype=torch.float64, requires_grad=True)

    rho = IAPWS95.rho_Tp(T, p)
    gT, gp = torch.autograd.grad(rho.sum(), [T, p])

    assert torch.all(gT != 0.0)
    fd_T = central_difference(lambda x: IAPWS95.rho_Tp(x, p.detach()), T, 1.0e-4)
    fd_p = central_difference(lambda x: IAPWS95.rho_Tp(T.detach(), x), p, 1.0e-4)
    assert gT.numpy() == pytest.approx(fd_T.numpy(), rel=1e-6)
    assert gp.numpy() == pytest.approx(fd_p.numpy(), rel=1e-6)


def test_T_hp_is_differentiable_in_both_arguments():
    """dT/dh is 1/cp, and dT/dp at fixed h comes through the nested density solve, so this
    only works if rho_Tp is differentiable too.
    """
    h = torch.tensor([2000.0, 2500.0, 3000.0], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([25.0, 25.0, 25.0], dtype=torch.float64, requires_grad=True)

    T = IAPWS95.T_hp(h, p)
    gh, gp = torch.autograd.grad(T.sum(), [h, p])

    fd_h = central_difference(lambda x: IAPWS95.T_hp(x, p.detach()), h, 1.0e-3)
    fd_p = central_difference(lambda x: IAPWS95.T_hp(h.detach(), x), p, 1.0e-4)
    assert gh.numpy() == pytest.approx(fd_h.numpy(), rel=1e-5)
    assert gp.numpy() == pytest.approx(fd_p.numpy(), rel=1e-5)

    # dT/dh must be 1/cp exactly, which is a check on the construction and not just on
    # the finite difference.
    rho = IAPWS95.rho_Tp(T.detach(), p.detach())
    cp = IAPWS95.cp(IAPWS95.helmholtz(rho, T.detach()), units='kJ')
    assert gh.numpy() == pytest.approx((1.0 / cp).numpy(), rel=1e-6)


def test_T_sat_is_differentiable_in_pressure():
    """dT_sat/dp is the reciprocal of the Clausius-Clapeyron slope, which is how it is
    constructed -- so this checks the construction against the finite difference of the
    bisection it sits on top of.
    """
    p = torch.tensor([1.0, 10.0, 20.0], dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(IAPWS95.T_sat(p).sum(), p)[0]
    fd = central_difference(IAPWS95.T_sat, p, 1.0e-5)
    assert g.numpy() == pytest.approx(fd.numpy(), rel=1e-6)


def test_saturation_is_differentiable_in_temperature():
    T = torch.tensor([400.0, 500.0, 600.0], dtype=torch.float64, requires_grad=True)
    sat = IAPWS95.saturation(T)
    g = torch.autograd.grad(sat['p'].sum(), T)[0]
    fd = central_difference(lambda x: IAPWS95.saturation(x)['p'], T, 1.0e-3)
    assert g.numpy() == pytest.approx(fd.numpy(), rel=1e-5)


def test_if97_region3_density_solve_is_differentiable():
    """Region 3 is the one IF97 region that has to be inverted, so it carries the same
    correction-step construction as IAPWS95.rho_Tp and needs the same check.
    """
    p = torch.tensor([25.0, 30.0], dtype=torch.float64, requires_grad=True)
    T = torch.tensor([650.0, 660.0], dtype=torch.float64, requires_grad=True)
    rho = if97.R3.rho_pT(p, T)
    gp_, gT = torch.autograd.grad(rho.sum(), [p, T])
    fd_p = central_difference(lambda x: if97.R3.rho_pT(x, T.detach()), p, 1.0e-4)
    fd_T = central_difference(lambda x: if97.R3.rho_pT(p.detach(), x), T, 1.0e-4)
    assert gp_.numpy() == pytest.approx(fd_p.numpy(), rel=1e-5)
    assert gT.numpy() == pytest.approx(fd_T.numpy(), rel=1e-5)


PATHOLOGICAL_STATES = [
    (322.0, 647.096, "exactly the critical point"),
    (322.0, 647.0,   "the critical density just below T_c"),
    (1.0e-6, 900.0,  "the dilute-gas limit, where the conductivity enhancement divides by rho"),
    (1188.0, 300.0,  "cold compressed liquid"),
]


@pytest.mark.parametrize("rho,T,why", PATHOLOGICAL_STATES)
def test_gradients_stay_finite_at_awkward_states(rho, T, why):
    """NaN in an unselected where() branch still reaches the backward pass.

    Three places in these modules evaluate both sides of a branch and select afterwards --
    the non-analytic terms of IAPWS-95 at delta = 1, and the two critical enhancements,
    where delta-chi is clamped at zero and then raised to a fractional power whose slope
    at zero is infinite. In each case the forward value is fine and only the gradient is
    poisoned, so a test that checks values would pass while a surrogate trained through
    the same call silently filled with NaN.
    """
    rho_t = torch.tensor([rho], dtype=torch.float64, requires_grad=True)
    T_t = torch.tensor([T], dtype=torch.float64, requires_grad=True)
    d = IAPWS95.helmholtz(rho_t, T_t)

    for name, value in [('p', IAPWS95.p(d)), ('h', IAPWS95.h(d)), ('s', IAPWS95.s(d)),
                        ('cv', IAPWS95.cv(d)), ('cp', IAPWS95.cp(d)),
                        ('c', IAPWS95.c(d)), ('mu', IAPWS95.mu(d)),
                        ('lam', IAPWS95.lam(d))]:
        assert torch.isfinite(value).all(), f"{name} is not finite at {why}"
        g_rho, g_T = torch.autograd.grad(value.sum(), [rho_t, T_t], retain_graph=True)
        assert torch.isfinite(g_rho).all(), f"d{name}/drho is not finite at {why}"
        assert torch.isfinite(g_T).all(), f"d{name}/dT is not finite at {why}"


def test_critical_point_reproduces_the_critical_pressure():
    """Eq. (6)'s two non-analytic terms exist so the equation is exact at the critical
    point; the delta = 1 nudge in Phir must not spoil that.
    """
    d = IAPWS95.helmholtz(322.0, 647.096)
    assert IAPWS95.p(d, units='MPa') == pytest.approx(22.064, rel=1e-6)


def test_surface_tension_is_zero_and_finite_above_the_critical_point():
    """There is no interface above T_c. tau^1.256 of a negative tau is NaN, and torch
    carries that NaN through the backward pass even when where() discarded it, so the
    supercritical branch is clamped rather than selected after the fact.
    """
    T = torch.tensor([600.0, 647.096, 700.0], dtype=torch.float64, requires_grad=True)
    sigma = transport.SIGMA.sigma(T)
    assert torch.isfinite(sigma).all()
    assert float(sigma[2].detach()) == 0.0
    g = torch.autograd.grad(sigma.sum(), T)[0]
    assert torch.isfinite(g).all()


# ---------------------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU on this machine")
def test_gpu_end_to_end_with_gradients():
    """A tensor on the accelerator is answered there, tables and all, with the graph intact.

    The coefficient tables used to be built on one fixed device at import, which is why
    ml/pinn.py carried a monkey-patch forcing the whole equation of state onto the cpu:
    the transport tables were on the host and mu/lam raised a device mismatch as soon as
    the equation of state ran on cuda.
    """
    dev = wb.accelerator
    T = torch.tensor([500.0, 650.0, 800.0], dtype=torch.float64, device=dev,
                     requires_grad=True)
    p = torch.tensor([25.0, 25.0, 25.0], dtype=torch.float64, device=dev,
                     requires_grad=True)

    rho = IAPWS95.rho_Tp(T, p)
    d = IAPWS95.helmholtz(rho, T)
    for value in (IAPWS95.h(d), IAPWS95.cp(d), IAPWS95.mu(d), IAPWS95.lam(d)):
        assert value.device.type == dev.type
        g = torch.autograd.grad(value.sum(), T, retain_graph=True)[0]
        assert torch.isfinite(g).all()

    # Same numbers as the cpu path.
    rho_cpu = IAPWS95.rho_Tp(T.detach().cpu(), p.detach().cpu())
    assert rho.detach().cpu().numpy() == pytest.approx(rho_cpu.numpy(), rel=1e-12)

    # The solvers and IF97 have to make the trip too.
    assert IAPWS95.T_sat(torch.tensor([10.0], dtype=torch.float64, device=dev)).device.type == dev.type
    assert IAPWS95.saturation(torch.tensor([500.0], dtype=torch.float64, device=dev))['p'].device.type == dev.type
    assert if97.R3.rho_pT(torch.tensor([25.0], dtype=torch.float64, device=dev),
                          torch.tensor([650.0], dtype=torch.float64, device=dev)).device.type == dev.type
