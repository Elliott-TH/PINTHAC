"""
Permanent tests for pinthac.ml.losses (Phase 6, docs/PHASE67_BRIEF.md part A1).

These are the shared physics-residual terms factored out of deeponet.py/pinn.py's
previously-inline expressions. Checked for algebraic behaviour (residual is zero exactly
when the balance holds, the two energy-balance forms agree when dh=cp*dT, the
positivity penalty is zero above the floor and quadratic below it) rather than for
values obtained by running either surrogate.
"""
import torch

from pinthac.ml import losses


def test_coolant_energy_residual_h_zero_when_balance_holds():
    mdot = torch.tensor([0.3, 0.5, 1.2])
    qp = torch.tensor([1.0e4, 2.0e4, 3.0e4])
    dh_dz = qp / mdot   # exactly the balance mdot*dh/dz = q'(z)
    res = losses.coolant_energy_residual_h(mdot, dh_dz, qp)
    torch.testing.assert_close(res, torch.zeros_like(res))


def test_coolant_energy_residual_T_matches_h_form_via_cp():
    mdot = torch.tensor([0.3, 0.5, 1.2])
    cp = torch.tensor([5200.0, 5300.0, 5400.0])
    dT_dz = torch.tensor([10.0, -5.0, 3.0])
    qp = torch.tensor([1.0e4, 2.0e4, 3.0e4])

    res_T = losses.coolant_energy_residual_T(mdot, cp, dT_dz, qp)
    res_h = losses.coolant_energy_residual_h(mdot, cp * dT_dz, qp)   # dh/dz = cp*dT/dz
    torch.testing.assert_close(res_T, res_h)


def test_normalized_residual_loss_is_mean_squared_ratio():
    res = torch.tensor([2.0, -4.0, 6.0])
    scale = torch.tensor([2.0, 2.0, 2.0])
    loss = losses.normalized_residual_loss(res, scale)
    expected = torch.mean((res / scale) ** 2)   # mean(1, 4, 9) = 14/3
    torch.testing.assert_close(loss, expected)


def test_positivity_penalty_zero_above_floor():
    y = torch.tensor([1.0, 2.0, 5.0])
    assert losses.positivity_penalty(y, floor=0.0).item() == 0.0


def test_positivity_penalty_quadratic_below_floor():
    y = torch.tensor([-1.0, -2.0])
    floor = 0.0
    loss = losses.positivity_penalty(y, floor=floor)
    expected = torch.mean(torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(loss, expected)


def test_residuals_stay_differentiable():
    # The whole point of these terms is to sit inside an autograd physics loss --
    # confirm gradients flow through every one of them.
    mdot = torch.tensor(0.5, requires_grad=True)
    dh_dz = torch.tensor(200.0, requires_grad=True)
    qp = torch.tensor(1.0e4, requires_grad=True)
    res = losses.coolant_energy_residual_h(mdot, dh_dz, qp)
    loss = losses.normalized_residual_loss(res, torch.tensor(1.0e4))
    loss.backward()
    assert mdot.grad is not None and dh_dz.grad is not None and qp.grad is not None
