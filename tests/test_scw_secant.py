"""Bracket safeguards and convergence of the shared SCW secant iteration."""
import pytest
import torch

from pinthac.correlations import htc


def solve(f, lo, hi, **kw):
    return htc._bracketed_secant(f, torch.tensor(lo, dtype=torch.float64),
                               torch.tensor(hi, dtype=torch.float64), **kw)


def test_narrow_bracket_secant_converges_in_a_few_steps():
    result = solve(lambda x: x*x-2, [1.4, 1.41], [1.5, 1.42])
    assert result.ok
    assert torch.allclose(result.root, torch.full_like(result.root, 2**.5), atol=1e-9, rtol=0)
    assert int(result.iterations.max()) <= 5


def test_secant_keeps_all_evaluations_in_the_original_bracket():
    points = []
    def residual(x):
        points.append(x.clone())
        return x**10-1
    result = solve(residual, .01, 2., max_iter=200)
    assert result.ok
    assert float(result.root) == pytest.approx(1., abs=1e-8)
    assert all(bool(((x >= .01) & (x <= 2.)).all()) for x in points)


def test_step_stagnation_is_not_convergence_at_a_discontinuity():
    result = solve(lambda x: torch.where(x < .12345, -1., 1.), 0., 1., max_iter=100)
    assert not result.ok
    assert torch.isnan(result.root)


def test_exact_endpoints_and_failed_elements_are_independent():
    result = solve(lambda x: x*x-1, [1., 0., 2.], [2., 2., 3.])
    assert result.converged.tolist() == [True, True, False]
    assert result.iterations[0] == 0
    assert result.root[:2].tolist() == pytest.approx([1., 1.])


def test_nonfinite_residual_and_iteration_limit_are_failures():
    bad = solve(lambda x: x*float('nan'), 0., 1.)
    assert not bad.ok
    limited = solve(lambda x: x*x-2, 1., 2., max_iter=1)
    assert not limited.ok


def test_branch_scan_evaluates_the_temperature_grid_as_one_batch():
    shapes = []
    def residual(T):
        shapes.append(T.shape)
        return (T-610)*(T-650)*(T-690)/1e5
    roots = htc._solve_Tw_scw(residual, torch.tensor([600., 605.]), 720.,
                              'test', anchor=660.)
    assert roots.tolist() == pytest.approx([610., 610.], abs=1e-6)
    assert shapes[0] == (65, 2)


@pytest.mark.parametrize('name,method', [('Swenson', 'Swenson_dT'), ('Chen', 'Chen_SCW_dT')])
def test_flux_solution_and_implicit_gradient_match_analytic_root(monkeypatch, name, method):
    def coefficient(bulk, wall, Tw, Tb, G, D, *args):
        return 1000.+2.*(Tw-Tb)
    monkeypatch.setattr(htc.SCW, method, staticmethod(coefficient))
    properties = lambda T: dict(rho=500.+0*T, mu=5e-5+0*T, k=.4+0*T,
                                cp=5000.+0*T, h=5000.*T)
    T = torch.full((2, 3), 600., dtype=torch.float64)
    q = torch.tensor([[1e5, 2e5, 3e5], [4e5, 5e5, 6e5]], dtype=torch.float64,
                     requires_grad=True)
    result = getattr(htc.SCW, name)(properties(T), properties, 1000., .008, q, T,
                                    anchor=658., hi=1100., return_state=True)
    delta = (-1000.+torch.sqrt(1e6+8.*q))/4.
    assert torch.allclose(result['Tw'], T+delta, atol=1e-6, rtol=0)
    gradient, = torch.autograd.grad(result['Tw'].sum(), q)
    assert torch.allclose(gradient, 1./(1000.+4.*delta), atol=1e-10, rtol=1e-6)
