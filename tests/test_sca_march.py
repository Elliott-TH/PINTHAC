"""Conservation, wall balance, branch selection, and profile reconstruction."""
import numpy as np
import pytest
import torch
import torchsolve as ts

from pinthac.correlations import htc
from pinthac.sca import coolant, annular, annular_march, rod
from pinthac.sca.run import run_channel

GEOM = dict(type='annular', ri=.0035, ro=.0055, tci=.0006, tco=.0006,
            delta_i=.0001, delta_o=.0001, Pitch=.013)
COND = dict(L=4.27, N=5, Tin_i=623.15, Tin_o=623.15, Pnom=25.,
            mdot_i=.01, mdot_o=.06, q0=10000.)


@pytest.fixture(scope='module')
def marched():
    return run_channel(GEOM, COND)


def test_march_conserves_each_cell_and_entire_channel(marched):
    out = marched['result']
    assert marched['convergence']['ok']
    dz = COND['L']/COND['N']
    assert np.allclose(out['q_i']+out['q_o'], out['qp'], atol=1e-9)
    for side in ('i', 'o'):
        assert np.allclose((out[f'h_out_{side}']-out[f'h_{side}'])*COND[f'mdot_{side}'],
                           out[f'q_{side}']*dz, rtol=1e-11)
        assert np.allclose(out[f'h_{side}'][1:], out[f'h_out_{side}'][:-1])
        assert out[f'dP_{side}'][0] > 0
    assert abs(sum(COND[f'mdot_{s}']*(out[f'h_out_{s}'][-1]-out[f'h_{s}'][0])
                   for s in ('i', 'o'))-sum(out['qp'])*dz) < 1e-8


def test_march_wall_flux_and_temperature_chain(marched):
    out = marched['result']
    geom = annular.geometry(dict(GEOM, **COND))
    for side, wall in (('i', 'Tci_i'), ('o', 'Tco_o')):
        for kind in ('conv', 'gap'):
            profile = out[f'htc_{kind}_{side}']
            assert profile.shape == out['z'].shape == (COND['N'],)
            assert np.all(np.isfinite(profile))
            assert np.all(profile > 0)
        assert np.allclose(out[f'htc_conv_{side}']*(out[wall]-out[f'Tm_{side}']),
                           out[f'q_{side}']/geom[f'Per_{side}'], rtol=2e-9)
    assert np.all(out['Tm_i'] <= out['Tci_i'])
    assert np.all(out['Tci_i'] <= out['Tco_i'])
    assert np.all(out['Tco_i'] <= out['Tfo_i'])
    assert np.all(out['Tm_o'] <= out['Tco_o'])
    assert np.all(out['Tco_o'] <= out['Tci_o'])
    assert np.all(out['Tci_o'] <= out['Tfo_o'])


def test_coefficients_reconstruct_surfaces_and_maximum(marched):
    out = marched['result']
    r = np.linspace(GEOM['ri'], GEOM['ro'], 1001)
    profile = annular_march.fuel_profile(out, r)
    assert np.allclose(profile[:, 0], out['Tfo_i'], atol=2e-5)
    assert np.allclose(profile[:, -1], out['Tfo_o'], atol=2e-5)
    assert np.allclose(profile.max(axis=1), out['Tf_max'], atol=.002)
    flux_i = 2*np.pi*out['C1']-np.pi*out['q3']*GEOM['ri']**2
    assert np.allclose(flux_i, out['q_i'], atol=1e-4)


def test_branch_scan_selects_lower_root_even_with_full_bracket_sign_change():
    # Three separated roots; a full-interval solve need not choose the first.
    residual = lambda T: (T-610)*(T-650)*(T-690)/1e5
    root = htc._solve_Tw_scw(residual, torch.tensor(600.), 720., 'synthetic', anchor=660.)
    assert float(root) == pytest.approx(610., abs=1e-6)


def test_branch_scan_reports_missing_root():
    with pytest.raises(ts.SolverFailure):
        htc._solve_Tw_scw(lambda T: T*0+1, torch.tensor(600.), 700., 'no root', anchor=650.)


@pytest.mark.parametrize('name', ['Swenson', 'Chen'])
def test_shared_flux_solver_batched_balance_and_gradient(name):
    table = coolant.build_table('scw', 25., device='cpu')
    props, _ = coolant.make_lookups('scw', 25., table=table)
    temperatures = [600., 630., 650.] if name == 'Chen' else [600., 640., 680.]
    T = torch.tensor(temperatures, dtype=torch.float64)
    q = torch.tensor([2e5, 3e5, 4e5], dtype=torch.float64, requires_grad=True)
    result = getattr(htc.SCW, name)(props(T), props, 1000., .008, q, T,
                                  psi=1.2, anchor=658., hi=1200., return_state=True)
    assert torch.allclose(result['htc']*(result['Tw']-T), q, rtol=2e-9, atol=1e-5)
    gradient, = torch.autograd.grad(result['Tw'].sum(), q)
    assert torch.isfinite(gradient).all()
    assert (gradient > 0).all()


def test_zero_power_annular_channel_has_uniform_temperature():
    cond = dict(COND, N=2, Tin_i=700., Tin_o=700., Pnom=.1, q0=0.)
    out = run_channel(GEOM, cond, coolant='sodium')['result']
    assert np.allclose(out['q_i'], 0, atol=1e-8)
    assert np.allclose(out['Tf_max'], 700., atol=1e-6)
    assert np.allclose(out['T_out_i'], 700., atol=1e-6)


def test_rod_tallies_all_radial_fields():
    geom = dict(type='rod', pitch=.0125, rco=.0045, tc=.00063, delta=.0005, kc=24.)
    cond = dict(G=1200., pval=.1, Tin=700., q0=8000., L=3., N=3)
    out = run_channel(geom, cond, coolant='sodium')['result']
    for key in ('Tm', 'htc_conv', 'Tco', 'Tci', 'Tfo', 'Tf_max'):
        assert len(out[key]) == cond['N']
    assert np.allclose(np.asarray(out['htc_conv'])*(np.asarray(out['Tco'])-out['Tm']),
                       np.asarray(out['qp'])/(2*np.pi*geom['rco']))


def test_signed_swenson_flux_uses_cooling_branch():
    table = coolant.build_table('scw', 25., device='cpu')
    props, _ = coolant.make_lookups('scw', 25., table=table)
    T = torch.tensor([680., 620.], dtype=torch.float64)
    flux = torch.tensor([-1e5, 2e5], dtype=torch.float64)
    out = htc.SCW.Swenson(props(T), props, 1000., .008, flux, T,
                           anchor=658., lo=500., hi=1200., return_state=True)
    assert out['Tw'][0] < T[0]
    assert out['Tw'][1] > T[1]
    assert torch.allclose(out['htc']*(out['Tw']-T), flux, rtol=2e-9, atol=1e-5)


def test_interchannel_transfer_at_zero_generated_power():
    cond = dict(COND, N=2, Tin_i=710., Tin_o=700., Pnom=.1, q0=0., mdot_i=.1, mdot_o=.1)
    out = run_channel(GEOM, cond, coolant='sodium')['result']
    assert out['q_i'][0] < 0 < out['q_o'][0]
    assert np.allclose(out['q_i']+out['q_o'], 0, atol=1e-9)
    assert out['T_out_i'][-1] < cond['Tin_i']
    assert out['T_out_o'][-1] > cond['Tin_o']


def test_chen_missing_heating_root_is_reported():
    table = coolant.build_table('scw', 25., device='cpu')
    props, _ = coolant.make_lookups('scw', 25., table=table)
    T = torch.tensor(680., dtype=torch.float64)
    with pytest.raises(ts.SolverFailure, match='no sampled sign-changing branch'):
        htc.SCW.Chen(props(T), props, 1000., .008, 4e5, T,
                     psi=1.2, anchor=658., hi=1200.)


def test_zero_flux_in_mixed_chen_batch_does_not_require_a_heating_root():
    table = coolant.build_table('scw', 25., device='cpu')
    props, _ = coolant.make_lookups('scw', 25., table=table)
    T = torch.tensor([680., 630.], dtype=torch.float64)
    flux = torch.tensor([0., 2e5], dtype=torch.float64, requires_grad=True)
    out = htc.SCW.Chen(props(T), props, 1000., .008, flux, T,
                       anchor=658., hi=1200., return_state=True)
    assert out['Tw'][0] == T[0]
    gradient, = torch.autograd.grad(out['Tw'].sum(), flux)
    assert torch.isfinite(gradient).all()
    assert (gradient > 0).all()


def test_heat_split_can_converge_when_an_extreme_trial_has_no_film_root(monkeypatch):
    from pinthac.sca import film
    original = film.solve
    rejected = []

    def limited(*args, **kwargs):
        flux = args[4]
        if flux > 35000.:
            rejected.append(flux)
            raise ts.SolverFailure('trial outside film model domain', None)
        return original(*args, **kwargs)

    monkeypatch.setattr(film, 'solve', limited)
    cond = dict(COND, N=2, Tin_i=700., Tin_o=700., Pnom=.1, q0=1000.)
    out = run_channel(GEOM, cond, coolant='sodium')['result']
    assert rejected
    assert np.allclose(out['q_i']+out['q_o'], out['qp'], atol=1e-8)
    assert np.max(np.abs(out['radial_residual'])) <= 1e-4
