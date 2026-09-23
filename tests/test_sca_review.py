"""Regression tests for integration errors found during the library review."""
import math

import numpy as np
import pytest
import torch

from pinthac.sca import coolant, rod, run


def test_property_lookup_observes_inplace_changes():
    lookup = coolant.make_property('sodium', 0.1)
    temperature = np.array([700.0])
    before = lookup(['T', temperature], 'rho').copy()
    temperature[:] = 800.0
    after = lookup(['T', temperature], 'rho')
    assert not np.array_equal(before, after)
    assert after == pytest.approx(lookup.props_at(temperature)['rho'])


def test_nonfinite_reports_earliest_node_across_fields():
    result = dict(z=[0., 1., 2.], a=[1., 1., np.nan], b=[1., np.inf, 1.])
    status = run._scan_for_nonfinite(result, 'z', ('a', 'b'))
    assert (status['node'], status['field']) == (1, 'b')


def test_csv_preserves_coolant_and_defaults_for_empty_cells(tmp_path):
    path = tmp_path / 'cases.csv'
    path.write_text('pitch,rco,tc,delta,kc,G,pval,Tin,q0,N,coolant,htc\n'
                    '.0125,.0045,.00063,.0005,24,1200,.1,700,8000,,sodium,\n')
    case, = run.read_cases_csv(path, 'rod')
    assert case['coolant'] == 'sodium'
    assert 'htc' not in case
    assert 'N' not in case['conditions']


def test_rod_wall_balance_uses_heated_perimeter(monkeypatch):
    # A wall-dependent coefficient makes an incorrect surface area observable.
    def coefficient(bulk, wall, Tw, Tb, G, D):
        return 1000.0 + 2.0 * (Tw - Tb)

    monkeypatch.setattr(rod.htc.SCW, 'Swenson_dT', coefficient)
    lookup = lambda query, key: torch.ones_like(query[1])
    inputs = dict(G=1200., pitch=.0125, rco=.0045, tc=.00063, delta=.0005, kc=24.)
    Tb = torch.tensor(600., dtype=torch.float64)
    q = torch.tensor(1000., dtype=torch.float64)
    Tw, _ = rod.rod_node(lookup, Tb, 25., q, inputs, bundle_func=None)
    flux = q / (2 * math.pi * inputs['rco'])
    assert float((1000 + 2 * (Tw - Tb)) * (Tw - Tb)) == pytest.approx(float(flux))


def test_unheated_gap_has_no_temperature_jump():
    temperature = torch.tensor(700., dtype=torch.float64)
    result = rod.gap(torch.zeros_like(temperature), .0001, temperature, .004, .0039)
    assert float(result) == pytest.approx(700.)


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_uncertainty_stream_is_reproducible_on_tensor_device(device):
    from pinthac import uncertainty
    value = torch.ones(8, device=device, dtype=torch.float64)
    try:
        uncertainty.enable(42)
        first = uncertainty.perturb(value, .1)
        uncertainty.enable(42)
        second = uncertainty.perturb(value, .1)
        assert torch.equal(first, second)
        assert first.device == value.device
    finally:
        uncertainty.disable()


def test_annular_final_temperature_matches_enthalpy(monkeypatch):
    from pinthac.sca import annular
    inp = dict(ri=.0035, ro=.0055, L=1., N=3, Tin_i=700., Tin_o=700., Pnom=.1,
               mdot_i=1., mdot_o=1., q0=100.)
    monkeypatch.setattr(annular, 'geometry', lambda _: dict(G_i=1., D_i=1., G_o=1., D_o=1.))
    from types import SimpleNamespace
    monkeypatch.setattr(annular.ht, 'Ann_Theta',
                        lambda _: SimpleNamespace(x=np.array([300., 1000.]), y=np.array([0., 700.])))
    monkeypatch.setattr(annular.coolant_mod, 'make_lookups',
                        lambda *args: (lambda T: {'h': np.asarray(T)*2}, lambda h: h/2))
    fields = ('C1', 'C2', 'q3', 'Tf_max', 'r_Tf_max', 'Tci_i', 'Tci_o', 'Tco_i', 'Tco_o', 'Tfo_i', 'Tfo_o', 'Tcldi_ID', 'Tcldi_OD', 'Tcldo_ID', 'Tcldo_OD',
              'htc_conv_i', 'htc_conv_o', 'htc_gap_i', 'htc_gap_o')
    def closure(Ti, To, q, *args, **kwargs):
        return dict.fromkeys(fields, np.ones(3)) | {'q_i': .8*q, 'q_o': .2*q, 'radial_converged': True, 'radial_residual': 0.}
    monkeypatch.setattr(annular, 'closure', closure)
    monkeypatch.setattr(annular, 'pressure_drop', lambda T, *args: np.zeros_like(T))
    out = annular.solve_field(inp, coolant='sodium', outer_iter=1)
    assert np.allclose(out['Tm_i']*2, out['h_i'])
    assert np.allclose(out['Tm_o']*2, out['h_o'])
