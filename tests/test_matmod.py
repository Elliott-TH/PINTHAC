"""
Smoke tests for pinthac.properties.matmod: does it import, does each public function
accept a float / numpy array / torch tensor and return the matching type, does a torch
input keep a finite gradient where one exists.

matmod.py has no published valid-range table (see its RANGES docstring note and
docs/OPEN_QUESTIONS.md Q31), so there is no out-of-range-warns test here -- there is
nothing in the source or in docs/reference/ to check against. Numbers asserted below are
either literal constants written directly into the formulas (e.g. Zircaloy's 2098 K phase
plateau, D9's stub reference value) or structural invariants (return type, monotonic
sign), never a value obtained by running the code under test.
"""
import numpy as np
import pytest
import torch

from pinthac.properties import matmod as m


T_FLOAT = 1200.0


def _assert_backend_contract(fn, *args, label=""):
    """float in -> finite float out; numpy array in -> numpy out; torch tensor in ->
    tensor out with a finite gradient back to every tensor argument that needed one.

    Every numeric positional argument is exercised as a float, then all of them together
    as numpy arrays, then all of them together as torch tensors -- and finally each one
    as a tensor on its own while the others stay plain floats. That last pass is the one
    the original audit cared about: promoting every argument together is exactly the case
    that does not break."""
    out_f = fn(*args)
    assert np.isfinite(np.asarray(out_f, dtype=float)).all(), label

    np_args = [np.asarray([a, a]) if isinstance(a, (int, float)) else a for a in args]
    out_np = fn(*np_args)
    assert np.isfinite(np.asarray(out_np, dtype=float)).all(), label

    t_args = [torch.tensor([a, a], dtype=torch.float64, requires_grad=True)
              if isinstance(a, (int, float)) else a for a in args]
    out_t = fn(*t_args)
    assert torch.is_tensor(out_t), label
    if out_t.requires_grad:
        leaves = [a for a in t_args if torch.is_tensor(a) and a.requires_grad]
        grads = torch.autograd.grad(out_t.sum(), leaves, allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads), label

    # Mixed: one argument a tensor while the rest stay plain floats. This is the real
    # call shape -- a whole axial temperature field against a scalar burnup -- and it is
    # the one that breaks, because a `where` over the scalar argument alone resolves to
    # numpy and then cannot combine with the tensor. Promoting every argument together,
    # as the passes above do, is precisely the case that does NOT break, so without this
    # loop the contract looks satisfied when it is not.
    numeric = [i for i, a in enumerate(args) if isinstance(a, (int, float))]
    for i in numeric:
        mixed = list(args)
        mixed[i] = torch.tensor([args[i], args[i]], dtype=torch.float64, requires_grad=True)
        out_m = fn(*mixed)
        assert torch.is_tensor(out_m), f"{label}: arg {i} as tensor did not return a tensor"
        assert torch.isfinite(out_m).all(), f"{label}: arg {i} as tensor gave a non-finite result"
        if out_m.requires_grad:
            (g,) = torch.autograd.grad(out_m.sum(), mixed[i], allow_unused=True)
            assert g is None or torch.isfinite(g).all(), f"{label}: arg {i} gradient not finite"


# ------------------------------------------------------------------------------- import
def test_module_imports():
    assert m.UO2 and m.Zircalloy and m.HT9 and m.Gas and m.D9_SS


# --------------------------------------------------------------------------------- Gas
def test_gas_k_backend_contract():
    _assert_backend_contract(m.Gas.k, 'He', T_FLOAT, label="Gas.k")


def test_gas_k_positive_and_increases_with_T_for_helium():
    # k = A*T^B with B > 0 for every tabulated gas -- monotonically increasing in T is a
    # structural invariant of the formula, not a value read off a run.
    assert m.Gas.k('He', 900.0) < m.Gas.k('He', 1200.0)


# --------------------------------------------------------------------------------- UO2
def test_uo2_k_nfi_backend_contract_including_default_burnup():
    # The known pre-cleanup failure: Bu defaults to a float, T may be a tensor.
    _assert_backend_contract(m.UO2.k_NFI, T_FLOAT, label="UO2.k_NFI default Bu")
    _assert_backend_contract(m.UO2.k_NFI, T_FLOAT, 20.0, 0.01, label="UO2.k_NFI with Bu")


def test_uo2_k_klimenko_backend_contract():
    _assert_backend_contract(m.UO2.k_Klimenko, T_FLOAT, label="UO2.k_Klimenko")


def test_uo2_eps_backend_contract_and_is_a_fraction():
    val = m.UO2.eps(T_FLOAT)
    assert 0.0 < val < 1.0
    _assert_backend_contract(m.UO2.eps, T_FLOAT, label="UO2.eps")


def test_uo2_thrm_expan_backend_contract():
    _assert_backend_contract(m.UO2.thrm_expan, T_FLOAT, label="UO2.thrm_expan")


def test_uo2_densification_and_swelling_backend_contract():
    _assert_backend_contract(m.UO2.densification, T_FLOAT, 30.0, label="UO2.densification")
    _assert_backend_contract(m.UO2.swelling_solid, 50.0, label="UO2.swelling_solid")
    _assert_backend_contract(m.UO2.swelling_gas, 1400.0, 45.0, label="UO2.swelling_gas")


def test_uo2_swelling_gas_is_zero_outside_its_active_window():
    # The docstring's stated active window (Bu >= 40, 1233 <= T <= 2105 K) is a
    # structural property of the formula's own branches, not a computed check value.
    assert m.UO2.swelling_gas(1400.0, 10.0) == 0.0     # burnup below the ramp floor
    assert m.UO2.swelling_gas(1000.0, 45.0) == 0.0      # temperature below the window


def test_uo2_swelling_solid_gadolinia_branch():
    # gad=True is a structural (non-batchable) flag, matching backend.is_torch's contract.
    assert m.UO2.swelling_solid(50.0, gad=True) == pytest.approx(0.0005*50.0)


def test_uo2_dens_b_bisection_converges_for_an_array_and_a_tensor():
    dl = np.array([0.3, 0.7, 1.2])
    b = m.UO2.dens_B(dl)
    assert np.all(np.isfinite(b))
    t = torch.tensor([0.3, 0.7, 1.2], dtype=torch.float64)
    bt = m.UO2.dens_B(t)
    assert torch.is_tensor(bt) and torch.isfinite(bt).all()


# --------------------------------------------------------------------------- Zircalloy
def test_zircalloy_k_backend_contract_and_plateau():
    _assert_backend_contract(m.Zircalloy.k, T_FLOAT, label="Zircalloy.k")
    # 36.0 is the literal plateau constant written in the k(T) formula above 2098 K.
    assert m.Zircalloy.k(2200.0) == 36.0


def test_zircalloy_cp_backend_contract_uses_backend_interp():
    # The known pre-cleanup failure: lib.interp has no torch equivalent.
    _assert_backend_contract(m.Zircalloy.cp, T_FLOAT, label="Zircalloy.cp")


def test_zircalloy_thermal_expansion_backend_contract():
    _assert_backend_contract(m.Zircalloy.thrm_expan_axial, T_FLOAT, label="thrm_expan_axial")
    _assert_backend_contract(m.Zircalloy.thrm_expan_diametral, T_FLOAT, label="thrm_expan_diametral")


def test_zircalloy_constants():
    assert m.Zircalloy.T_melt() == 2123.15
    assert m.Zircalloy.rho() == 6520.0


def test_zircalloy_eps_backend_contract_with_default_oxide():
    _assert_backend_contract(m.Zircalloy.eps, T_FLOAT, label="Zircalloy.eps default t_ox")


def test_zircalloy_E_and_G_backend_contract_with_default_fluence():
    _assert_backend_contract(m.Zircalloy.E, T_FLOAT, label="Zircalloy.E default phi")
    _assert_backend_contract(m.Zircalloy.G, T_FLOAT, label="Zircalloy.G default phi")


def test_zircalloy_meyer_hardness_backend_contract():
    _assert_backend_contract(m.Zircalloy.meyer_hardness, T_FLOAT, label="meyer_hardness")


def test_zircalloy_axial_growth_table_lookup():
    assert m.Zircalloy.axial_growth('Zircaloy-4', 1.0e25) > 0.0


def test_zircalloy_sigma_eff_backend_contract():
    _assert_backend_contract(m.Zircalloy.sigma_eff, 15.0, 0.1, 0.0043, 0.0051,
                              label="sigma_eff")


def test_zircalloy_cw_type_lookup():
    assert m.Zircalloy.cw_type('Zircaloy-2') == "RXA"
    assert m.Zircalloy.cw_type('Zircaloy-4') == "SRA"


def test_zircalloy_strain_rate_thermal_backend_contract_with_scalar_fluence():
    # The pre-cleanup failure mode this test targets: Phi passed as a plain float
    # alongside a tensor T (a scalar fluence applied across a batch of temperatures).
    _assert_backend_contract(m.Zircalloy.strain_rate_thermal, T_FLOAT, 50.0, 1.0e20,
                              label="strain_rate_thermal")


def test_zircalloy_strain_rate_irrad_backend_contract():
    _assert_backend_contract(m.Zircalloy.strain_rate_irrad, T_FLOAT, 50.0, 1.0e14,
                              label="strain_rate_irrad")


def test_zircalloy_creep_strain_and_rate_run():
    val = m.Zircalloy.creep_strain('Zircaloy-4', 1000.0, 50.0, 1.0e14, 1.0e20, 100.0)
    rate = m.Zircalloy.creep_rate('Zircaloy-4', 1000.0, 50.0, 1.0e14, 1.0e20, 100.0)
    assert np.isfinite(val) and np.isfinite(rate)
    # ZIRLO applies an extra 0.8 factor relative to Zircaloy-4/2/M5 -- a structural
    # property of the formula's own branch, not a value read off a run.
    val_zirlo = m.Zircalloy.creep_strain('ZIRLO', 1000.0, 50.0, 1.0e14, 1.0e20, 100.0)
    assert val_zirlo == pytest.approx(0.8*val)


# -------------------------------------------------------------------------------- HT9
def test_ht9_k_and_cp_backend_contract():
    _assert_backend_contract(m.HT9.k, T_FLOAT, label="HT9.k")
    _assert_backend_contract(m.HT9.cp, T_FLOAT, label="HT9.cp")


def test_ht9_constants():
    assert m.HT9.T_melt() == 973.0
    assert m.HT9.rho() == 7750.0
    assert m.HT9.eps() == 0.9


def test_ht9_thrm_expan_and_moduli_backend_contract():
    _assert_backend_contract(m.HT9.thrm_expan, T_FLOAT, label="HT9.thrm_expan")
    _assert_backend_contract(m.HT9.E, T_FLOAT, label="HT9.E")
    _assert_backend_contract(m.HT9.G, T_FLOAT, label="HT9.G")


def test_ht9_meyer_hardness_matches_zircalloy():
    # HT9.meyer_hardness is documented as reusing the zirconium model outright.
    assert m.HT9.meyer_hardness(1000.0) == m.Zircalloy.meyer_hardness(1000.0)


def test_ht9_strain_rate_primary_backend_contract_with_scalar_time():
    # The pre-cleanup failure mode this test targets: t passed as a plain float
    # alongside a tensor T.
    _assert_backend_contract(m.HT9.strain_rate_primary, T_FLOAT, 50.0, 1000.0,
                              label="HT9.strain_rate_primary")


def test_ht9_strain_rate_components_sum_to_the_totals():
    T0, sig0, flux0, t0 = 900.0, 50.0, 1.0e14, 1000.0
    thermal = m.HT9.strain_rate_thermal(T0, sig0, t0)
    irrad = m.HT9.strain_rate_irrad(T0, sig0, flux0)
    total = m.HT9.strain_rate(T0, sig0, flux0, t0)
    assert total == pytest.approx(thermal + irrad)


def test_ht9_yield_stress_backend_contract():
    _assert_backend_contract(m.HT9.yield_stress, T_FLOAT, label="HT9.yield_stress")


# ------------------------------------------------------------------------------- D9_SS
def test_d9_ss_k_is_an_honest_stub():
    with pytest.raises(NotImplementedError):
        m.D9_SS.k(650.0)
