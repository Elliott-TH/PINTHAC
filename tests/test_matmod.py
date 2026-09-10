"""
Smoke tests for pinthac.properties.matmod: does it import, does each public function
accept a float / numpy array / torch tensor and return the matching type, does a torch
input keep a finite gradient where one exists.

Phase 2 found no published valid-range table for this module; Phase 3 populated RANGES
from docs/reference/MatLib_Info.pdf (PNNL-35702) -- see the module docstring and each
function's own "Valid range" field for the section numbers. A few representative
out-of-range-warns checks are added below now that there is something to check against;
the untested functions are the ones RANGES still has no entry for (see
docs/OPEN_QUESTIONS.md Q31 for the ones PNNL-35702 does not state a range for either).

Theta_Klimenko and Theta_NFI are checked against scipy.integrate.quad of the k function
each one integrates -- an independent numerical reference, not a value read off this
module's own output, per docs/PHASE3_BRIEF.md item 3 ("that check is the whole point:
it proves the integral matches the conductivity it claims to integrate").

Numbers asserted below are either literal constants written directly into the formulas
(e.g. Zircaloy's 2098 K phase plateau, D9's stub reference value) or structural invariants
(return type, monotonic sign), never a value obtained by running the code under test.
"""
import warnings

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from pinthac.properties import matmod as m
from pinthac.ranges import RangeWarning


T_FLOAT = 1200.0

# T_FLOAT (1200 K) sits inside UO2's and most of Zircaloy's validated ranges, but not
# inside the narrower ranges Phase 3 added for Zircaloy creep (570-625 K per PNNL-35702
# section 3.1.10.3) or for any HT-9 property (293-1073.15 K depending on the property, or
# 25-600 C for HT9.E/HT9.G specifically, which document their T argument in Celsius).
# Using T_FLOAT there would now trip pyproject.toml's "error::RangeWarning" filter, so
# these three narrower-range functions get their own in-range constants instead.
T_ZR_CREEP = 600.0    # K, inside Zircaloy's 570-625 K creep range
T_HT9 = 600.0         # K, inside every HT-9 property's own range (293-1073.15 K)
T_HT9_CELSIUS = 300.0  # C, inside HT9.E/HT9.G's 25-600 C range


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


@pytest.mark.parametrize("T1,T2", [(500.0, 1000.0), (1000.0, 2000.0), (2000.0, 3000.0),
                                    (300.0, 3120.0)])
def test_uo2_theta_klimenko_matches_quadrature_of_k_klimenko(T1, T2):
    # scipy.integrate.quad of k_Klimenko itself is the independent reference here, per
    # docs/PHASE3_BRIEF.md item 3 -- not a value obtained from Theta_Klimenko.
    theta_diff = float(m.UO2.Theta_Klimenko(np.array([T2]))[0]
                        - m.UO2.Theta_Klimenko(np.array([T1]))[0])
    quad_val, _ = quad(m.UO2.k_Klimenko, T1, T2)
    assert theta_diff == pytest.approx(quad_val, rel=1.0E-4)


def test_uo2_theta_klimenko_backend_contract():
    _assert_backend_contract(m.UO2.Theta_Klimenko, T_FLOAT, label="UO2.Theta_Klimenko")


def test_uo2_theta_klimenko_derivative_matches_k_klimenko():
    # Theta's whole purpose is to be an antiderivative of k_Klimenko -- checking that its
    # autograd derivative reproduces k_Klimenko is an independent structural check
    # (calculus, not a self-referential run of the function under test).
    T = torch.tensor([800.0, 1500.0, 2500.0], dtype=torch.float64, requires_grad=True)
    theta = m.UO2.Theta_Klimenko(T)
    (dTheta_dT,) = torch.autograd.grad(theta.sum(), T)
    k = m.UO2.k_Klimenko(T.detach())
    assert torch.allclose(dTheta_dT, k, rtol=1.0E-4)


@pytest.mark.parametrize("T_ref,T,Bu,f_gad", [(300.0, 1000.0, 0.0, 0.0),
                                               (300.0, 2000.0, 20.0, 0.0),
                                               (300.0, 2800.0, 50.0, 0.02)])
def test_uo2_theta_nfi_matches_quadrature_of_k_nfi(T_ref, T, Bu, f_gad):
    # Same independent-reference check as Theta_Klimenko, against k_NFI instead --
    # the whole point of docs/PHASE3_BRIEF.md item 3's verification requirement.
    theta = float(m.UO2.Theta_NFI(np.array([T]), Bu, f_gad, T_ref=T_ref, n=1000)[0])
    quad_val, _ = quad(lambda TT: m.UO2.k_NFI(TT, Bu, f_gad), T_ref, T)
    assert theta == pytest.approx(quad_val, rel=1.0E-4)


def test_uo2_theta_nfi_backend_contract_including_default_burnup():
    _assert_backend_contract(m.UO2.Theta_NFI, T_FLOAT, label="UO2.Theta_NFI default Bu")


def test_uo2_theta_nfi_derivative_matches_k_nfi():
    T = torch.tensor([1500.0, 2500.0], dtype=torch.float64, requires_grad=True)
    theta = m.UO2.Theta_NFI(T, 20.0, 0.0, n=500)
    (dTheta_dT,) = torch.autograd.grad(theta.sum(), T)
    k = m.UO2.k_NFI(T.detach(), 20.0, 0.0)
    assert torch.allclose(dTheta_dT, k, rtol=1.0E-3)


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
    # 2200 K is above the 285-1770 K validated range (PNNL-35702 section 3.1.1.3) by
    # design -- this checks the phase-transition plateau, which only exists above it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        # 36.0 is the literal plateau constant written in the k(T) formula above 2098 K.
        assert m.Zircalloy.k(2200.0) == 36.0


def test_zircalloy_cp_backend_contract_uses_backend_interp():
    # The known pre-cleanup failure: lib.interp has no torch equivalent.
    _assert_backend_contract(m.Zircalloy.cp, T_FLOAT, label="Zircalloy.cp")


def test_zircalloy_thermal_expansion_backend_contract():
    # thrm_expan_axial's range is 300-1273 K, so T_FLOAT (1200 K) fits; thrm_expan_diametral's
    # is narrower (300-1080 K per PNNL-35702 section 3.1.4.3), so it needs its own value.
    _assert_backend_contract(m.Zircalloy.thrm_expan_axial, T_FLOAT, label="thrm_expan_axial")
    _assert_backend_contract(m.Zircalloy.thrm_expan_diametral, 1000.0, label="thrm_expan_diametral")


def test_zircalloy_constants():
    assert m.Zircalloy.T_melt() == 2123.15
    assert m.Zircalloy.rho() == 6520.0


def test_zircalloy_eps_backend_contract_with_default_oxide():
    _assert_backend_contract(m.Zircalloy.eps, T_FLOAT, label="Zircalloy.eps default t_ox")


def test_zircalloy_E_and_G_backend_contract_with_default_fluence():
    _assert_backend_contract(m.Zircalloy.E, T_FLOAT, label="Zircalloy.E default phi")
    _assert_backend_contract(m.Zircalloy.G, T_FLOAT, label="Zircalloy.G default phi")


def test_zircalloy_meyer_hardness_backend_contract():
    # 350-875 K (PNNL-35702 section 3.1.8.3) is narrower than T_FLOAT.
    _assert_backend_contract(m.Zircalloy.meyer_hardness, 600.0, label="meyer_hardness")


def test_zircalloy_axial_growth_table_lookup():
    # 1e25 n/cm^2 is far above the 0-8.5e21 validated fluence cap for Zircaloy-4
    # (PNNL-35702 section 3.1.9.3) by design -- this checks the formula stays positive
    # under extrapolation, not that the extrapolation is validated.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
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
    # T_ZR_CREEP (600 K) is inside the 570-625 K creep-model range (PNNL-35702 section
    # 3.1.10.3); T_FLOAT is not.
    _assert_backend_contract(m.Zircalloy.strain_rate_thermal, T_ZR_CREEP, 50.0, 1.0e20,
                              label="strain_rate_thermal")


def test_zircalloy_strain_rate_irrad_backend_contract():
    _assert_backend_contract(m.Zircalloy.strain_rate_irrad, T_ZR_CREEP, 50.0, 1.0e14,
                              label="strain_rate_irrad")


def test_zircalloy_creep_strain_and_rate_run():
    val = m.Zircalloy.creep_strain('Zircaloy-4', T_ZR_CREEP, 50.0, 1.0e14, 1.0e20, 100.0)
    rate = m.Zircalloy.creep_rate('Zircaloy-4', T_ZR_CREEP, 50.0, 1.0e14, 1.0e20, 100.0)
    assert np.isfinite(val) and np.isfinite(rate)
    # ZIRLO applies an extra 0.8 factor relative to Zircaloy-4/2/M5 -- a structural
    # property of the formula's own branch, not a value read off a run.
    val_zirlo = m.Zircalloy.creep_strain('ZIRLO', T_ZR_CREEP, 50.0, 1.0e14, 1.0e20, 100.0)
    assert val_zirlo == pytest.approx(0.8*val)


# -------------------------------------------------------------------------------- HT9
def test_ht9_k_and_cp_backend_contract():
    _assert_backend_contract(m.HT9.k, T_HT9, label="HT9.k")
    _assert_backend_contract(m.HT9.cp, T_HT9, label="HT9.cp")


def test_ht9_constants():
    assert m.HT9.T_melt() == 973.0
    assert m.HT9.rho() == 7750.0
    assert m.HT9.eps() == 0.9


def test_ht9_thrm_expan_and_moduli_backend_contract():
    # HT9.E/HT9.G take Celsius (25-600 C validated range, PNNL-35702 section 3.3.7.1's
    # own "Where," clause), unlike every other T argument in this module.
    _assert_backend_contract(m.HT9.thrm_expan, T_HT9, label="HT9.thrm_expan")
    _assert_backend_contract(m.HT9.E, T_HT9_CELSIUS, label="HT9.E")
    _assert_backend_contract(m.HT9.G, T_HT9_CELSIUS, label="HT9.G")


def test_ht9_meyer_hardness_matches_zircalloy():
    # HT9.meyer_hardness is documented as reusing the zirconium model outright.
    # 600 K is inside Zircalloy.meyer_hardness's 350-875 K range.
    assert m.HT9.meyer_hardness(T_HT9) == m.Zircalloy.meyer_hardness(T_HT9)


def test_ht9_strain_rate_primary_backend_contract_with_scalar_time():
    # The pre-cleanup failure mode this test targets: t passed as a plain float
    # alongside a tensor T. 298.15-873.15 K is the validated range (PNNL-35702 section
    # 3.3.10.2), narrower than T_FLOAT.
    _assert_backend_contract(m.HT9.strain_rate_primary, T_HT9, 50.0, 1000.0,
                              label="HT9.strain_rate_primary")


def test_ht9_strain_rate_components_sum_to_the_totals():
    T0, sig0, flux0, t0 = T_HT9, 50.0, 1.0e14, 1000.0
    thermal = m.HT9.strain_rate_thermal(T0, sig0, t0)
    irrad = m.HT9.strain_rate_irrad(T0, sig0, flux0)
    total = m.HT9.strain_rate(T0, sig0, flux0, t0)
    assert total == pytest.approx(thermal + irrad)


def test_ht9_yield_stress_backend_contract():
    _assert_backend_contract(m.HT9.yield_stress, T_HT9, label="HT9.yield_stress")


# ------------------------------------------------------------------------------- D9_SS
def test_d9_ss_constant_properties():
    """D9 is the cladding of the SCWR lattice the supercritical work is built around, and
    `k` was an unimplemented stub until Phase 4. Both values are single evaluated numbers
    from Hughes et al. (2014) -- section 4 for the conductivity, Table 1 for the density
    -- not correlations, so they are asserted exactly."""
    assert m.D9_SS.k() == 18.9
    assert m.D9_SS.k(650.0) == 18.9          # T is accepted and ignored, by design
    assert m.D9_SS.rho() == 8100.0



# ------------------------------------------------------------------- RANGES (Phase 3)
def test_uo2_eps_warns_above_range():
    # PNNL-35702 section 2.1.5.3: 300 to 2500 K.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.UO2.eps(3000.0)
    assert len(caught) == 1
    assert issubclass(caught[0].category, RangeWarning)


def test_uo2_k_nfi_warns_above_burnup_range():
    # PNNL-35702 section 2.1.1.3: rod-average burnup 0 to 90 GWd/MTU for UO2.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.UO2.k_NFI(1200.0, Bu=150.0)
    assert len(caught) == 1
    assert issubclass(caught[0].category, RangeWarning)


def test_zircalloy_k_warns_above_range():
    # PNNL-35702 section 3.1.1.3: 285 to 1770 K.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.Zircalloy.k(2000.0)
    assert len(caught) == 1
    assert issubclass(caught[0].category, RangeWarning)


def test_zircalloy_axial_growth_warns_above_the_alloy_specific_fluence_cap():
    # PNNL-35702 section 3.1.9.3: 0 to 8.5e21 n/cm^2 for Zircaloy-4 (narrower cap),
    # 0 to 1e22 n/cm^2 for Zircaloy-2 (wider cap) -- both exceeded here to check each
    # alloy is checked against its own RANGES entry, not the other one's.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.Zircalloy.axial_growth('Zircaloy-4', 9.0E21)
    assert len(caught) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.Zircalloy.axial_growth('Zircaloy-2', 9.0E21)
    assert caught == []


def test_ht9_k_warns_above_range():
    # PNNL-35702 section 3.3.1.3: 293 to 873 K.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.HT9.k(1000.0)
    assert len(caught) == 1
    assert issubclass(caught[0].category, RangeWarning)


def test_gas_k_warns_above_range_for_helium_but_not_for_air():
    # PNNL-35702 section 4.1.3: helium 273 to 2500 K. Air has no range in that section
    # (see Gas.k's docstring) -- RANGES has no "gas_k_Air" entry, so this must not warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.Gas.k('He', 3000.0)
    assert len(caught) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.Gas.k('Air', 3000.0)
    assert caught == []
