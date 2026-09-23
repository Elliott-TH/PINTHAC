"""IAPWS verification-table tests.

Why this file matters more than the rest of the suite: every other test in this repository
checks a structural property -- does it run on a tensor, does the gradient stay finite, does
a piecewise branch switch where it should. Those catch regressions but they cannot tell you
the physics is *right*, because the expected values ultimately come from the same code.

The IAPWS releases publish check values precisely so an implementation can be proved
correct against something external. Every number below is transcribed from a published
release, not produced by this library. If one of these fails, the library is wrong.

Sources, all in :
    IAPWS95-2018.pdf     R6-95(2018), Tables 7 and 8
    IAWPS_Viscosity.pdf  R12-08, Tables 4 and 5
    ThCond.pdf           R15-11, Tables 4 and 5
    IAPWS_97.pdf         R7-97(2012), Tables 5, 7, 9, 15, 24, 29, 33, 42 and Sec. 8
"""
import numpy as np
import pytest

from pinthac.properties import iapws97 as if97
from pinthac.properties.iapws95 import IAPWS95


def state(rho, T):
    """The Helmholtz state dict every property accessor consumes."""
    return IAPWS95.helmholtz(np.array([rho]), np.array([T]))


def scalar(x):
    """First element as a plain float. The accessors return one-element arrays for the
    one-element inputs used here, and numpy 2 refuses float() on those.
    """
    return float(np.ravel(np.asarray(x))[0])


# --------------------------------------------------------------------------------------
# - IAPWS-95 Table 7 -- single-phase region.
TABLE_7 = [
    (300.0, 0.9965560e3,  0.992418352e-1, 0.413018112e1, 0.150151914e4, 0.393062643),
    (300.0, 0.1005308e4,  0.200022515e2,  0.406798347e1, 0.153492501e4, 0.387405401),
    (300.0, 0.1188202e4,  0.700004704e3,  0.346135580e1, 0.244357992e4, 0.132609616),
    (500.0, 0.4350000,    0.999679423e-1, 0.150817541e1, 0.548314253e3, 0.794488271e1),
    (500.0, 0.4532000e1,  0.999938125,    0.166991025e1, 0.535739001e3, 0.682502725e1),
    (500.0, 0.8380250e3,  0.100003858e2,  0.322106219e1, 0.127128441e4, 0.256690919e1),
    (500.0, 0.1084564e4,  0.700000405e3,  0.307437693e1, 0.241200877e4, 0.203237509e1),
    (647.0, 0.3580000e3,  0.220384756e2,  0.618315728e1, 0.252145078e3, 0.432092307e1),
    (900.0, 0.2410000,    0.100062559,    0.175890657e1, 0.724027147e3, 0.916653194e1),
    (900.0, 0.5261500e2,  0.200000690e2,  0.193510526e1, 0.698445674e3, 0.659070225e1),
    (900.0, 0.8707690e3,  0.700000006e3,  0.266422350e1, 0.201933608e4, 0.417223802e1),
]


@pytest.mark.parametrize("T,rho,p_ref,cv_ref,w_ref,s_ref", TABLE_7)
def test_iapws95_table_7_single_phase(T, rho, p_ref, cv_ref, w_ref, s_ref):
    d = state(rho, T)

    # Every point in this table now reproduces the release's own nine figures. The one
    # documented relaxation is footnote a to Table 7: in the liquid region at low pressure,
    # small density changes along an isotherm swing the pressure hard, so accumulated
    # rounding can stop any given machine reproducing p to nine figures there. Measured on
    # this one: 1.5e-10, comfortably inside the claim, but the tolerance is left loose
    # because the release says the result is machine-dependent.
    #
    # The T = 647 K, rho = 358 point used to need a relaxation of 1e-5 on p and 1e-3 on w
    # and no longer does -- it is now exact to 1.3e-9 and 1.1e-9. It sits 0.6 K below T_c,
    # which is the only place in this table where the two non-analytic terms of Eq. (6)
    # contribute anything at all, and two of their published delta-derivatives were
    # mistranscribed in the module. The degradation was a formulation error after all, not
    # the conditioning it was taken for; see the comment in iapws95.Phir group 4.
    p_tol, w_tol = 1e-8, 1e-8
    if T == 300.0 and rho < 1.0e3:
        p_tol = 1e-7

    assert scalar(IAPWS95.p(d, units='MPa')) == pytest.approx(p_ref, rel=p_tol)
    assert scalar(IAPWS95.cv(d, units='kJ')) == pytest.approx(cv_ref, rel=1e-8)
    assert scalar(IAPWS95.c(d)) == pytest.approx(w_ref, rel=w_tol)
    assert scalar(IAPWS95.s(d, units='kJ')) == pytest.approx(s_ref, rel=1e-8)


# ---------------------------------------------------------------------------------------
# IAPWS-95 Table 8 -- two-phase region, from the Maxwell criterion.
# T [K], p_sat [MPa], rho_f, rho_g [kg/m^3], h_f, h_g [kJ/kg], s_f, s_g [kJ/kg/K].
# ---------------------------------------------------------------------------------------
TABLE_8 = [
    (275.0, 0.698451167e-3, 0.999887406e3, 0.550664919e-2,
     0.775972202e1, 0.250428995e4, 0.283094670e-1, 0.910660121e1),
    (450.0, 0.932203564,    0.890341250e3, 0.481200360e1,
     0.749161585e3, 0.277441078e4, 0.210865845e1, 0.660921221e1),
    (625.0, 0.169082693e2,  0.567090385e3, 0.118290280e3,
     0.168626976e4, 0.255071625e4, 0.380194683e1, 0.518506121e1),
]


@pytest.mark.parametrize("T,p_ref,rhof_ref,rhog_ref,hf_ref,hg_ref,sf_ref,sg_ref", TABLE_8)
def test_iapws95_table_8_saturation(T, p_ref, rhof_ref, rhog_ref, hf_ref, hg_ref,
                                     sf_ref, sg_ref):
    sat = IAPWS95.saturation(np.array([T]))
    assert scalar(sat['p']) == pytest.approx(p_ref, rel=1e-7)
    assert scalar(sat['rho_f']) == pytest.approx(rhof_ref, rel=1e-8)
    assert scalar(sat['rho_g']) == pytest.approx(rhog_ref, rel=1e-8)

    d_f = state(scalar(sat['rho_f']), T)
    d_g = state(scalar(sat['rho_g']), T)
    assert scalar(IAPWS95.h(d_f, units='kJ')) == pytest.approx(hf_ref, rel=1e-8)
    assert scalar(IAPWS95.h(d_g, units='kJ')) == pytest.approx(hg_ref, rel=1e-8)
    assert scalar(IAPWS95.s(d_f, units='kJ')) == pytest.approx(sf_ref, rel=1e-8)
    assert scalar(IAPWS95.s(d_g, units='kJ')) == pytest.approx(sg_ref, rel=1e-8)

    # T_sat is the inverse of all of this, and is solved a different way -- an ancillary
    # bracket, then Newton with the Clausius-Clapeyron slope -- so round-tripping the
    # published p_sat back to T checks the inverse against the release too.
    assert scalar(IAPWS95.T_sat(p_ref)) == pytest.approx(T, rel=1e-8)


# ---------------------------------------------------------------------------------------
# IAPWS R12-08 viscosity, Table 5 -- the near-critical points, which exercise the critical
# enhancement mu_2. Table 4's points are quoted with mu_2 = 1 (enhancement suppressed);
# they are all far enough from the critical point that the enhancement is negligible there
# anyway, so both tables are checked against the same full correlation.
# T [K], rho [kg/m^3], mu [micro-Pa.s].
# ---------------------------------------------------------------------------------------
VISCOSITY_TABLE_4 = [
    (298.15, 998.0,  889.735100), (298.15, 1200.0, 1437.649467),
    (373.15, 1000.0, 307.883622), (433.15, 1.0,     14.538324),
    (433.15, 1000.0, 217.685358), (873.15, 1.0,     32.619287),
    (873.15, 100.0,   35.802262), (873.15, 600.0,   77.430195),
    (1173.15, 1.0,    44.217245), (1173.15, 100.0,  47.640433),
    (1173.15, 400.0,  64.154608),
]

VISCOSITY_TABLE_5 = [
    (647.35, 122.0, 25.520677), (647.35, 222.0, 31.337589),
    (647.35, 272.0, 36.228143), (647.35, 322.0, 42.961579),
    (647.35, 372.0, 45.688204), (647.35, 422.0, 49.436256),
]


@pytest.mark.parametrize("T,rho,mu_ref", VISCOSITY_TABLE_4)
def test_iapws_viscosity_table_4(T, rho, mu_ref):
    mu = scalar(IAPWS95.mu(state(rho, T))) * 1.0e6      # Pa.s -> micro-Pa.s
    assert mu == pytest.approx(mu_ref, rel=1e-6)


@pytest.mark.parametrize("T,rho,mu_ref", VISCOSITY_TABLE_5)
def test_iapws_viscosity_table_5_near_critical(T, rho, mu_ref):
    """The critical enhancement mu_2, R12-08 Eqs. (14)-(21).

    This table is the sharpest test in the file of the non-analytic terms of IAPWS-95.
    mu_2 is driven by delta-chi, a difference of two isothermal compressibilities
    (Eq. 21), and (drho/dp)_T is the second delta-derivative of the Helmholtz energy --
    so an error in those terms shows up here magnified rather than damped. It used to
    need a tolerance of 2e-4; with the group 4 derivatives corrected the whole table
    reproduces to 1e-8, and the tolerance below is set at 1e-6 only to leave room for
    a different machine's rounding.
    """
    mu = scalar(IAPWS95.mu(state(rho, T))) * 1.0e6
    assert mu == pytest.approx(mu_ref, rel=1.0e-6)


# ---------------------------------------------------------------------------------------
# IAPWS R15-11 thermal conductivity, Table 5 -- includes the critical enhancement lambda_2,
# which is the hard part of the formulation and the part that needs (drho/dp)_T, cp and cv
# from the equation of state. T [K], rho [kg/m^3], lambda [mW/m/K].
#
# Table 4's zero-density points are deliberately not tested: the release notes that several
# IAPWS-95 derivatives diverge at rho = 0, so those entries require lambda_2 to be forced to
# zero by hand. That is a property of the check table, not of the correlation.
# ---------------------------------------------------------------------------------------
CONDUCTIVITY_TABLE_4 = [
    (298.15, 998.0,  607.712868),
    (298.15, 1200.0, 799.038144),
]

# The two end points of Table 5 -- the dilute limit and the dense liquid -- bracket the
# critical enhancement, so checking them separately says whether a failure in the middle of
# the isotherm belongs to lambda_0 and lambda_1 or to lambda_2.
CONDUCTIVITY_TABLE_5_EXACT = [
    (647.35, 1.0,   51.9298924),
    (647.35, 750.0, 600.961346),
]

CONDUCTIVITY_TABLE_5_ENHANCED = [
    (647.35, 122.0, 130.922885), (647.35, 222.0,  367.787459),
    (647.35, 272.0, 757.959776), (647.35, 322.0, 1443.75556),
    (647.35, 372.0, 650.319402), (647.35, 422.0,  448.883487),
]


@pytest.mark.parametrize("T,rho,lam_ref", CONDUCTIVITY_TABLE_4)
def test_iapws_conductivity_table_4(T, rho, lam_ref):
    """These two 298.15 K liquid points are the only check values in any of the releases
    that exercise the i >= 1 rows of the lambda_1 coefficient table.

    On the 647.35 K isotherm, where every other conductivity check value sits, the factor
    (1/T_bar - 1) is about -3.9e-4, so its second power is ~1.5e-7 and the i = 2 row is
    invisible. At 298.15 K that factor is 1.17 and every row contributes. A transposed
    digit in L_22 -- 3.55772244 shipped against 3.55777244 published -- was therefore
    undetectable anywhere except here, and cost -0.094 % and -0.190 % at these two points.
    Fixed in the coefficient file; these now reproduce exactly.
    """
    lam = scalar(IAPWS95.lam(state(rho, T))) * 1.0e3     # W/m/K -> mW/m/K
    assert lam == pytest.approx(lam_ref, rel=1e-6)


@pytest.mark.parametrize("T,rho,lam_ref", CONDUCTIVITY_TABLE_5_EXACT)
def test_iapws_conductivity_table_5_end_points(T, rho, lam_ref):
    """The dilute-gas and dense-liquid ends of the 647.35 K isotherm, where the critical
    enhancement is small, so these check lambda_0 and lambda_1 nearly on their own.

    Measured: 3.5e-9 at rho = 1, where the enhancement is 0.00013 of 51.93 mW/m/K, and
    2.3e-6 at rho = 750, where it is 3.34 of 600.96. The tolerance is set above the
    larger of the two.
    """
    lam = scalar(IAPWS95.lam(state(rho, T))) * 1.0e3
    assert lam == pytest.approx(lam_ref, rel=1e-5)


@pytest.mark.parametrize("T,rho,lam_ref", CONDUCTIVITY_TABLE_5_ENHANCED)
def test_iapws_conductivity_table_5_critical_enhancement(T, rho, lam_ref):
    """The critical enhancement lambda_2, R15-11 Eqs. (18)-(24).

    The strictest end-to-end test in the file: lambda_2 needs (drho/dp)_T, cp, cv and a
    viscosity that carries its own critical enhancement, so it exercises three separate
    IAPWS releases and the second derivatives of the Helmholtz energy at once. It was
    held to 3e-3 while the group 4 delta-derivatives were wrong and reproduces to 4.6e-6
    now; 1e-5 leaves room for a different machine's rounding.

    Before the viscosity enhancement existed at all these points were off by up to +2.7 %
    and returned NaN at rho_c, because Eq. (18) divides by a mu that had no enhancement.
    """
    lam = scalar(IAPWS95.lam(state(rho, T))) * 1.0e3
    assert lam == pytest.approx(lam_ref, rel=1.0e-5)


# =======================================================================================
# IAPWS-IF97, R7-97(2012).
#
# Every region of IF97 has a published check table, and between them they pin down the
# basic equations, the backward equations, the two auxiliary boundary equations and the
# saturation line. They are worth having in full because IF97 is the formulation whose
# coefficient tables are longest and most error-prone to transcribe -- eight of the nine
# backward-equation tables in this repository turned out to have digits transposed in
# them, and every one of those errors was invisible to a structural test and obvious to
# these.
# =======================================================================================

def one(x):
    """IF97's accessors take a state dict; scalars in, scalars out."""
    return float(np.ravel(np.asarray(x))[0])


# ---------------------------------------------------------------------------------------
# Region 1 basic equation, Table 5. T [K], p [MPa], then v, h, u, s, cp, w.
# ---------------------------------------------------------------------------------------
IF97_TABLE_5 = [
    (300.0, 3.0,  0.100215168e-2, 0.115331273e3, 0.112324818e3,
     0.392294792,    0.417301218e1, 0.150773921e4),
    (300.0, 80.0, 0.971180894e-3, 0.184142828e3, 0.106448356e3,
     0.368563852,    0.401008987e1, 0.163469054e4),
    (500.0, 3.0,  0.120241800e-2, 0.975542239e3, 0.971934985e3,
     0.258041912e1,  0.465580682e1, 0.124071337e4),
]


@pytest.mark.parametrize("T,p,v_ref,h_ref,u_ref,s_ref,cp_ref,w_ref", IF97_TABLE_5)
def test_if97_region1_basic(T, p, v_ref, h_ref, u_ref, s_ref, cp_ref, w_ref):
    d = if97.R1.gibbs(p, T)
    assert one(if97.R1.v(d)) == pytest.approx(v_ref, rel=1e-8)
    assert one(if97.R1.h(d, units='kJ')) == pytest.approx(h_ref, rel=1e-8)
    assert one(if97.R1.u(d, units='kJ')) == pytest.approx(u_ref, rel=1e-8)
    assert one(if97.R1.s(d, units='kJ')) == pytest.approx(s_ref, rel=1e-8)
    assert one(if97.R1.cp(d, units='kJ')) == pytest.approx(cp_ref, rel=1e-8)
    assert one(if97.R1.c(d)) == pytest.approx(w_ref, rel=1e-8)


# Tables 7 and 9: region 1 backward equations. p [MPa], h or s, T [K].
IF97_TABLE_7 = [(3.0, 500.0, 0.391798509e3), (80.0, 500.0, 0.378108626e3),
                (80.0, 1500.0, 0.611041229e3)]
IF97_TABLE_9 = [(3.0, 0.5, 0.307842258e3), (80.0, 0.5, 0.309979785e3),
                (80.0, 3.0, 0.565899909e3)]


@pytest.mark.parametrize("p,h,T_ref", IF97_TABLE_7)
def test_if97_region1_backward_ph(p, h, T_ref):
    assert one(if97.R1.Tph(p, h)) == pytest.approx(T_ref, rel=1e-8)


@pytest.mark.parametrize("p,s,T_ref", IF97_TABLE_9)
def test_if97_region1_backward_ps(p, s, T_ref):
    assert one(if97.R1.Tps(p, s)) == pytest.approx(T_ref, rel=1e-8)


# ---------------------------------------------------------------------------------------
# Region 2 basic equation, Table 15.
# ---------------------------------------------------------------------------------------
IF97_TABLE_15 = [
    (300.0, 0.0035, 0.394913866e2, 0.254991145e4, 0.241169160e4,
     0.852238967e1, 0.191300162e1, 0.427920172e3),
    (700.0, 0.0035, 0.923015898e2, 0.333568375e4, 0.301262819e4,
     0.101749996e2, 0.208141274e1, 0.644289068e3),
    (700.0, 30.0,   0.542946619e-2, 0.263149474e4, 0.246861076e4,
     0.517540298e1, 0.103505092e2, 0.480386523e3),
]


@pytest.mark.parametrize("T,p,v_ref,h_ref,u_ref,s_ref,cp_ref,w_ref", IF97_TABLE_15)
def test_if97_region2_basic(T, p, v_ref, h_ref, u_ref, s_ref, cp_ref, w_ref):
    """cp and cv are the reason this test matters beyond transcription."""
    d = if97.R2.gibbs(p, T)
    assert one(if97.R2.v(d)) == pytest.approx(v_ref, rel=1e-8)
    assert one(if97.R2.h(d, units='kJ')) == pytest.approx(h_ref, rel=1e-8)
    assert one(if97.R2.u(d, units='kJ')) == pytest.approx(u_ref, rel=1e-8)
    assert one(if97.R2.s(d, units='kJ')) == pytest.approx(s_ref, rel=1e-8)
    assert one(if97.R2.cp(d, units='kJ')) == pytest.approx(cp_ref, rel=1e-8)
    assert one(if97.R2.c(d)) == pytest.approx(w_ref, rel=1e-8)
    # cv has no column in Table 15, but it must stay below cp for a single-phase fluid.
    assert one(if97.R2.cv(d, units='kJ')) < one(if97.R2.cp(d, units='kJ'))


# Tables 24 and 29: region 2 backward equations, three points per subregion. These also
# check the subregion selection, since the three 2a points, three 2b points and three 2c
# points are only reached if the p <= 4 MPa split, the B2bc boundary and the s = 5.85
# kJ/kg-K split all resolve the way the release says.
IF97_TABLE_24 = [
    (0.001, 3000.0, 0.534433241e3), (3.0, 3000.0, 0.575373370e3),
    (3.0, 4000.0, 0.101077577e4),
    (5.0, 3500.0, 0.801299102e3), (5.0, 4000.0, 0.101531583e4),
    (25.0, 3500.0, 0.875279054e3),
    (40.0, 2700.0, 0.743056411e3), (60.0, 2700.0, 0.791137067e3),
    (60.0, 3200.0, 0.882756860e3),
]
IF97_TABLE_29 = [
    (0.1, 7.5, 0.399517097e3), (0.1, 8.0, 0.514127081e3), (2.5, 8.0, 0.103984917e4),
    (8.0, 6.0, 0.600484040e3), (8.0, 7.5, 0.106495556e4), (90.0, 6.0, 0.103801126e4),
    (20.0, 5.75, 0.697992849e3), (80.0, 5.25, 0.854011484e3),
]


@pytest.mark.parametrize("p,h,T_ref", IF97_TABLE_24)
def test_if97_region2_backward_ph(p, h, T_ref):
    assert one(if97.R2.Tph(p, h)) == pytest.approx(T_ref, rel=1e-8)


@pytest.mark.parametrize("p,s,T_ref", IF97_TABLE_29)
def test_if97_region2_backward_ps(p, s, T_ref):
    assert one(if97.R2.Tps(p, s)) == pytest.approx(T_ref, rel=1e-8)


def test_if97_b2bc_boundary():
    """Sec. 6.3.1's verification point for the B2bc-equation, p = 100 MPa -> h = 3516.0043230."""
    assert one(if97.R2.h_B2bc(100.0)) == pytest.approx(3516.0043230, rel=1e-9)


def test_if97_b23_boundary():
    """Sec. 4's verification point: the B23 line passes through T = 623.15 K, p = 16.5291643
    MPa, and Eqs. (5) and (6) must both reproduce it.
    """
    assert one(if97.B23.p(623.15)) == pytest.approx(16.5291643, rel=1e-8)
    assert one(if97.B23.T(16.5291643)) == pytest.approx(623.15, rel=1e-8)


# --------------------------------------------------------------------------------------
# - Region 3 basic equation, Table 33.
IF97_TABLE_33 = [
    (650.0, 500.0, 0.255837018e2, 0.186343019e4, 0.181226279e4,
     0.405427273e1, 0.138935717e2, 0.502005554e3),
    (650.0, 200.0, 0.222930643e2, 0.237512401e4, 0.226365868e4,
     0.485438792e1, 0.446579342e2, 0.383444594e3),
    (750.0, 500.0, 0.783095639e2, 0.225868845e4, 0.210206932e4,
     0.446971906e1, 0.634165359e1, 0.760696041e3),
]


@pytest.mark.parametrize("T,rho,p_ref,h_ref,u_ref,s_ref,cp_ref,w_ref", IF97_TABLE_33)
def test_if97_region3_basic(T, rho, p_ref, h_ref, u_ref, s_ref, cp_ref, w_ref):
    d = if97.R3.helmholtz(rho, T)
    assert one(if97.R3.p(d, units='MPa')) == pytest.approx(p_ref, rel=1e-8)
    assert one(if97.R3.h(d, units='kJ')) == pytest.approx(h_ref, rel=1e-8)
    assert one(if97.R3.u(d, units='kJ')) == pytest.approx(u_ref, rel=1e-8)
    assert one(if97.R3.s(d, units='kJ')) == pytest.approx(s_ref, rel=1e-8)
    assert one(if97.R3.cp(d, units='kJ')) == pytest.approx(cp_ref, rel=1e-8)
    assert one(if97.R3.c(d)) == pytest.approx(w_ref, rel=1e-8)


@pytest.mark.parametrize("T,rho,p_ref,h_ref,u_ref,s_ref,cp_ref,w_ref", IF97_TABLE_33)
def test_if97_region3_density_solve(T, rho, p_ref, h_ref, u_ref, s_ref, cp_ref, w_ref):
    """rho_pT inverts Eq. (28) for the density the table started from.

    Held to 1e-6 rather than the solver's own precision, and the reason is the check
    table rather than the solver: the published pressures carry nine figures, and at
    (650 K, 200 kg/m^3) the isotherm is so flat that a 1e-9 relative perturbation in p
    moves the density by 1.6e-8. That sensitivity is what region 3 exists to describe.
    """
    assert one(if97.R3.rho_pT(p_ref, T)) == pytest.approx(rho, rel=1e-6)


# ---------------------------------------------------------------------------------------
# Region 4, the saturation line, Sec. 8's verification points.
# ---------------------------------------------------------------------------------------
IF97_SAT_P = [(300.0, 0.353658941e-2), (500.0, 0.263889776e1), (600.0, 0.123443146e2)]
IF97_SAT_T = [(0.1, 0.372755919e3), (1.0, 0.453035632e3), (10.0, 0.584149488e3)]


@pytest.mark.parametrize("T,p_ref", IF97_SAT_P)
def test_if97_region4_psat(T, p_ref):
    assert one(if97.R4.p(T)) == pytest.approx(p_ref, rel=1e-8)


@pytest.mark.parametrize("p,T_ref", IF97_SAT_T)
def test_if97_region4_tsat(p, T_ref):
    assert one(if97.R4.T(p)) == pytest.approx(T_ref, rel=1e-8)


# ---------------------------------------------------------------------------------------
# Region 5 basic equation, Table 42.
# ---------------------------------------------------------------------------------------
IF97_TABLE_42 = [
    (1500.0, 0.5,  0.138455090e1,  0.521976855e4, 0.452749310e4,
     0.965408875e1, 0.261609445e1, 0.917068690e3),
    (1500.0, 30.0, 0.230761299e-1, 0.516723514e4, 0.447495124e4,
     0.772970133e1, 0.272724317e1, 0.928548002e3),
    (2000.0, 30.0, 0.311385219e-1, 0.657122604e4, 0.563707038e4,
     0.853640523e1, 0.288569882e1, 0.106736948e4),
]


@pytest.mark.parametrize("T,p,v_ref,h_ref,u_ref,s_ref,cp_ref,w_ref", IF97_TABLE_42)
def test_if97_region5_basic(T, p, v_ref, h_ref, u_ref, s_ref, cp_ref, w_ref):
    d = if97.R5.gibbs(p, T)
    assert one(if97.R5.v(d)) == pytest.approx(v_ref, rel=1e-8)
    assert one(if97.R5.h(d, units='kJ')) == pytest.approx(h_ref, rel=1e-8)
    assert one(if97.R5.u(d, units='kJ')) == pytest.approx(u_ref, rel=1e-8)
    assert one(if97.R5.s(d, units='kJ')) == pytest.approx(s_ref, rel=1e-8)
    assert one(if97.R5.cp(d, units='kJ')) == pytest.approx(cp_ref, rel=1e-8)
    assert one(if97.R5.c(d)) == pytest.approx(w_ref, rel=1e-8)


def test_if97_region_selector():
    """region() against the corners of Fig. 1, one state known to belong to each region.

    The last two entries are the ones worth having: a state above 100 MPa and one above
    2273.15 K are outside the formulation entirely, and must be reported as such rather
    than handed to whichever equation is nearest.
    """
    p = np.array([3.0, 0.0035, 25.0, 0.5, 20.0, 40.0, 150.0, 10.0])
    T = np.array([300.0, 300.0, 650.0, 1500.0, 700.0, 700.0, 500.0, 2500.0])
    expect = np.array([1, 2, 3, 5, 2, 3, 0, 0])
    assert np.array_equal(if97.region(p, T), expect)


def test_if97_agrees_with_iapws95_away_from_boundaries():
    """IF97 is a fit of IAPWS-95, so the two must agree to within IF97's stated tolerance.

    This is the one test here that is not against a published number, and it earns its
    place by checking something the tables cannot: that the two modules use compatible
    unit conventions and state-dict layouts. A factor-of-1000 slip in either would pass
    every check-value test in its own module and fail here.

    The tolerances are set from what was measured, not from a claimed consistency figure:
    across these three region 1 states the density agrees to 3e-5 relative and the
    enthalpy to 0.20 kJ/kg, the larger deviation being at 550 K where region 1 is nearest
    its upper temperature limit.
    """
    T = np.array([320.0, 400.0, 550.0])           # region 1, comfortably subcooled
    p = np.full(3, 15.0)
    rho95 = IAPWS95.rho_Tp(T, p)
    h95 = IAPWS95.h(IAPWS95.helmholtz(rho95, T), units='kJ')

    d97 = if97.R1.gibbs(p, T)
    assert if97.R1.rho(d97) == pytest.approx(rho95, rel=1e-4)
    assert if97.R1.h(d97, units='kJ') == pytest.approx(h95, abs=0.3)
