"""
Radial temperature profile of a solid cylindrical fuel pellet.

Moved verbatim from PinHT.py in Phase 1. Phase 2 brings it up to the docstring standard
without changing the formula; it already passed the float/numpy/torch contract at
Phase 0 (docs/AUDIT.md's baseline). Only the constant-conductivity profile exists so
far; the conductivity-integral form described in
docs/reference/PINTHA_Code_Summary.pdf section 4.3, and the heat-flux and
LHGR-at-radius helpers, are for a later phase to add (pin/annular.py already has the
conductivity-integral solve for the annular case, Ann_HT/Ann_Theta -- Cyl_HT is that
solve's simpler, constant-kf special case).
"""


def Cyl_HT(r, q_vol, kint, C):
    """
    Radial temperature profile for a solid cylinder with uniform volumetric heat
    generation and constant thermal conductivity.

    Why this model is here:
        The simplest fuel-pellet radial temperature solve: a constant-property special
        case of the conductivity-integral (Kirchhoff-transformed) annular solve in
        pin/annular.py, useful wherever kf's own temperature dependence can be ignored
        or has already been folded into an effective kint.

    Formulation:
        T(r) = C - q_vol*r^2/(4*kint)

        Sign convention confirmed against docs/reference/Annular_Heat_Transfer_Final.pdf
        Eq. (2), which carries the same -q'''*r^2/4 term for the general (annular) case
        this is the r_i=0 special case of -- see docs/OPEN_QUESTIONS.md Q17 item 4,
        where a different source document had the sign the other way and the PDF (and
        this code) were confirmed correct.

    Valid range:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31). Standard cylindrical
        conduction; no fitted database to be out of range of.

    Uncertainty:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31).

    Reference:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31). Standard steady-state
        cylindrical (Fourier) conduction with uniform volumetric generation.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        r     : radial position, m
        q_vol : volumetric heat generation rate, W/m^3
        kint  : thermal conductivity, W/m-K
        C     : centerline (reference) temperature, K
    Returns:
        val : temperature at r, K, same type as r
    """
    val = C - q_vol * r**2 / (4*kint)
    return val
