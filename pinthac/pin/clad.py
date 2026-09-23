"""Radial conduction across the cladding."""
from pinthac import backend


def T_ci(Rco, Rci, kc, Tco, qp):
    """Temperature of the inner cladding surface, from a known outer-surface temperature.

    Formulation:
        Tci = Tco + qp * ln(Rco/Rci) / (2*pi*kc)

        Handles the annulus case, where the *inner* channel's clad has Rco < Rci by
        naming convention (see docs/OPEN_QUESTIONS.md Q11 on the two geometry-naming
        conventions in this repository): the larger of the two radii goes on top of the
        log ratio either way, by flipping the sign rather than the ratio, so the result
        is identical whichever of Rco/Rci happens to be the larger one.

    Valid range:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31). Standard cylindrical
        conduction; no fitted database to be out of range of.

    Uncertainty:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31).

    Reference:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31). Standard steady-state
        cylindrical (Fourier) conduction.

    Inputs (Rco, Rci, Tco, qp broadcastable float / numpy array / torch tensor; kc
            float, numpy array, or torch tensor, broadcastable against the others):
        Rco : outer cladding radius, m
        Rci : inner cladding radius, m
        kc  : cladding thermal conductivity, W/m-K
        Tco : outer cladding temperature, K
        qp  : linear heat generation rate carried across the cladding, W/m
    Returns:
        val : inner cladding temperature, K, same type as Tco
    """
    Rco = backend.promote(Rco, Tco)
    Rci = backend.promote(Rci, Tco)
    xp = backend.lib(Rco, Rci, kc, Tco, qp)

    log_term = xp.log(Rco/Rci)
    # Was a Python `if (Rco < Rci)` -- fine for a scalar geometry, but a batched
    # Rco/Rci (unusual, but the contract must still hold) needs backend.where instead,
    # the same fix used everywhere else in this library for a value-dependent branch.
    log_term = backend.where(Rco < Rci, -log_term, log_term)
    # (Sign flip only, not a print: an annulus geometry legitimately has Rco < Rci --
    # see the Formulation note above -- so this is routine, not a warning-worthy input.)

    val = Tco + qp * log_term/(2*xp.pi*kc)
    return val
