"""
Rod-bundle correction factors for round-tube heat transfer correlations.

Moved verbatim from PinHT.Bundle in Phase 1. Phase 2 brings it up to the docstring and
backend standard without changing either formula. Applied as
htc_pin = psi * htc_round_tube (Hughes, Pelaez, Schubring & Jordan, Nucl. Eng. Des. 270
(2014) 412-420, Eq. 11) -- a round-tube correlation (e.g. Dittus-Boelter, Swenson)
systematically under- or over-predicts a rod bundle's actual htc, and psi corrects for
that geometry difference. See docs/PHYSICS_REVIEW.md and
docs/reference/PINTHA_Code_Summary.pdf section 4.5.
"""
from pinthac import backend


RANGES = {}
# Neither correlation has a published validated P/D range in the source, in
# docs/reference/, or in docs/PHYSICS_REVIEW.md -- Hughes (2014) reports psi at two
# specific P/D values (1.28 -> 0.98, 1.15 -> 0.94) as worked examples, not as the bounds
# of the fit. RANGES stays empty rather than inventing bounds from those two points; see
# docs/OPEN_QUESTIONS.md Q31.


class Bundle:
    def Weissman(P, D):
        """
        Weissman rod-bundle correction factor.

        Why this model is here:
            One of two interchangeable psi factors (with Presser, below) that convert a
            round-tube heat transfer coefficient to a rod-bundle one via
            htc_pin = psi * htc_round_tube.

        Formulation:
            R = P/D
            psi = c1*R + c2,   c1 = 1.826, c2 = -1.0430

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Weissman".

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            P : rod pitch, m
            D : rod outer diameter, m
        Returns:
            val : bundle correction factor psi, dimensionless, same type as P
        """
        R = P/D
        c1 = 1.826
        c2 = -1.0430
        val = c1*R + c2
        return val

    def Presser(P, D):
        """
        Presser rod-bundle correction factor.

        Why this model is here:
            The other of the two psi factors, used through sca/annular.py's outer
            channel and sca/lut.py; per docs/PHYSICS_REVIEW.md this is Hughes et al.
            (2014) Eq. (10), giving psi = 0.98 at P/D = 1.28 and 0.94 at P/D = 1.15 -- a
            few percent correction, systematically in one direction.

        Formulation:
            R = P/D
            psi = c1 + c2*R - c3*exp(-7*(R-1)),   c1 = 0.9217, c2 = 0.1478, c3 = 0.1130

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Exercised by Hughes
            (2014) at P/D = 1.15 and 1.28.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Presser (1967), as cited by Hughes, Pelaez, Schubring & Jordan, Nucl. Eng.
            Des. 270 (2014) 412-420, Eq. (10) -- see docs/PHYSICS_REVIEW.md.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            P : rod pitch, m
            D : rod outer diameter, m
        Returns:
            val : bundle correction factor psi, dimensionless, same type as P
        """
        R = P/D
        c1 = 0.9217
        c2 = 0.1478
        c3 = 0.1130
        xp = backend.lib(P, D)
        exp_term = -7*(R-1)
        val = c1+c2*R-c3*xp.exp(exp_term)
        return val
