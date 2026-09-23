"""Rod-bundle correction factors for round-tube heat transfer correlations."""
from pinthac import backend


RANGES = {}


class Bundle:
    def Weissman(P, D):
        """Weissman rod-bundle correction factor.

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
        """Presser rod-bundle correction factor.

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
            Des. 270 (2014) 412-420, Eq. (10) -- see the model references.

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
