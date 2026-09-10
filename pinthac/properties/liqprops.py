"""
Liquid-metal (sodium, lead, lead-bismuth eutectic) thermophysical properties.

Moved from Liquid_Metals.py in Phase 1. Phase 2 brought it up to the docstring and
backend-dispatch standard without changing any formula, constant or exponent, but left
every citation as "Not established" (Q31) -- Sobolev 2020 was not yet in the repository.
Per docs/DECISIONS.md ("Scope: lead is a toy" / "Scope: liquid metals"), no second
correlation per property is added here -- this module is supporting infrastructure for
the property library and the Phase 7 uncertainty figure, not an active SCA path.

Phase 3 checked every formula and every rho/sigma/cp/h/mu/k coefficient below against
Useful_pdfs/sobolev2020.pdf (V. Sobolev, "Properties of Liquid Metal Coolants: Na, Pb,
Pb-Bi", 2020) -- the brief's preferred source for these three coolants -- and every one
matches exactly (Sobolev's Tables 7, 9, 10, 11 and 13; see each function's own docstring
for the equation and table number). Two things did not reconcile and are not fixed here,
per CLAUDE.md ("change no physics"; a genuine defect is reported, not silently patched):

  - Every h() function's last term has the opposite sign from Sobolev's own Equation 14
    integral of its own cp -- see Sodium.h's docstring and docs/OPEN_QUESTIONS.md.
  - Sodium.uncert_k's stated [0, 8%] band does not match Sobolev's text, which instead
    states an up-to-15% spread for Na; and Lead.uncert_sig / LBE.uncert_sig do not match
    the single collective "(3-6)%" figure Sobolev states for surface tension across all
    three coolants -- see those functions' docstrings and docs/OPEN_QUESTIONS.md.

Each class carries its own melting/boiling points, per-property validated temperature
range (range_rho, range_cp, ...) and per-property relative uncertainty band
(uncert_rho, uncert_cp, ..., each a [low, high] pair, already divided by 100) as class
attributes, next to the correlations that use them -- these are the values D11
(docs/DUPLICATES.md) checked and, where wrong, already corrected (Sodium.uncert_k's
missing /100, Lead.range_rho's bare scalar). RANGES, below, is built from those same
attributes rather than inventing new bounds, so every public function here has a real
ranges.check() call.
"""
from pinthac import backend, ranges


R = 8.31432   # universal gas constant, J/mol-K -- used by each class's mu(T)


class Sodium:
    Tm = 371
    Tb = 1155
    M = 0.02299

    range_rho = [Tm, Tb]
    range_cp = [Tm, Tb]
    range_h = [Tm, Tb]
    range_mu = [Tm, Tb]
    range_sig = [Tm, Tb]
    range_k = [Tm, Tb]

    uncert_rho = [0.3/100, 3/100]
    uncert_sig = [3.0/100, 6/100]
    uncert_cp = [0/100, 1/100]
    uncert_h = [5/100, 7/100]
    uncert_mu = [5/100, 5/100]
    uncert_k = [0/100, 8/100]

    def rho(T):
        """
        Sodium density.

        Why this model is here:
            Feeds channel mass-flux and pressure-drop calculations for a sodium-cooled
            channel.

        Formulation:
            rho = rho0 - A0*(T - Tm)

        Valid range:
            Tm = 371 K to Tb = 1155 K (Sodium.range_rho).

        Uncertainty:
            0.3 to 3 percent (Sodium.uncert_rho), matching Sobolev (2020) section 4.1's
            stated "0.3-3% for Na".

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.1,
            Equation [4], Table 7 (rho0=927.0, A0=0.235, Tm=371.0 -- match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : density, kg/m^3, same type as T
        """
        ranges.check("sodium_rho", {"T": T}, RANGES["sodium_rho"])
        Tm = Sodium.Tm
        rho0, A0 = 927, 0.235
        return rho0 - A0*(T-Tm)

    def sigma(T):
        """
        Sodium surface tension.

        Why this model is here:
            Feeds two-phase / boiling-margin calculations for a sodium-cooled channel.

        Formulation:
            sigma = (sig0 - A0*(T - Tm)) * 1e-3

        Valid range:
            Tm = 371 K to Tb = 1155 K (Sodium.range_sig).

        Uncertainty:
            3 to 6 percent (Sodium.uncert_sig). Sobolev (2020) section 4.3 states a
            variation of "(3-6)%" between sources for surface tension, but as one
            collective figure covering Na, Pb and Pb-Bi(e) together, not broken out per
            metal -- so this is consistent with, but not independently confirmed
            specifically for, sodium. See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.3,
            Equation [11], Table 9 (sigma0=195e-3, A0=0.0966e-3, Tm=371.0 -- match
            exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : surface tension, N/m, same type as T
        """
        ranges.check("sodium_sigma", {"T": T}, RANGES["sodium_sigma"])
        Tm = Sodium.Tm
        sig0, A0 = 195, 0.0966
        return (sig0 - A0*(T-Tm))*1E-3

    def cp(T):
        """
        Sodium specific heat capacity.

        Why this model is here:
            Feeds enthalpy-rise (LMprop_plots.py-style Monte Carlo) and energy-balance
            calculations for a sodium-cooled channel.

        Formulation:
            Cp = a + b*T + c*T^2 + d*T^(-2), molar; divided by molar mass M for the
            per-kg value returned.

        Valid range:
            Tm = 371 K to Tb = 1155 K (Sodium.range_cp).

        Uncertainty:
            0 to 1 percent (Sodium.uncert_cp), matching Sobolev (2020) section 4.4's
            stated "about +/-1% for Na".

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.4,
            Equation [12], Table 10 (a=38.12, b=-1.9493e-2, c=1.024e-5, d=-6.9e4 -- match
            exactly, molar values divided by Sodium.M here for the per-kg value returned).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : specific heat capacity, J/kg-K, same type as T
        """
        ranges.check("sodium_cp", {"T": T}, RANGES["sodium_cp"])
        a, b, c, d = 38.12, -1.9493E-2, 1.024E-5, -6.9E4
        Cp = a + b*T + c*T**2 + d*T**(-2)
        return Cp/Sodium.M

    def h(T):
        """
        Sodium specific enthalpy, referenced to the melting point.

        Why this model is here:
            The analytic integral of Sodium.cp from Tm to T, used for channel
            enthalpy-rise calculations (see figures/liquid_metal_uncertainty.py).

        Formulation:
            h = [a*(T-Tm) + (b/2)*(T^2-Tm^2) + (c/3)*(T^3-Tm^3) + d*(1/T - 1/Tm)] / M

        Valid range:
            Tm = 371 K to Tb = 1155 K (Sodium.range_h).

        Uncertainty:
            5 to 7 percent (Sodium.uncert_h), matching Sobolev (2020) section 4.4's
            stated "+/-(5-7)%" (given jointly for Pb and Pb-Bi(e) there, but Sodium's own
            heat-capacity uncertainty in the same section is the tighter +/-1%, so this
            enthalpy figure is carried over from the source without being independently
            re-derived from it).

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.4,
            Equation [14]. The a*(T-Tm), (b/2)*(T^2-Tm^2) and (c/3)*(T^3-Tm^3) terms match
            Equation [14] exactly, but this function's last term is
            d*(1/T - 1/Tm) where Equation [14] gives d*(1/Tm - 1/T) -- the opposite sign.
            Independent check: integrating Sodium.cp's own d*T^-2 term from Tm to T gives
            d*(-1/T) - d*(-1/Tm) = d*(1/Tm - 1/T), i.e. Sobolev's sign, confirming this is
            a real discrepancy and not a transcription difference in the reference.
            Not fixed here per CLAUDE.md ("change no physics") -- see
            docs/OPEN_QUESTIONS.md.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            hout : specific enthalpy, J/kg, same type as T
        """
        ranges.check("sodium_h", {"T": T}, RANGES["sodium_h"])
        Tm = Sodium.Tm
        a, b, c, d = 38.12, -1.9493E-2, 1.024E-5, -6.9E4
        hout = a*(T-Tm) + (b/2)*(T**2-Tm**2) + (c/3)*(T**3-Tm**3) + d*(1/T-1/Tm)
        return hout/Sodium.M

    def mu(T):
        """
        Sodium dynamic viscosity.

        Why this model is here:
            Feeds Reynolds-number and pressure-drop calculations (see liqprops.RePr).

        Formulation:
            mu = mu0 * exp(E0/(R*T))

        Valid range:
            Tm = 371 K to Tb = 1155 K (Sodium.range_mu).

        Uncertainty:
            5 percent (Sodium.uncert_mu), matching Sobolev (2020) section 5.1's stated
            "does not exceed +/-5%" for the most reliable Na recommendations.

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 5.1,
            Equation [17], Table 11 (eta_inf=0.0844e-3, E_eta=6500 -- match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : dynamic viscosity, Pa-s, same type as T
        """
        ranges.check("sodium_mu", {"T": T}, RANGES["sodium_mu"])
        mu0, E0 = 0.0844E-3, 6500
        xp = backend.lib(T)
        return mu0 * xp.exp(E0/(R*T))

    def k(T):
        """
        Sodium thermal conductivity.

        Why this model is here:
            Feeds Prandtl-number and heat-transfer-coefficient calculations (see
            liqprops.RePr and correlations/htc.py::Lead.Shen's liquid-metal form).

        Formulation:
            k = 104 - 0.0466*T

        Valid range:
            Tm = 371 K to Tb = 1155 K (Sodium.range_k).

        Uncertainty:
            0 to 8 percent (Sodium.uncert_k). This does not match Sobolev (2020) section
            5.3, which instead reports that Fink and Leibowitz's examination of the Na
            thermal-conductivity literature found differences of "up to +/-15%" over
            371-1500 K -- a wider band than this module states, from an examination
            rather than a single recommended sigma. Not adjusted to match; see
            docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 5.3,
            Equation [22], Table 13. The formula here (kval = 104 - 0.0466*T) is Table
            13's lambda_M,0 + A_lambda,0*(T-Tm) with Bl,0=0 for Na (lambda_M,0=86.7,
            A_lambda,0=-0.0466, Tm=371.0), algebraically simplified to a single line --
            86.7 - 0.0466*(T-371) = 103.99 - 0.0466*T, matching the "104" constant here
            to within rounding.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            kval : thermal conductivity, W/m-K, same type as T
        """
        ranges.check("sodium_k", {"T": T}, RANGES["sodium_k"])
        kval = 104-0.0466*T
        return kval


class Lead:
    Tm = 600.6
    Tb = 2021
    M = 0.2072

    range_rho = [Tm, Tb]
    range_cp = [Tm, 1100]
    range_h = [Tm, 1100]
    range_mu = [Tm, 1270]
    range_sig = [Tm, Tb]
    range_k = [Tm, 1300]

    uncert_rho = [0.7/100, 0.8/100]
    uncert_sig = [0/100, 5/100]
    uncert_cp = [5/100, 7/100]
    uncert_h = [5/100, 7/100]
    uncert_mu = [5/100, 5/100]
    uncert_k = [0/100, 15/100]

    def rho(T):
        """
        Lead density.

        Why this model is here:
            Feeds channel mass-flux and pressure-drop calculations for a lead-cooled
            channel; per docs/DECISIONS.md the lead/LBE channel in the annular SCA files
            is a toy, so this is property-library infrastructure rather than an active
            SCA path today.

        Formulation:
            rho = rho0 - A0*(T - Tm)

        Valid range:
            Tm = 600.6 K to Tb = 2021 K (Lead.range_rho).

        Uncertainty:
            0.7 to 0.8 percent (Lead.uncert_rho), matching Sobolev (2020) section 4.1's
            stated "0.7-0.8% for Pb and Pb-Bi(e)".

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.1,
            Equation [4], Table 7 (rho0=10671, A0=1.2795, Tm=600.6 -- match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : density, kg/m^3, same type as T
        """
        ranges.check("lead_rho", {"T": T}, RANGES["lead_rho"])
        Tm = Lead.Tm
        rho0, A0 = 10671, 1.2795
        return rho0 - A0*(T-Tm)

    def sigma(T):
        """
        Lead surface tension.

        Why this model is here:
            Feeds two-phase / boiling-margin calculations for a lead-cooled channel.

        Formulation:
            sigma = (sig0 - A0*(T - Tm)) * 1e-3

        Valid range:
            Tm = 600.6 K to Tb = 2021 K (Lead.range_sig).

        Uncertainty:
            0 to 5 percent (Lead.uncert_sig). Sobolev (2020) section 4.3 gives only a
            collective "(3-6)%" spread across Na, Pb and Pb-Bi(e) together, not a
            per-metal number -- this module's [0, 5%] is not independently confirmable
            from that single combined figure. See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.3,
            Equation [11], Table 9 (sigma0=458e-3, A0=0.113e-3, Tm=600.6 -- match
            exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : surface tension, N/m, same type as T
        """
        ranges.check("lead_sigma", {"T": T}, RANGES["lead_sigma"])
        Tm = Lead.Tm
        sig0, A0 = 458, 0.113
        return (sig0 - A0*(T-Tm))*1E-3

    def cp(T):
        """
        Lead specific heat capacity.

        Why this model is here:
            Feeds enthalpy-rise (LMprop_plots.py-style Monte Carlo) and energy-balance
            calculations for a lead-cooled channel.

        Formulation:
            Cp = a + b*T + c*T^2 + d*T^(-2), molar; divided by molar mass M for the
            per-kg value returned.

        Valid range:
            Tm = 600.6 K to 1100 K (Lead.range_cp).

        Uncertainty:
            5 to 7 percent (Lead.uncert_cp), matching Sobolev (2020) section 4.4's stated
            "+/-(5-7)% for Pb and Pb-Bi(e)".

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.4,
            Equation [12], Table 10 (a=36.50, b=-1.020e-2, c=3.2e-6, d=-3.158e5 -- match
            exactly, molar values divided by Lead.M here for the per-kg value returned).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : specific heat capacity, J/kg-K, same type as T
        """
        ranges.check("lead_cp", {"T": T}, RANGES["lead_cp"])
        a, b, c, d = 36.5, -1.020E-2, 3.2E-6, -3.158E5
        Cp = a + b*T + c*T**2 + d*T**(-2)
        return Cp/Lead.M

    def h(T):
        """
        Lead specific enthalpy, referenced to the melting point.

        Why this model is here:
            The analytic integral of Lead.cp from Tm to T, used for channel
            enthalpy-rise calculations (see figures/liquid_metal_uncertainty.py).

        Formulation:
            h = [a*(T-Tm) + (b/2)*(T^2-Tm^2) + (c/3)*(T^3-Tm^3) + d*(1/T - 1/Tm)] / M

        Valid range:
            Tm = 600.6 K to 1100 K (Lead.range_h).

        Uncertainty:
            5 to 7 percent (Lead.uncert_h), matching Sobolev (2020) section 4.4's stated
            "+/-(5-7)% for Pb and Pb-Bi(e)" heat-capacity uncertainty, carried over to the
            enthalpy this integrates.

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.4,
            Equation [14]. Same sign discrepancy in the last term as Sodium.h -- see that
            function's docstring for the derivation. Equation [14] gives
            d*(1/Tm - 1/T); this function computes d*(1/T - 1/Tm). Not fixed here per
            CLAUDE.md ("change no physics") -- see docs/OPEN_QUESTIONS.md.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            hout : specific enthalpy, J/kg, same type as T
        """
        ranges.check("lead_h", {"T": T}, RANGES["lead_h"])
        Tm = Lead.Tm
        a, b, c, d = 36.5, -1.020E-2, 3.2E-6, -3.158E5
        hout = a*(T-Tm) + (b/2)*(T**2-Tm**2) + (c/3)*(T**3-Tm**3) + d*(1/T-1/Tm)
        return hout/Lead.M

    def mu(T):
        """
        Lead dynamic viscosity.

        Why this model is here:
            Feeds Reynolds-number and pressure-drop calculations (see liqprops.RePr),
            and correlations/htc.py::Lead.Shen's Peclet number.

        Formulation:
            mu = mu0 * exp(E0/(R*T))

        Valid range:
            Tm = 600.6 K to 1270 K (Lead.range_mu).

        Uncertainty:
            5 percent (Lead.uncert_mu), matching Sobolev (2020) section 5.1's stated
            "+/-5%" for Pb over TM,0 to 1270 K.

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 5.1,
            Equation [17], Table 11 (eta_inf=0.455e-3, E_eta=8888 -- match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : dynamic viscosity, Pa-s, same type as T
        """
        ranges.check("lead_mu", {"T": T}, RANGES["lead_mu"])
        mu0, E0 = 0.455E-3, 8888
        xp = backend.lib(T)
        return mu0 * xp.exp(E0/(R*T))

    def k(T):
        """
        Lead thermal conductivity.

        Why this model is here:
            Feeds Prandtl-number and heat-transfer-coefficient calculations (see
            liqprops.RePr and correlations/htc.py::Lead.Shen).

        Formulation:
            k = 15.8 + 0.011*(T - Tm)

        Valid range:
            Tm = 600.6 K to 1300 K (Lead.range_k).

        Uncertainty:
            0 to 15 percent (Lead.uncert_k), matching Sobolev (2020) section 5.3's stated
            "maximum difference of +/-15%" for the recommended Pb correlation over
            TM,0-1300 K.

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 5.3,
            Equation [22], Table 13 (lambda_M,0=15.8, A_lambda,0=0.011, Bl,0=0,
            Tm=600.6 -- match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            kval : thermal conductivity, W/m-K, same type as T
        """
        ranges.check("lead_k", {"T": T}, RANGES["lead_k"])
        Tm = Lead.Tm
        kval = 15.8 + 0.011*(T-Tm)
        return kval


class LBE:
    Tm = 398
    Tb = 1927
    M = 0.20818

    range_rho = [Tm, Tb]
    range_cp = [Tm, 1100]
    range_h = [Tm, 1100]
    range_mu = [Tm, 1180]
    range_sig = [Tm, Tb]
    range_k = [Tm, 1100]

    uncert_rho = [0.7/100, 0.8/100]
    uncert_sig = [0/100, 0.3/100]
    uncert_cp = [5/100, 7/100]
    uncert_h = [5/100, 7/100]
    uncert_mu = [7/100, 10/100]
    uncert_k = [10/100, 15/100]

    def rho(T):
        """
        Lead-bismuth eutectic (LBE) density.

        Why this model is here:
            Feeds channel mass-flux and pressure-drop calculations for an LBE-cooled
            channel.

        Formulation:
            rho = rho0 - A0*(T - Tm)

        Valid range:
            Tm = 398 K to Tb = 1927 K (LBE.range_rho).

        Uncertainty:
            0.7 to 0.8 percent (LBE.uncert_rho), matching Sobolev (2020) section 4.1's
            stated "0.7-0.8% for Pb and Pb-Bi(e)".

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.1,
            Equation [4], Table 7 (rho0=10550, A0=1.293, Tm=398 -- match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : density, kg/m^3, same type as T
        """
        ranges.check("lbe_rho", {"T": T}, RANGES["lbe_rho"])
        Tm = LBE.Tm
        rho0, A0 = 10550, 1.293
        return rho0 - A0*(T-Tm)

    def sigma(T):
        """
        LBE surface tension.

        Why this model is here:
            Feeds two-phase / boiling-margin calculations for an LBE-cooled channel.

        Formulation:
            sigma = (sig0 - A0*(T - Tm)) * 1e-3

        Valid range:
            Tm = 398 K to Tb = 1927 K (LBE.range_sig).

        Uncertainty:
            0 to 0.3 percent (LBE.uncert_sig). As with Lead.sigma, Sobolev (2020) section
            4.3 gives only a collective "(3-6)%" spread across all three coolants, not a
            per-metal number, so this module's much tighter [0, 0.3%] is not
            independently confirmable from that figure. See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.3,
            Equation [11], Table 9 (sigma0=416.7e-3, A0=0.0799e-3, Tm=398 -- match
            exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : surface tension, N/m, same type as T
        """
        ranges.check("lbe_sigma", {"T": T}, RANGES["lbe_sigma"])
        Tm = LBE.Tm
        sig0, A0 = 416.7, 0.0799
        return (sig0 - A0*(T-Tm))*1E-3

    def cp(T):
        """
        LBE specific heat capacity.

        Why this model is here:
            Feeds enthalpy-rise and energy-balance calculations for an LBE-cooled
            channel.

        Formulation:
            Cp = a + b*T + c*T^2 + d*T^(-2), molar; divided by molar mass M for the
            per-kg value returned.

        Valid range:
            Tm = 398 K to 1100 K (LBE.range_cp).

        Uncertainty:
            5 to 7 percent (LBE.uncert_cp), matching Sobolev (2020) section 4.4's stated
            "+/-(5-7)% for Pb and Pb-Bi(e)".

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.4,
            Equation [12], Table 10 (a=34.30, b=-8.20e-3, c=2.6e-6, d=-9.5e4 -- match
            exactly, molar values divided by LBE.M here for the per-kg value returned).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : specific heat capacity, J/kg-K, same type as T
        """
        ranges.check("lbe_cp", {"T": T}, RANGES["lbe_cp"])
        a, b, c, d = 34.3, -8.2E-3, 2.6E-6, -9.5E4
        Cp = a + b*T + c*T**2 + d*T**(-2)
        return Cp/LBE.M

    def h(T):
        """
        LBE specific enthalpy, referenced to the melting point.

        Why this model is here:
            The analytic integral of LBE.cp from Tm to T, used for channel
            enthalpy-rise calculations.

        Formulation:
            h = [a*(T-Tm) + (b/2)*(T^2-Tm^2) + (c/3)*(T^3-Tm^3) + d*(1/T - 1/Tm)] / M

        Valid range:
            Tm = 398 K to 1100 K (LBE.range_h).

        Uncertainty:
            5 to 7 percent (LBE.uncert_h), matching Sobolev (2020) section 4.4's stated
            "+/-(5-7)% for Pb and Pb-Bi(e)" heat-capacity uncertainty, carried over to
            the enthalpy this integrates.

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 4.4,
            Equation [14]. Same sign discrepancy in the last term as Sodium.h -- see that
            function's docstring for the derivation. Equation [14] gives
            d*(1/Tm - 1/T); this function computes d*(1/T - 1/Tm). Not fixed here per
            CLAUDE.md ("change no physics") -- see docs/OPEN_QUESTIONS.md.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            hout : specific enthalpy, J/kg, same type as T
        """
        ranges.check("lbe_h", {"T": T}, RANGES["lbe_h"])
        Tm = LBE.Tm
        a, b, c, d = 34.3, -8.2E-3, 2.6E-6, -9.5E4
        hout = a*(T-Tm) + (b/2)*(T**2-Tm**2) + (c/3)*(T**3-Tm**3) + d*(1/T-1/Tm)
        return hout/LBE.M

    def mu(T):
        """
        LBE dynamic viscosity.

        Why this model is here:
            Feeds Reynolds-number and pressure-drop calculations (see liqprops.RePr).

        Formulation:
            mu = mu0 * exp(E0/(R*T))

        Valid range:
            Tm = 398 K to 1180 K (LBE.range_mu).

        Uncertainty:
            7 to 10 percent (LBE.uncert_mu), matching Sobolev (2020) section 5.1's stated
            "higher variation (7-10%)" for Pb-Bi(e) viscosity recommendations.

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 5.1,
            Equation [17], Table 11 (eta_inf=0.494e-3, E_eta=6270 -- match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : dynamic viscosity, Pa-s, same type as T
        """
        ranges.check("lbe_mu", {"T": T}, RANGES["lbe_mu"])
        mu0, E0 = 0.494E-3, 6270
        xp = backend.lib(T)
        return mu0 * xp.exp(E0/(R*T))

    def k(T):
        """
        LBE thermal conductivity.

        Why this model is here:
            Feeds Prandtl-number and heat-transfer-coefficient calculations (see
            liqprops.RePr).

        Formulation:
            k = lam + A*(T - Tm) + B*(T - Tm)^2

        Valid range:
            Tm = 398 K to 1100 K (LBE.range_k).

        Uncertainty:
            10 to 15 percent (LBE.uncert_k), matching Sobolev (2020) section 5.3's stated
            "uncertainty of 10-15%" for the Pb-Bi(e) parabolic thermal-conductivity fit
            up to 1100 K.

        Reference:
            Sobolev, V. (2020), "Properties of Liquid Metal Coolants: Na, Pb, Pb-Bi",
            Reference Module in Materials Science and Materials Engineering, section 5.3,
            Equation [22], Table 13 (lambda_M,0=9.35, A_lambda,0=0.01434,
            Bl,0=2.305e-6, Tm=398 -- match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            kval : thermal conductivity, W/m-K, same type as T
        """
        ranges.check("lbe_k", {"T": T}, RANGES["lbe_k"])
        Tm = LBE.Tm
        lam, A, B = 9.35, 0.01434, 2.305E-6
        kval = lam + A*(T-Tm) + B*(T-Tm)**2
        return kval


# Built directly from the range_* attributes each class already carries (the values
# D11/docs/DUPLICATES.md audited and, where wrong, already corrected) rather than
# inventing anything new -- one plain dictionary per CLAUDE.md section 7, keyed by the
# name each property's ranges.check() call above already uses.
RANGES = {
    "sodium_rho":   {"T": tuple(Sodium.range_rho)},
    "sodium_sigma": {"T": tuple(Sodium.range_sig)},
    "sodium_cp":    {"T": tuple(Sodium.range_cp)},
    "sodium_h":     {"T": tuple(Sodium.range_h)},
    "sodium_mu":    {"T": tuple(Sodium.range_mu)},
    "sodium_k":     {"T": tuple(Sodium.range_k)},
    "lead_rho":     {"T": tuple(Lead.range_rho)},
    "lead_sigma":   {"T": tuple(Lead.range_sig)},
    "lead_cp":      {"T": tuple(Lead.range_cp)},
    "lead_h":       {"T": tuple(Lead.range_h)},
    "lead_mu":      {"T": tuple(Lead.range_mu)},
    "lead_k":       {"T": tuple(Lead.range_k)},
    "lbe_rho":      {"T": tuple(LBE.range_rho)},
    "lbe_sigma":    {"T": tuple(LBE.range_sig)},
    "lbe_cp":       {"T": tuple(LBE.range_cp)},
    "lbe_h":        {"T": tuple(LBE.range_h)},
    "lbe_mu":       {"T": tuple(LBE.range_mu)},
    "lbe_k":        {"T": tuple(LBE.range_k)},
}


def Props(mat, T):
    """
    Bundle a liquid metal's full property set at one temperature into a dict.

    Why this model is here:
        The single entry point correlations/friction.py and correlations/htc.py expect
        (a 'Props' dict of rho/mu/k/cp, plus here sigma and h) -- called from
        properties/getprop.py for the "Lead"/"Pb"/"Sodium"/"Na" substance names.

    Formulation:
        Calls mat.rho(T), mat.sigma(T), mat.cp(T), mat.h(T), mat.mu(T), mat.k(T) and
        collects the results.

    Valid range:
        Whatever the weakest of the six per-property ranges on `mat` covers -- see that
        class's own range_* attributes.

    Uncertainty:
        Not applicable -- a dict assembly, not a correlation.

    Reference:
        Not applicable.

    Inputs:
        mat : one of Sodium, Lead, LBE (the class itself, not an instance)
        T   : temperature (float, numpy array, or torch tensor), K
    Returns:
        props : dict with keys 'rho', 'sigma', 'cp', 'h', 'mu', 'k', each the same type
                as T
    """
    props = {
        'rho': mat.rho(T),
        'sigma': mat.sigma(T),
        'cp': mat.cp(T),
        'h': mat.h(T),
        'mu': mat.mu(T),
        'k': mat.k(T)
    }
    return props


def RePr(G, D, prop):
    """
    Reynolds and Prandtl numbers from a mass flux, hydraulic diameter and Props dict.

    Why this model is here:
        The shared Re/Pr calculation used ahead of a liquid-metal heat-transfer
        correlation (e.g. correlations/htc.py::Lead.Shen).

    Formulation:
        Re = G*D/mu
        Pr = mu*cp/k

    Valid range:
        Not applicable -- a dimensionless-group calculation, not a correlation.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable -- standard definitions.

    Inputs:
        G    : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
        D    : hydraulic diameter, m (float, numpy array, or torch tensor)
        prop : Props dict (see liqprops.Props) with keys 'mu', 'cp', 'k'
    Returns:
        (Re, Pr) : Reynolds number, Prandtl number, both dimensionless, same type as G
    """
    mu, cp, k = prop['mu'], prop['cp'], prop['k']
    Re = G*D/mu
    Pr = mu*cp/k
    return Re, Pr
