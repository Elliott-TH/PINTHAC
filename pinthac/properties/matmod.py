from pinthac.backend import lib as compat


class Gas:
    def k(gas,T):
        '''
        Matlib gas thermal conductivity model

        Inputs:
        gas (string): Gas type [Symbolic (eg He, O2)]
        T (float): Temperature [Kelvin]
        '''
        dic = {
                "He": [2.531E-3, 0.7146],
                "Ar": [4.092E-4, 0.6748],
                "Kr": [1.966E-4, 0.7006],
                "Xe": [9.825E-5, 0.7334],
                "H2": [1.349E-3, 0.8408],
                "N2": [2.984E-4, 0.7799],
                "Air": [1.945E-4, 0.8586]
            }
        A,B = dic[gas]
        kval = A*T**B
        return kval


class UO2:
    def k_NFI(T,Bu=0.0,f_gad=0.0):
        """
        Modified Frapcon-4 model for UO2 thermal conductivity,
        taking into acoount gadolinia fraction.

        Inputs:
        T (float): Temperature [K]
        Bu (float): Burnup [GWd/MTU]
        f_gad (float): gadolinia weight fraction []
        """
        A = 0.0452
        B = 2.46E-4
        E = 3.5E9
        F = 16361
        a = 1.1599
        Q = 6380

        lib = compat(T, Bu)
        f_Bu = 0.00187 * Bu
        g_Bu = 0.038 * Bu**(0.28)
        h_T = 1/(1+396*lib.exp(-Q/T))

        denom_1 = A + a*f_gad + B*T + f_Bu
        denom_2 = (1-0.9*lib.exp(-0.04*Bu))*g_Bu*h_T
        Term1 = 1/(denom_1+denom_2)
        Term2 = E/(T**2) * lib.exp(-F/T)
        K = Term1 + Term2
        return K

    def k_Klimenko(T):
        """
        Klimenki-Zorin model for thermal conductivity of UO2 fuel with
        95% densities

        Inputs:
        T (float): Temperature [K]
        """
        lib = compat(T)
        tau = T/1000
        denom = 7.5408+17.692*tau+3.6142*tau**2
        exp_term = lib.exp(-16.35/tau)
        kval = 100/denom + 6400*exp_term * tau**(-5/2)
        return kval

    def eps(T):
        """
        Correlation for the emmisivity of UO2 fuel as it
        appears in MATPRO, error of +- 6.8%

        Inputs:
        T (float): Temperature [K]
        """
        a = 0.7856
        b = 1.5263E-5
        val = a + b*T
        return val

    def thrm_expan(T):
        """
        Frapcon-4 model for thermal expansion of solid UO2 fuel

        Inputs:
        T (float): Temperature [K]
        """
        K1 = 9.8E-6
        K2 = 2.61E-3
        K3 = 3.16E-1
        Ed = 1.32E-19
        k = 1.38E-23
        lib = compat(T)
        strain = K1*T - K2 + K3*lib.exp(-Ed/(k*T))
        return strain

    def dens_max(T,rho_TD=95.0,T_sint=1873.15):
        """
        Maximum in-reactor dimension change due to densification, based
        on the fabrication sintering temperature

        Inputs:
        T (float): Fuel temperature [K]
        rho_TD (float): Initial density [% theoretical]
        T_sint (float): Sintering temperature [K]
        """
        lib = compat(T)
        low = -22.2*(100-rho_TD)/(T_sint-1453.15)
        high = -66.6*(100-rho_TD)/(T_sint-1453.15)
        val = lib.where(T<1000, low, high)
        return val

    def dens_B(dL_max,n_iter=100):
        """
        Offset constant B in the densification model, found by bisection so
        that dL/L = 0 at zero burnup

        Inputs:
        dL_max (float): Maximum dimension change [%]
        n_iter (int): Number of bisection iterations []
        """
        lib = compat(dL_max)
        target = -dL_max
        lo = lib.zeros_like(target)
        hi = lib.zeros_like(target) + 50.0
        for i in range(n_iter):
            mid = 0.5*(lo+hi)
            f = lib.exp(-3*mid) + 2*lib.exp(-35*mid) - target
            lo = lib.where(f>0, mid, lo)
            hi = lib.where(f>0, hi, mid)
        Bval = 0.5*(lo+hi)
        return Bval

    def densification(T,Bu,rho_TD=95.0,T_sint=1873.15):
        """
        Rolstad model for the densification of UO2, MOX and UO2-Gd2O3

        Inputs:
        T (float): Fuel temperature [K]
        Bu (float): Burnup [MWd/kgU]
        rho_TD (float): Initial density [% theoretical]
        T_sint (float): Sintering temperature [K]
        """
        lib = compat(T, Bu)
        dL_max = UO2.dens_max(T,rho_TD,T_sint)
        B = UO2.dens_B(dL_max)
        Term1 = lib.exp(-3*(Bu+B))
        Term2 = 2*lib.exp(-35*(Bu+B))
        val = dL_max + Term1 + Term2
        return val

    def swelling_solid(Bu,gad=False):
        """
        Solid fission product swelling of UO2, MOX and UO2-Gd2O3

        Inputs:
        Bu (float): Pellet average burnup [GWd/MTU]
        gad (bool): Gadolinia bearing fuel []
        """
        lib = compat(Bu)
        if gad:
            val = 0.0005*Bu
            return val
        low = lib.zeros_like(lib.asarray(Bu)*1.0)
        mid = 0.00062*(Bu-6)
        high = 0.00062*(80-6) + 0.00086*(Bu-80)
        val = lib.where(Bu<=6, low, lib.where(Bu<=80, mid, high))
        return val

    def swelling_gas(T,Bu):
        """
        Gaseous swelling of UO2, UO2-Gd2O3 and MOX, active above
        40 GWd/MTU and between 1233 and 2105 K

        Inputs:
        T (float): Pellet ring temperature [K]
        Bu (float): Pellet average burnup [GWd/MTU]
        """
        lib = compat(T, Bu)
        low = -4.37E-2 + 4.55E-5*T
        high = 7.40E-2 - 4.05E-5*T
        zero = lib.zeros_like(lib.asarray(T)*1.0)
        shape = lib.where((T>=1233)&(T<1643), low,
                 lib.where((T>=1643)&(T<=2105), high, zero))
        ramp = lib.where(Bu<40, 0.0, lib.where(Bu<50, (Bu-40)/10, 1.0))
        val = ramp*shape
        return val


class Zircalloy:
    def k(T):
        """
        Matlib model for the thermal conductivity of Zircaloy-4,
        Zircaloy-2, M5, ZIRLO and Optimized ZIRLO, sigma = 1.9 W/m-K

        Inputs:
        T (float): Temperature [K]
        """
        lib = compat(T)
        poly = 7.511 + 2.088E-2*T - 1.45E-5*T**2 + 7.668E-9*T**3
        val = lib.where(T>=2098, 36.0, poly)
        return val

    def cp(T):
        """
        Lookup table for the specific heat capacity of Zircaloy-4,
        Zircaloy-2, M5, ZIRLO and Optimized ZIRLO

        Inputs:
        T (float): Temperature [K]
        """
        lib = compat(T)
        T_tab = [290.0, 300.0, 400.0, 640.0, 1090.0, 1093.0, 1113.0, 1133.0,
                 1153.0, 1173.0, 1193.0, 1213.0, 1233.0, 1248.0]
        C_tab = [279.0, 281.0, 302.0, 331.0, 375.0, 502.0, 590.0, 615.0,
                 719.0, 816.0, 770.0, 619.0, 469.0, 356.0]
        val = lib.interp(T,T_tab,C_tab)
        return val

    def T_melt():
        """
        Melting temperature of Zircaloy-4, Zircaloy-2, M5, ZIRLO and
        Optimized ZIRLO
        """
        val = 2123.15
        return val

    def rho():
        """
        Density of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO
        """
        val = 6520.0
        return val

    def thrm_expan_axial(T):
        """
        Axial thermal expansion of Zircaloy-4, Zircaloy-2, M5, ZIRLO and
        Optimized ZIRLO, sigma = 4.8E-5 m/m

        Inputs:
        T (float): Temperature [K]
        """
        lib = compat(T)
        T_tab = [1073.15, 1083.15, 1093.15, 1103.15, 1113.15, 1123.15,
                 1133.15, 1143.15, 1153.15, 1163.15, 1173.15, 1183.15,
                 1193.15, 1203.15, 1213.15, 1223.15, 1233.15, 1243.15,
                 1253.15, 1263.15, 1273.15]
        e_tab = [0.00352774, 0.00353000, 0.00350000, 0.00346000, 0.00341000,
                 0.00333000, 0.00321000, 0.00307000, 0.00280000, 0.00250000,
                 0.00200000, 0.00150000, 0.00130000, 0.00116000, 0.00113000,
                 0.00110000, 0.00111000, 0.00113000, 0.00120000, 0.00130000,
                 0.00140000]
        low = -2.5060E-5 + 4.4410E-6*(T-273.15)
        high = -8.300E-3 + 9.70E-6*(T-273.15)
        table = lib.interp(T,T_tab,e_tab)
        val = lib.where(T<=1073.15, low, lib.where(T>=1273.15, high, table))
        return val

    def thrm_expan_diametral(T):
        """
        Circumferential thermal expansion of Zircaloy-4, Zircaloy-2, M5,
        ZIRLO and Optimized ZIRLO, sigma = 4.6E-4 m/m

        Inputs:
        T (float): Temperature [K]
        """
        lib = compat(T)
        T_tab = [1073.15, 1083.15, 1093.15, 1103.15, 1113.15, 1123.15,
                 1133.15, 1143.15, 1153.15, 1163.15, 1173.15, 1183.15,
                 1193.15, 1203.15, 1213.15, 1223.15, 1233.15, 1243.15,
                 1253.15, 1263.15, 1273.15]
        e_tab = [0.00513950, 0.00522000, 0.00525000, 0.00528000, 0.00528000,
                 0.00524000, 0.00522000, 0.00515000, 0.00508000, 0.00490000,
                 0.00470000, 0.00445000, 0.00410000, 0.00350000, 0.00313000,
                 0.00297000, 0.00292000, 0.00287000, 0.00286000, 0.00288000,
                 0.00290000]
        low = -2.3730E-4 + 6.7210E-6*(T-273.15)
        high = -6.800E-3 + 9.70E-6*(T-273.15)
        table = lib.interp(T,T_tab,e_tab)
        val = lib.where(T<=1073.15, low, lib.where(T>=1273.15, high, table))
        return val

    def eps(T,t_ox=0.0):
        """
        Emissivity of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized
        ZIRLO, sigma = 0.054

        Inputs:
        T (float): Temperature [K]
        t_ox (float): Inner surface oxide thickness [m]
        """
        lib = compat(T)
        eps_1 = lib.where(t_ox<3.88E-6, 0.325+0.1246E6*t_ox,
                                       0.808642-50.0*t_ox)
        eps_2 = lib.maximum(0.325, lib.exp((1500-T)/300)*eps_1)
        val = lib.where(T>1500, eps_2, eps_1)
        return val

    def E(T,CW=0.0,phi=0.0,d_ox=0.0012):
        """
        Young's modulus of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized
        ZIRLO, linearly interpolated between 1090 and 1255 K

        Inputs:
        T (float): Temperature [K]
        CW (float): Effective cold work, ratio of areas []
        phi (float): Fast neutron (>1 MeV) fluence [n/m^2]
        d_ox (float): Average oxygen concentration [kg-O/kg-Zr]
        """
        lib = compat(T)
        c2 = 0.88 + 0.12*lib.exp(-phi/1E25)
        c3 = -2.6E10
        low = (1.088E11 - 5.475E7*T + (6.61E11+5.912E8*T)*d_ox + c3*CW)/c2
        high = 9.21E10 - 4.05E7*T
        E_1090 = (1.088E11 - 5.475E7*1090.0
                  + (6.61E11+5.912E8*1090.0)*d_ox + c3*CW)/c2
        E_1255 = 9.21E10 - 4.05E7*1255.0
        mid = E_1090 + (E_1255-E_1090)*(T-1090.0)/(1255.0-1090.0)
        val = lib.where(T<1090, low, lib.where(T>1255, high, mid))
        return val

    def G(T,CW=0.0,phi=0.0,d_ox=0.0012):
        """
        Shear modulus of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized
        ZIRLO, linearly interpolated between 1090 and 1255 K.
        PNNL-35702 adds the bare constant c3 here rather than c3*CW.

        Inputs:
        T (float): Temperature [K]
        CW (float): Effective cold work, ratio of areas []
        phi (float): Fast neutron (>1 MeV) fluence [n/m^2]
        d_ox (float): Average oxygen concentration [kg-O/kg-Zr]
        """
        lib = compat(T)
        c2 = 0.88 + 0.12*lib.exp(-phi/1E25)
        c3 = -0.867E10
        low = (4.04E10 - 2.168E7*T + (7.07E11-2.315E8*T)*d_ox + c3)/c2
        high = 3.49E10 - 1.66E7*T
        G_1090 = (4.04E10 - 2.168E7*1090.0
                  + (7.07E11-2.315E8*1090.0)*d_ox + c3)/c2
        G_1255 = 3.49E10 - 1.66E7*1255.0
        mid = G_1090 + (G_1255-G_1090)*(T-1090.0)/(1255.0-1090.0)
        val = lib.where(T<1090, low, lib.where(T>1255, high, mid))
        return val

    def meyer_hardness(T):
        """
        Meyer's hardness of Zircaloy-2, Zircaloy-4, M5, ZIRLO and
        Optimized ZIRLO

        Inputs:
        T (float): Temperature [K]
        """
        lib = compat(T)
        arg = (26.034 - 2.6394E-2*T + 4.3502E-5*T**2 - 2.5621E-8*T**3)
        val = lib.where(T<=1235, lib.exp(arg), 1.0E5)
        return val

    def axial_growth(alloy,phi):
        """
        Axial irradiation growth of zirconium based cladding, valid for
        fuel rod cladding only

        Inputs:
        alloy (string): Cladding alloy [Zircaloy-2, Zircaloy-4, ZIRLO,
                        Optimized ZIRLO, M5]
        phi (float): Fast neutron (>1 MeV) fluence [n/cm^2]
        """
        dic = {
                "Zircaloy-2": [1.09E-21, 0.845],
                "Zircaloy-4": [2.18E-21, 0.845],
                "ZIRLO": [9.7893E-25, 0.98239],
                "Optimized ZIRLO": [9.7893E-25, 0.98239],
                "M5": [7.013E-21, 0.81787]
            }
        A,n = dic[alloy]
        val = A*phi**n
        return val

    def sigma_eff(Pi,Po,ri,ro,r=None):
        """
        Effective cladding stress from the thick wall principal stresses,
        evaluated at the mid wall radius by default

        Inputs:
        Pi (float): Inner pressure [MPa]
        Po (float): Outer pressure [MPa]
        ri (float): Inner radius [cm]
        ro (float): Outer radius [cm]
        r (float): Radius within the tube [cm]
        """
        lib = compat(Pi, Po)
        if r is None:
            r = 0.5*(ri+ro)
        denom = ro**2 - ri**2
        common = Pi*ri**2 - Po*ro**2
        extra = ri**2*ro**2*(Po-Pi)/r**2
        sig_r = (common+extra)/denom
        sig_t = (common-extra)/denom
        sig_l = common/denom
        val = lib.sqrt(0.5*((sig_l-sig_t)**2+(sig_t-sig_r)**2+(sig_r-sig_l)**2))
        return val

    def cw_type(alloy):
        """
        Cold work class used by the strain rate correlations

        Inputs:
        alloy (string): Cladding alloy [Zircaloy-2, Zircaloy-4, ZIRLO,
                        Optimized ZIRLO, M5]
        """
        dic = {
                "Zircaloy-2": "RXA",
                "Zircaloy-4": "SRA",
                "M5": "RXA",
                "ZIRLO": "SRA",
                "Optimized ZIRLO": "SRA"
            }
        val = dic[alloy]
        return val

    def strain_rate_thermal(T,sig,Phi,cw="SRA"):
        """
        Thermal strain rate of zirconium based cladding

        Inputs:
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        Phi (float): Fast neutron (>1 MeV) fluence [n/cm^2]
        cw (string): Cold work class [SRA, RXA]
        """
        lib = compat(T, sig)
        dic = {
                "SRA": [1.08E9, 2.0],
                "RXA": [5.47E8, 3.5]
            }
        A,n = dic[cw]
        Q = 201000.0
        R = 8.314
        E = 1.148E5 - 59.9*T
        a_i = 650*(1 - 0.56*(1 - lib.exp(-1.4E-27*Phi**1.3)))
        val = A*(E/T)*lib.sinh(a_i*sig/E)**n * lib.exp(-Q/(R*T))
        return val

    def strain_rate_irrad(T,sig,flux,cw="SRA"):
        """
        Irradiation strain rate of zirconium based cladding

        Inputs:
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        flux (float): Fast neutron (>1 MeV) flux [n/m^2-s]
        cw (string): Cold work class [SRA, RXA]
        """
        lib = compat(T, sig)
        dic = {
                "SRA": [4.0985E-24, 0.7283, -7.0237, 0.0136, 1.4763],
                "RXA": [1.87473E-24, 0.7994, -3.18562, 0.006699132, 1.1840]
            }
        c0,f_lo,f_a,f_b,f_hi = dic[cw]
        c1 = 0.85
        c2 = 1.0
        f_T = lib.where(T<=570, f_lo, lib.where(T>=625, f_hi, f_a+f_b*T))
        val = c0*flux**c1 * sig**c2 * f_T
        return val

    def strain_sat_primary(eps_dot):
        """
        Saturated primary hoop strain

        Inputs:
        eps_dot (float): Combined thermal and irradiation strain rate [1/hr]
        """
        lib = compat(eps_dot)
        val = 0.0216*eps_dot**0.109 * (2-lib.tanh(3.55E4*eps_dot))**(-2.05)
        return val

    def creep_strain(alloy,T,sig,flux,Phi,t):
        """
        Total hoop creep strain of zirconium based cladding

        Inputs:
        alloy (string): Cladding alloy [Zircaloy-2, Zircaloy-4, M5, ZIRLO,
                        Optimized ZIRLO]
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        flux (float): Fast neutron (>1 MeV) flux [n/m^2-s]
        Phi (float): Fast neutron (>1 MeV) fluence [n/cm^2]
        t (float): Time [hr]
        """
        lib = compat(T, sig)
        cw = Zircalloy.cw_type(alloy)
        eps_th = Zircalloy.strain_rate_thermal(T,sig,Phi,cw)
        eps_irr = Zircalloy.strain_rate_irrad(T,sig,flux,cw)
        eps_dot = eps_th + eps_irr
        eps_sp = Zircalloy.strain_sat_primary(eps_dot)
        val = eps_sp*(1-lib.exp(-52*lib.sqrt(t*eps_dot))) + eps_dot*t
        if alloy in ("ZIRLO","Optimized ZIRLO"):
            val = 0.8*val
        return val

    def creep_rate(alloy,T,sig,flux,Phi,t):
        """
        Total hoop creep rate of zirconium based cladding, sigma = 21.6%
        for Zircaloy-2 and M5, sigma = 14.5% for Zircaloy-4, ZIRLO and
        Optimized ZIRLO

        Inputs:
        alloy (string): Cladding alloy [Zircaloy-2, Zircaloy-4, M5, ZIRLO,
                        Optimized ZIRLO]
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        flux (float): Fast neutron (>1 MeV) flux [n/m^2-s]
        Phi (float): Fast neutron (>1 MeV) fluence [n/cm^2]
        t (float): Time [hr]
        """
        lib = compat(T, sig)
        cw = Zircalloy.cw_type(alloy)
        eps_th = Zircalloy.strain_rate_thermal(T,sig,Phi,cw)
        eps_irr = Zircalloy.strain_rate_irrad(T,sig,flux,cw)
        eps_dot = eps_th + eps_irr
        eps_sp = Zircalloy.strain_sat_primary(eps_dot)
        Term1 = 26*eps_sp*lib.sqrt(eps_dot)/lib.sqrt(t)
        Term2 = lib.exp(-52*lib.sqrt(t*eps_dot))
        val = Term1*Term2 + eps_dot
        if alloy in ("ZIRLO","Optimized ZIRLO"):
            val = 0.8*val
        return val


class HT9:
    def k(T):
        """
        Akiyama model for the thermal conductivity of HT-9 cladding

        Inputs:
        T (float): Temperature [K]
        """
        A0 = 22.47
        A1 = 4.397E-3
        val = A0 + A1*T
        return val

    def cp(T):
        """
        Yamanouchi model for the specific heat capacity of HT-9 cladding

        Inputs:
        T (float): Temperature [K]
        """
        lib = compat(T)
        low = 416.642 + 0.167*T
        high = 69.910 + 0.600*T
        val = lib.where(T<800.15, low, high)
        return val

    def T_melt():
        """
        Melting temperature of HT-9 cladding, taken as the eutectic
        temperature between HT-9 and metallic fuel
        """
        val = 973.0
        return val

    def rho():
        """
        Density of HT-9 cladding
        """
        val = 7750.0
        return val

    def eps():
        """
        Emissivity of HT-9 cladding, no temperature or burnup dependence
        """
        val = 0.9
        return val

    def thrm_expan(T):
        """
        Yamanouchi model for the thermal expansion of HT-9 cladding,
        assumed isotropic

        Inputs:
        T (float): Temperature [K]
        """
        A1 = -2.882E-3
        A2 = 9.226E-6
        A3 = 1.842E-9
        strain = A1 + A2*T + A3*T**2
        return strain

    def E(T):
        """
        Akiyama model for the Young's modulus of HT-9 cladding

        Inputs:
        T (float): Temperature [C]
        """
        A0 = 2.137E11
        A1 = -1.0274E8
        val = A0 + A1*T
        return val

    def G(T):
        """
        Shear modulus of HT-9 cladding

        Inputs:
        T (float): Temperature [C]
        """
        A0 = 8.964E10
        A1 = -5.378E7
        val = A0 + A1*T
        return val

    def meyer_hardness(T):
        """
        Meyer's hardness of HT-9 cladding, taken as the zirconium based
        cladding model

        Inputs:
        T (float): Temperature [K]
        """
        val = Zircalloy.meyer_hardness(T)
        return val

    def strain_rate_primary(T,sig,t):
        """
        Akiyama model for the primary thermal strain rate of HT-9 cladding

        Inputs:
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        t (float): Time [s]
        """
        lib = compat(T, sig)
        C1 = 13.4
        C2 = 8.43E-3
        C3 = 4.08E18
        C4 = 1.6E-6
        Q1 = 15027.0
        Q2 = 26451.0
        Q3 = 89167.0
        R = 1.987
        Term1 = C1*sig*lib.exp(-Q1/(R*T))
        Term2 = C2*sig**4*lib.exp(-Q2/(R*T))
        Term3 = C3*lib.sqrt(sig)*lib.exp(-Q3/(R*T))
        val = (Term1+Term2+Term3)*C4*lib.exp(-C4*t)
        return val

    def strain_rate_secondary(T,sig):
        """
        Akiyama model for the secondary thermal strain rate of HT-9 cladding

        Inputs:
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        """
        lib = compat(T, sig)
        C5 = 1.17E9
        C6 = 8.33E9
        Q4 = 83142.0
        Q5 = 108276.0
        R = 1.987
        Term1 = C5*sig**2*lib.exp(-Q4/(R*T))
        Term2 = C6*sig**5*lib.exp(-Q5/(R*T))
        val = Term1 + Term2
        return val

    def strain_rate_tertiary(T,sig,t):
        """
        Akiyama model for the tertiary thermal strain rate of HT-9 cladding

        Inputs:
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        t (float): Time [s]
        """
        lib = compat(T, sig)
        C7 = 2.12E7
        Q6 = 94233.3
        R = 1.987
        val = 4*sig**10 * (C7*lib.exp(-Q6/(R*T))*t)**3
        return val

    def strain_rate_thermal(T,sig,t):
        """
        Total thermal strain rate of HT-9 cladding, the sum of the primary,
        secondary and tertiary creep rates

        Inputs:
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        t (float): Time [s]
        """
        eps_1 = HT9.strain_rate_primary(T,sig,t)
        eps_2 = HT9.strain_rate_secondary(T,sig)
        eps_3 = HT9.strain_rate_tertiary(T,sig,t)
        val = eps_1 + eps_2 + eps_3
        return val

    def strain_rate_irrad(T,sig,flux):
        """
        Irradiation strain rate of HT-9 cladding

        Inputs:
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        flux (float): Fast neutron (>1 MeV) flux [n/cm^2-s]
        """
        lib = compat(T, sig)
        B0 = 1.83E-4
        A1 = 2.59E14
        Q_irr = 73000.0
        R = 1.987
        val = (B0 + A1*lib.exp(-Q_irr/(R*T)))*flux*sig**1.3 * 1E-22
        return val

    def strain_rate(T,sig,flux,t):
        """
        Total strain rate of HT-9 cladding, the sum of the thermal and
        irradiation strain rates

        Inputs:
        T (float): Temperature [K]
        sig (float): Effective stress [MPa]
        flux (float): Fast neutron (>1 MeV) flux [n/cm^2-s]
        t (float): Time [s]
        """
        eps_th = HT9.strain_rate_thermal(T,sig,t)
        eps_irr = HT9.strain_rate_irrad(T,sig,flux)
        val = eps_th + eps_irr
        return val

    def yield_stress(T):
        """
        Akiyama model for the yield stress of HT-9 cladding, the ultimate
        tensile stress is assumed equal to the yield stress

        Inputs:
        T (float): Temperature [K]
        """
        A1 = 1.290E9
        A2 = -3.561E6
        A3 = 6.371E3
        A4 = -3.959
        val = A1 + A2*T + A3*T**2 + A4*T**3
        return val


class D9_SS:
    def k(T):
        return
