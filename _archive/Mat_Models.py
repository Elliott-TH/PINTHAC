import scipy
from Arr_Compat import compat

class Zircalloy:
    def k(T):
        return


class D9_SS:
    def k(T):
        return

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
        A = 0.0452, B = 2.46E-4,
        C = 5.47E-9, D = 2.29E14 
        E = 3.5E9, F = 16361
        a = 1.1599, Q = 6380

        lib = compat(T, Bu)
        f_Bu = 0.00187 * Bu
        g_Bu = 0.038 * Bu**(0.28)
        h_T = (1+396*lib.exp(-Q/T))

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
        """
        a = 0.7856, b=1.5263E-5
        val = a + b*T
        return val

    def thrm_expan(T):
        """
        Frapcon-4 model for thermal conductivity of solid UO2 fuel
        """
        K1 = 9.8E-6
        K2 = 2.61E-3
        K3 = 3.16E-1
        Ed = 1.32E-19
        k=1.38E-23
        lib = compat(T)
        strain = K1*T - K2 + K3*lib.exp(Ed/(k*T))
        return strain


    




