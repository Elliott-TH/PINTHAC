import numpy as np
import torch 
import scipy



class Zircalloy:
    def k(T):
        return


class D9_SS:
    def k(T):
        return


class UO2:
    def k_NFI(T,Bu,f_gad):
        A = 0.0452, B = 2.46E-4,
        C = 5.47E-9, D = 2.29E14 
        E = 3.5E9, F = 16361
        a = 1.1599, Q = 6380

        f_Bu = 0.00187 * Bu
        g_Bu = 0.038 * Bu**(0.28)
        h_T = (1+396*np.exp(-Q/T))

        denom_1 = A + a*f_gad + B*T + f_Bu 
        denom_2 = (1-0.9*np.exp(-0.04*Bu))*g_Bu*h_T
        Term1 = 1/(denom_1+denom_2)
        Term2 = E/(T**2) * np.exp(-F/T)
        K = Term1 + Term2
        return K

    def eps(T):
        """
        Correlation for the emmisivity of UO2 fuel as it 
        appears in MATPRO, error of +- 6.8%
        """
        a = 0.7856, b=1.5263E-5
        val = a + b*T
        return T

    def thrm_exp(T):
        K1 = 9.8E-6
        K2 = 2.61E-3
        K3 = 3.16E-1
        Ed = 1.32E-19
        k=1.38E-23
        strain = K1*T - K2 + K3*np.exp(Ed/(k*T))
        return strain


    




