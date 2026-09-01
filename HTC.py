import numpy as np
import torch
import scipy
import array_api_compat as api




class HTC:
    def __init__(self):
        self.err = None
        self.value = None
            
    def Dittus(self,Props,G,D):
        rho = Props['rho']
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        Nu = 0.026 * Re**(0.8) * Pr**(0.4)
        val = Nu * k / D

        self.err = [0.25,0.45]

        return val

    def Petchukov():
        return

    def Gnielinski(self,Props,G,D):
        return

    def SchrockGrossman():
        return

    def Chen_H2O_dT():
        return

    def Chen_H2O():
        return

    def Bjorge_dT():
        return

    def Bjorge():
        return

    def Swenson_dT(self,Props_b,Props_w,Tw,Tb,G,D):
        rho_b = Props_b['rho']
        mu_b = Props_b['mu']
        k_b = Props_b['k']
        cp_b = Props_b['cp']
        h_b = Props_b['h']
        Pr_b = mu_b * cp_b / k_b
        Re_b = G * D / mu_w

        rho_w = Props_w['rho']
        mu_w = Props_w['mu']
        k_w = Props_w['k']
        cp_w = Props_w['cp']
        h_w = Props_w['h']
        Pr_w = mu_w * cp_w / k_w
        Re_w = G * D / mu_w

        cp_bar = (h_w-h_b)/(Tw-Tb)
        c1 = 0.00459
        c_Re = 0.92
        c_Pr = 0.61
        c_cp = 0.61
        c_rho = 0.23

        R_rho = rho_w/rho_b
        R_cp = cp_bar/cp_b

        Nu = c1 * Re_w**(c_Re) * Pr_w**(c_Pr) * R_cp**(c_cp) * R_rho**(c_rho)
        htc = Nu * k_w / D
        return htc

    def Swenson():
        return 

    def Chen_SCW_dT(self,Props_b,Props_w,Tw,Tb,G,D):
        rho_b = Props_b['rho']
        mu_b = Props_b['mu']
        k_b = Props_b['k']
        cp_b = Props_b['cp']
        h_b = Props_b['h']
        nu_b = mu_b/rho_b
        Pr_b = mu_b * cp_b / k_b
        Re_b = G * D / mu_w

        rho_w = Props_w['rho']
        mu_w = Props_w['mu']
        k_w = Props_w['k']
        cp_w = Props_w['cp']
        h_w = Props_w['h']
        nu_w = mu_w/rho_w
        Pr_w = mu_w * cp_w / k_w
        Re_w = G * D / mu_w

        cp_bar = (h_w-h_b)/(Tw-Tb)
        c1 = 0.00459
        c_Re = 0.92
        c_Pr = 0.61
        c_cp = 0.61
        c_rho = 0.23

        R_rho = rho_w/rho_b
        R_cp = cp_bar/cp_b

        Nu = c1 * Re_w**(c_Re) * Pr_w**(c_Pr) * R_cp**(c_cp) * R_rho**(c_rho)
        htc = Nu * k_w / D
        return

    def Chen_SCW():
        return