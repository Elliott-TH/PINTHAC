import scipy
from Arr_Compat import compat



class f_water:
    def Blasius(self,Props,G,D):
        '''
        Blasius correlation for friction factor
        Re<1E5
        '''
        mu = Props['mu']
        Re = G * D / mu
        fval = 0.316*Re**(-0.25)
        return fval
    
    def McAdams(self,Props,G,D):
            '''
            McAdams correlation for friction factor
            30,000 < Re < 1,000,000
            '''
            mu = Props['mu']
            Re = G * D / mu
            fval = 0.184*Re**(-0.2)
            return fval


class f_SCW:
    def Filonenko(self,Props,G,D):
        """
        Correlation for friction factor of supercritical water
        """
        rho = Props['rho']
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        Nu = 0.026 * Re**(0.8) * Pr**(0.4)
        lib = compat(G, D)
        fval = 1/(1.82*lib.log10(Re)-1.64)**(2)
        return fval

    def Wu(self,Props,G,D):
        """
        Wu correlation for friction factor of supercritical water
        around rod bundles
        """
        rho = Props['rho']
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        f_iso = f_SCW.Filonenko(self,Props,G,D)
        fnew = 0.014*f_iso**(-0.12)*Pr**(-0.23)
        return fnew


class Spacer:
    def blah2():
        return

