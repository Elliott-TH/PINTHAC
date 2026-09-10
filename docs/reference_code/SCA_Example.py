import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import os
from tqdm import tqdm
import time
#Finds lookup table path and input file path
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'Project_Prop.csv') #Property Table name
inputs = os.path.join(script_dir, 'Inputs.xlsx') #Input Excel name
df = pd.read_csv(csv_file) #Imports property table
df.columns = df.columns.str.strip() # Remove leading/trailing whitespace
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') #convert values to numeric floats

def Property(Prop, prop):# Property Lookup function
    f = interp1d(df[Prop[0]], df[prop], kind='linear', fill_value="extrapolate") #Utilize Scipy interpolation
    val = f(Prop[1])
    return val

#Property lookup fcn takes in an array and a string. The array is first filled with a string
#that says what the input prop is, followed by its value. The string input is to define the property you want to look up.
#ex: given a T of 500 you want to find h_f: h_f = Property(['T',500],'h_f')

dfInput = pd.read_excel(inputs) #Imports input data table
dfInput.columns = dfInput.columns.str.strip() # Remove leading/trailing whitespace
N = dfInput.shape[0] #Checks how many cases inside input table to run

print('=============================================================')
print(f'         {N} datasets detected, calculating results')
print('=============================================================')

#Loop to iterate through each input set given in the excel
for j in tqdm(range(0,N)): 
    print('')
    #Extracts input data
    LWR,L,Le,D,Pitch,Tm_in, mdot, Pnom, q0, Dci, Dfo, SCBflag = dfInput[
        ['Type','L', 'Le','D', 'Pitch', 'Tm_in', 'mdot', 'Pnom', 'q0', 'Dci', 'Dfo', 'SCB']
    ].iloc[j]

    #Other Constants
    A = np.pi * D**2 /4 #Rod area
    Ah = Pitch**2 - np.pi/4 * D**2 # Wetted Area
    G = mdot/Ah #Mass Flux
    Per_w = np.pi * D #Wetted Perimeter
    Dh = 4*Ah/Per_w #Hydraulic Diameter
    psi = 1.826*(Pitch/D) - 1.0430 #Weissman 
    Rco = D/2 #Clad outer radius
    Rfo = Dfo/2 #Fuel outer radius
    Rci=Dci/2 #Clad inner radius
    Rg = 1/2 * (Rfo + Rci) #Gap radius
    Dg = 1/2 * (Dfo + Dci) #Gap diameter
    kc=15 #Cladding Conductivity
    stephanboltz = 5.67E-8 #Stephan Boltzman Constant
    Tinit = ['T',Tm_in] # Inlet Temp
    delta = (Dci-Dfo)/2 #Gas gap
    hf_sat = Property(['P_sat',Pnom],'h_f') #hf_sat
    hg_sat = Property(['P_sat',Pnom],'h_g')#hg_sat
    Tsat = Property(['P_sat',Pnom],'T') #Tsat
    dz = L/400 #Cell length
    Z = np.arange(-L/2+dz,L/2+dz,dz) #Cell coords
    cells = np.arange(1,401,1) #Cell numbers
    hin = Property(Tinit,'h_f') #Inlet enthalpy
    Zmax = Le/2 #Max Z
    Zmin = -Le/2 # Minimum Z
    g = 9.81  #Gravitational Accel
    Cheng=0.1339+0.09059*((Pitch/D)-1)-0.09926*((Pitch/D)-1)**2 #Cheng Correlation

    def htcLo(Tsat,G,D): #Computes the liquid only heat transfer coeff
            T = ['T', Tsat]
            mu_l = Property(T,'mu_f')
            k_l = Property(T,'k_f')
            Pr = Property(T,'Pr_f')
            Re_lo = G*D/mu_l
            htc_lo = 0.023 * Re_lo**(0.8) * Pr**(0.333) * k_l/D
            return htc_lo

    def Bowring(P,G,D,x1,T1): #Bowring Correlation
            Tin = ['T',T1]
            h_fg = Property(Tin, 'h_g')-Property(Tin,'h_f')
            p_r = 0.145 * P/1000000
            n = 2-0.5 * p_r
            if np.isclose(p_r,1,rtol=0.000001):
                F1=F2=F3=F4=1
            elif p_r<1:
                c1, c2, c3, c4 = 18.942, 1.316, 17.023, 1.649
                F1 = 1/(1.917) * (p_r**c1 * np.exp(20.89*(1-p_r)) + 0.917)
                F2 = 1.309 * F1 * (p_r**(1.316) * np.exp(2.444*(1-p_r)) + 0.309)**(-1)
                F3 = (1/1.667) * (p_r**c3 * np.exp(16.658*(1-p_r)) + 0.667)
                F4 = F3 * p_r**c4
            else:
                c1, c2, c3, c4 = -0.368, -0.448, 0.219, 1.649
                F1 = p_r**(c1) * np.exp(0.648 * (1-p_r))
                F2 = F1 * (p_r**(c2) * np.exp(0.245*(1-p_r)))**(-1)
                F3 = p_r**(c3)
                F4 = F3 * p_r**(c4)
            Anum = 2.317 * h_fg*D*G * 0.25 * F1
            Adenom =(1+0.0143*F2*G*np.sqrt(D))
            A = Anum/Adenom
            B = G*D/4
            C = 0.077*F3*G*D/(1+0.347*F4*(G/1356)**n)
            qpp = (A-B*h_fg*x1)/C
            return qpp

    def htc2phi(Tsat,q_pp,x,G,D): #Computes Schrock & Grossman correlation
            a1, a2, b = 7400, 1.11, 0.66
            T = ['T', Tsat]
            h_fg = Property(T,'h_g')-Property(T,'h_f')
            mu_g = Property(T,'mu_g')
            mu_l = Property(T,'mu_f')
            vol_v = Property(T,'vol_g')
            vol_f = Property(T,'vol_f')
            Pr = Property(T,'Pr_f')
            k_l = Property(T,'k_f')
            Re_lo = G*D/mu_l
            htc_lo = 0.023 * Re_lo**(0.8) * Pr**(0.333) * k_l/D
            Xtt = ((1-x)/x)**0.9 * (vol_f/vol_v)**0.5 * (mu_l/mu_g)**0.1
            htc_2phi = htc_lo * (a1*q_pp/(G*h_fg) + a2 * Xtt**(-b))
            return htc_2phi

    def d_P(G,D,P,x0,x1,dz,T,SCB,Boil): #Pressure Loss
        T0 = ['T',T]
        mu_f = Property(T0,'mu_f')
        vol_f = Property(T0,'vol_f')
        vol_g = Property(T0,'vol_g')
        vol_fg = vol_g - vol_f
        Re = G*D/mu_f
        f_lo =Cheng * Re**(-0.18)
        rho_m = 1/(x1*vol_g + (1-x1)*vol_f)
        if SCB or Boil == True:
            dP = f_lo*dz*G**2/(2*rho_m*D) + G**2 *(x1-x0)*vol_fg + rho_m*g*dz
        else:
            dP = f_lo*vol_f*dz*G**2/(2*D) + g*dz/vol_f + G**2 *(x1-x0)*vol_fg 
        return dP

    def q_p(z): # Linear Heat Generation Rate
        return q0 * np.cos(np.pi * z/Le)

    #Iterative solver for T_fo
    def T_fo(T_ci,z):
        htcg_g=5000 #htc guess
        err = 1
        Tfo_new=T_ci+ q_p(z)/(2*np.pi*Rg * htcg_g) #Tfo guess
        Tfo_old = 0 #Initialize Tfo_old
        while err>=0.001:
            Tm = (Tfo_new + T_ci)/2
            T_cik = T_ci + 273.15
            T_fok = Tfo_new + 273.15
            kgas = (15.8)*10**(-4) * (Tm+273.15)**(0.79)
            htcg_g = kgas/delta + stephanboltz * (T_fok**3 + T_fok**2 * T_cik + T_cik**3 + T_cik**2 * T_fok)
            Tfo_new = T_ci+ q_p(z)/(np.pi*Dg* htcg_g)
            err = np.absolute(Tfo_new-Tfo_old)
            Tfo_old = Tfo_new
        return Tfo_new

    #Iterative solver for Tmax
    def Tmax(T_fo,z):
        kbar=3 #k guess
        err = 1 
        c1, c2, c3 = 3824, 402.4, 6.1256E-11
        T_maxnew = T_fo + q_p(z)/(4*np.pi*kbar) #Tmax guess
        while err>=0.0001:
            kdTmax = c1*np.log(c2+T_fo)+ c3/4 * (T_fo+273)**4 + q_p(z)/(4*np.pi)
            kdTcheck = c1*np.log(c2+T_maxnew)+ c3/4 * (T_maxnew+273)**4
            T_maxnew = T_maxnew + (kdTmax - kdTcheck)/100
            err = np.abs(kdTmax - kdTcheck)
        return T_maxnew

    #Computes the quality
    def quality(h_i,i_zd,Boil,SCB):
        xe_i = (h_i-hf_sat)/(hg_sat-hf_sat) 
        if SCB: #If SCB, calculate x using eqn 75 in module 60
            hscb = h[i_zd]
            xe_zd = (hscb-hf_sat)/(hg_sat-hf_sat)
            x_i = xe_i - xe_zd * np.exp(xe_i/xe_zd -1)
        elif Boil and not SCB: #If boiling w/o SCB, calculate x as xe
            x_i = xe_i
        else: #If no boiling and no SCB, xe = x = 0
            xe_i=0 
            x_i=0
        return x_i,xe_i

    #Setting Initial flag and index vals
    SCB = False
    i_SCB = None
    i_B = None
    Boil = False

    #Setting SCB toggle and CHFR limits based on input file
    if SCBflag == 'On' or SCBflag == 'on' or SCBflag == 'ON':
        SCBtoggle = True
    else: SCBtoggle = False

    if LWR == 'BWR':
        chfrlim=1.9
    else:
        chfrlim = 1.3
    chfrvio = False

    #Defining all values at step zero
    h0 = hin + q_p(-L/2 + dz/2) * dz/mdot
    T0 = Property(['h_f',h0],'T') 
    htc_in = psi * htcLo(T0,G,Dh)
    Tco_0 = T0 + q_p(Z[0]) / (np.pi * D * htc_in)
    Tci_0 = Tco_0 + q_p(Z[0]) / (2 * np.pi * kc) * np.log(Rco/Rci)
    Tfo_0 = T_fo(Tci_0,Z[0])
    Tmax_0 = Tmax(Tfo_0,Z[0])
    x_0, xe_0 = quality(h0,0,Boil,SCB)
    dP_0 = d_P(G,Dh,Pnom,0,0,dz,T0, False, False) 
    qpp_0 = q_p(-L/2 + dz/2)/(np.pi * D)
    chfr_0 = 0

    # Initializing data arrays with their zero'th value
    h=[h0]
    T=[T0]
    h=[h0]
    htc = [htc_in]
    T_co=[Tco_0]
    T_ci = [Tci_0]
    Tfo = [Tfo_0]
    T_max = [Tmax_0]
    x=[x_0]
    Zd=[]
    xe =[xe_0]
    dP = [dP_0]
    CHFR = [chfr_0]

    # Channel Iteration Loop
    for i in range(1,400):

        zi = Z[i] # i-th z value
        h_i = h[i-1] + q_p(zi - dz/2) * dz / (mdot)  #Enthalpy at i-th z value from i-1th val
        h.append(h_i)  #Add i-th enthalpy value

        qp = q_p(zi)  #LHGR in current cell
        q_pp = q_p(zi)/(np.pi*D)  #Heat flux in current cell

        if Property(['h_f',h_i], 'T')>=Tsat:  #Checking if enthalpy is at saturation
            Tm_i = Tsat #If saturated, Tm_i is T_sat
        else:
            Tm_i = Property(['h_f',h_i], 'T')  #If not Tm_i is computed from table and i-th h val
        T.append(Tm_i) #Add i-th T_m value list

        #Boiling Flag
        for i, T_i in enumerate(T): 
            if not Boil and T_i >= Tsat:
                Boil = True
                i_B = i
        
        x_i, xe_i = quality(h[i],i_SCB, Boil, SCB)  #Get Qualities
        x.append(x_i)
        xe.append(xe_i)

        dP_i = d_P(G,Dh,Pnom,x[i-1],x_i,dz,Tm_i,SCB, Boil)  #Compute ith pressure drop
        dP.append(dP_i)  #Add ith pressure drop to list
        
        if Boil or SCB:  #If scb or Boil, set htc to 2 phase
            htc_i = htc2phi(Tm_i,q_pp, x_i, G, Dh) * psi
        else:
            htc_i = htcLo(Tm_i,G,Dh) * psi

        Tco_i = Tm_i + qp/(np.pi * D * htc_i)  #Compute T_co w/ new htc
        T_co.append(Tco_i)

        Tci_i = Tco_i + q_p(zi) / (2 * np.pi * kc) * np.log(Rco/Rci)  #Compute T_ci
        T_ci.append(Tci_i)

        Tfo_i = T_fo(Tci_i, zi)  #Get T_fo 
        Tfo.append(Tfo_i)

        Tmax_i = Tmax(Tfo_i,zi)  #Get T_max
        T_max.append(Tmax_i)

        #CHFR calculation
        if SCB or Boil == True: #If SCB or Boil, use Bowring
            chfr_i = (Bowring(Pnom,G,Dh,x_i,Tm_i)*psi)/q_pp
            CHFR.append(chfr_i)
        else: #If not SCB or Boil, CHFR=0
            chfr_i = 0
            CHFR.append(chfr_i)

        for i, Tco_i in enumerate(T_co): #Setting SCB Flag
            if not SCB and Tco_i > Tsat and SCBtoggle==True:
                SCB = True 
                i_SCB = i

    #CHFR Violation Check, alerts user if violation detected
    if (Boil or SCB) and any((chfr < chfrlim and chfr >0) for chfr in CHFR): #Checks CHFR list for non-zero CHFR less than limit
        print("+-------------------------------------------------------+")
        print("|    ALERT: CHFR limit violated in one or more cells    |")
        print("+-------------------------------------------------------+")
    print('')

    # Rounding Values:
    T = np.round(T,6)
    T_co = np.round(T_co,6)
    T_ci = np.round(T_ci,6)
    Tfo = np.round(Tfo,6)
    T_max = np.round(T_max,6)
    CHFR = np.round(CHFR,9)
    dP = np.round(dP,6)

    #Saving Outputs
    filename = os.path.join(script_dir,f'Final{LWR}_SCB{SCBflag}.csv') #Filename
    #print(f'Saving outputs to {filename}') #Printing the path file is saved to
    df3 = pd.DataFrame({
        "Cell":cells,
        "z": Z,
        "Tm": T,
        "Tco": T_co,
        "Tci": T_ci,
        "Tfo": Tfo,
        "Tmax": T_max,
        "x": x,
        "xe": xe,
        "CHFR": CHFR,
        "dP":dP
    })
    df3.to_csv(filename, sep='\t', index=False, header=True) #Convert to xlsx
    #print('Outputs Succesfully Saved') #Indicate user file sucessfully saved

    
