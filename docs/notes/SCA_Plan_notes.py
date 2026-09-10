import numpy as np
import torch
import matplitlib.pyplot


'''
SCA Parameters:

Coolant: Water, SCW, Lead, Lead-Bi, Sodium
Length: L
Number of cells: N
Geometry: Cylinder or Annular
    - Make subroutine for handling annular geometry
Cladding material:
Fuel Geom: rco,rci,delta,rfo, etc
Flow Rate: Either mdot or G
Power Profile: qp(z)
Peak power: qp_max

Correlations and model parameters:

HTC model
Friction model
Fuel thermal conductivity model
cladding thermal conductivity model
iterate cladding conductivity flag

For water, we need to describe how much of
the two phase physics needs to be specified,
I dont want to implement a routine to handle 
the semantics of two phase rn, so ill ignore it.

So when the coolant is specified by the user:
-The property library is selected and used
To do this the easiest way, the notation
about LMProps and IAPWS need to be unified. 

-- Make a module call Prop_Routine that 
does the bulk of the unification using 
a standard property function that returns
the Prop dictionary used about the 
library.
'''