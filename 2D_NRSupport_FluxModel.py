"""
Author:
    Corey R. Randall (17 May 2018)

2D Diffusion Model:
    This model examines the diffusion taking place in a layer of spin on glass
    atop a porous silicon wafer. Depending on the pore size and density, there
    should be an ideal glass thickness such that the variation in concentration
    is small after diffusing through the glass.
    
    | inlet  |_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_|
    |________|_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_| #constant c_k top
    |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
    |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
    |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| #symmetry - left/right
    |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
    |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
    |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| #constant flux bot
        
"""

""" Load any needed modeules """
"-----------------------------------------------------------------------------"
import numpy as np
from scipy.integrate import odeint
import cantera as ct
# import cantera.DustyGas as DGM
# import matplotlib.pyplot as plt

""" Set up inputs and givens """
"-----------------------------------------------------------------------------"
# Geometry and physical constants:
L_pore = 5.0    # Diameter of pore in microns (or side length of square pore)
spacing = 10.0  # Horizontal spacing (edge to edge) of pores in microns
t_glass = 20.0  # Thickness of spin on glass layer in microns
phi_g = 0.5     # Constant porosity of spin on glass
Temp = 25       # Constant temperature [C]
t_sim = 30.0    # Simulation time [s]

# Call cantera for gas-phase:
gas = ct.Solution('Air.cti')

# Number of species to track:
Species = gas.n_species

# Boundary conditions:
C_k_top = np.array([1,0])   # Constant c_k of each species at top inlet
J_k_bot = np.array([1,0])   # Constant J_k of each species at bottom outlet

# Switch inputs:
Diff_mod = 1    # Diffusion model: 1 - Fick's, 2 - Knudsen, 3 - Dusty Gas
Geom = 1        # Lattice type: 1 - Square, 2 - Hexagonal

# Discretization for 2D mesh and time:
Nx = 5
Ny = 5
Nt = 1000

# Initialize solution vector:
SV_0 = np.zeros([1,Nx*Ny*Species])

""" Pre-process varialbes/expressions """
"-----------------------------------------------------------------------------"
rad = 1e-6*L_pore/2.0   # length of top inlet [m]

""" Determine half of max distance between pores for domain """
if Geom == 1: # square lattice
    max_space = np.sqrt(2)*spacing / 2.0
elif Geom == 2: # hexagonal lattice
    max_space = spacing / 2.0

BC_in = Nx*round(rad / (rad+max_space)) # Number x-nodes affected by inlet BC

""" Begin solution/integrator """
"-----------------------------------------------------------------------------"
def Diff_coeff(Diff_mod):
    if Diff_mod == 1:
        # Fick's Diffusion model - negligible interactions with solid
        D_AB = 1
    elif Diff_mod == 2:
        # Knudsen's Diffusion model - solid/gas interactions dominate
        D_AB = 1
    elif Diff_mod == 3:
        # Dusty Gas model - solid/gas interactions both contribute
        D_AB = 1
    else:
        print('Error: Diff_mod must be an integer between 1 and 3.')
        
    return D_AB

def dSVdt_func():
    for j in Ny:
        for i in Nx:
            SV = 1.
            # Vector math here for dSVdt

    return SV

res = odeint(dSVdt_func, SV_0, np.linspace(0, t_sim, Nt))






















