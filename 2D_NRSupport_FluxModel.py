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
import matplotlib.pyplot as plt

""" Set up inputs and givens """
"-----------------------------------------------------------------------------"
# Geometry and physical constants:
L_pore = 5.0    # Diameter of pore in microns (or side length of square pore)
spacing = 10.0  # Horizontal spacing (center to center) of pores in microns
t_glass = 20.0  # Thickness of spin on glass layer in microns
phi_g = 0.5     # Constant porosity of spin on glass
Temp = 25       # Constant temperature [C]
t_sim = 30.0    # Simulation time [s]

# Call cantera for gas-phase:
gas = ct.Solution('Air.cti')

# Number of species to track:
Nspecies = gas.n_species

# Boundary conditions:
# Species in Air: ['O', 'O2', 'N', 'NO', 'NO2', 'N2O', 'N2', 'AR']
# Constant mass density of each species at top inlet
m_k_top = np.array([0.,1.,2.,3.,4.,5.,6.,7.])
# Constant flux of each species at bottom outlet  
J_k_bot = np.array([7.,6.,5.,4.,3.,2.,1.,0.])  

# Switch inputs:
Diff_mod = 1    # Diffusion model: 1 - Fick's, 2 - Knudsen, 3 - Dusty Gas
Geom = 1        # Lattice type: 1 - Square, 2 - Hexagonal

# Discretization for 2D mesh and time:
Nx = 5
Ny = 3
Nt = 2

# Initialize solution vector:
SV_0 = np.zeros(Nx*Ny*Nspecies)

# Species name for contour plots
# Species in Air: ['O', 'O2', 'N', 'NO', 'NO2', 'N2O', 'N2', 'AR']
plt_species = 'O2'

""" Pre-process variables/expressions """
"-----------------------------------------------------------------------------"
rad = 1e-6*L_pore/2.0   # length of top inlet [m]

# Determine half of max solid distance between pores for domain [m]
if Geom == 1: # square lattice
    max_space = 1e-6*np.sqrt(2)*(spacing - L_pore) / 2.0
elif Geom == 2: # hexagonal lattice
    max_space = 1e-6*(spacing - L_pore) / 2.0

BC_in = int(round(Nx*rad / (rad+max_space))) # x-nodes affected by inlet BC

# Define differential distances
dX = (rad + max_space) / Nx
dY = 1e-6*t_glass / Ny

# Set up initial values at inlet
for i in range(BC_in):
    SV_0[i*Nspecies:(i+1)*Nspecies] = m_k_top
    
# Create vectors from m_k_top and J_k_bot for sub-functions
m_k_vec = np.zeros(Nx*Nspecies)
m_k_vec[0:BC_in*Nspecies] = np.tile(m_k_top, BC_in)
J_k_vec = np.tile(J_k_bot, (Nx,1))
    
""" Sub-functions for integrator """
"-----------------------------------------------------------------------------"
def J(Diff_mod,SV): # Diffusion model for flux calculations
    if Diff_mod == 1:
        # Fick's Diffusion model - negligible interactions with solid
        D_AB = 1.
        J = -D_AB*1.
    elif Diff_mod == 2:
        # Knudsen's Diffusion model - solid/gas interactions dominate
        D_AB = 1.
    elif Diff_mod == 3:
        # Dusty Gas model - solid/gas interactions both contribute
        D_AB = 1.
    else:
        print('Error: Diff_mod must be an integer between 1 and 3.')
        
    return J

def dJdt_func(SV,t): # System of ODEs to solve
    dJdt = np.zeros(Nx*Ny*Nspecies)
    Fluxes_X = np.zeros([Ny,Nx-1,Nspecies])
    Fluxes_Y = np.zeros([Ny+1,Nx,Nspecies])
    Fluxes_Y[-1,:,:] = J_k_vec # Set constant flux out BC for each species
       
    # Initialize at BC (1st row)
    ind1 = 0 # First cell of first row
    ind2 = Nspecies # Second cell of first row
    ind3 = (Nx-1)*Nspecies # Second-to-last cell of first row
    ind4 = Nx*Nspecies # Last cell of first row
    
    dJdt[ind1:ind2] = (Fluxes_Y[0,0,:] - Fluxes_Y[1,0,:])/dY \
                    - Fluxes_X[0,0,:]/dX
    dJdt[ind2:ind3] = np.reshape((Fluxes_Y[0,1:-1,:] - Fluxes_Y[1,1:-1,:])/dY \
                    + (Fluxes_X[0,0:-1,:] - Fluxes_X[0,1:,:])/dX, ind3-ind2)
    dJdt[ind3:ind4] = 0.
    
    # Vector math for inside domain
    for j in range(1,Ny+1):
        ind1 = j*Nx*Nspecies # First cell of each row
        ind2 = j*Nx*Nspecies + Nspecies # Second cell of each row
        ind3 = (j+1)*Nx*Nspecies - Nspecies # Second-to-last cell of each row
        ind4 = (j+1)*Nx*Nspecies # Last cell of each row
        
        dJdt[ind1:ind2] = 0.
        dJdt[ind2:ind3] = 0.
        dJdt[ind3:ind4] = 0.
        
    # Last row BC differentials
    ind1 = (Ny-1)*Nx*Nspecies # First cell of last row
    ind2 = (Ny-1)*Nx*Nspecies + Nspecies # Second cell of last row
    ind3 = Ny*Nx*Nspecies - Nspecies # Second-to-last cell of last row
    ind4 = Ny*Nx*Nspecies # Last cell of last row
    
    dJdt[ind1:ind2] = -1.
    dJdt[ind2:ind3] = -1.
    dJdt[ind3:ind4] = -1.

    return dJdt

""" Call ODE integrator and process results """
"-----------------------------------------------------------------------------"
res = odeint(dJdt_func, SV_0, np.linspace(0, t_sim, Nt)) # Use ODE solver

plt_species_ind = gas.species_index(plt_species) # Extract index for species

# Define variable vectors to be plotted
x_plt = np.linspace(0, rad+max_space, Nx) / 1e-6
y_plt = -1*np.linspace(0, t_glass, Ny)
SV_plt = np.reshape(res[-1,plt_species_ind::Nspecies], (Ny,Nx))

# Create contour plot
plt.contourf(x_plt,y_plt,SV_plt)
plt.colorbar()
plt.title('2D %s Plot' %plt_species)
plt.xlabel('Horizontal distance, x [micron]')
plt.ylabel('Glass thickness, y [micron]')





















""" Comments and future steps """
"-----------------------------------------------------------------------------"
# Integrate calls for diffusion coefficients and properly incorporate into the
# ODE function. 

# Apply appropriate values for inlet and outlet boundary conditions.

# Determine appropriate discretization in time and space and apply for a time
# that allows for a steady-state solution to be reached.

# Re-define flux parameters to be calculated from DGM.

# Exercise model for different geometries. 