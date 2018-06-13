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
from scipy.integrate import solve_ivp
import cantera as ct
import matplotlib.pyplot as plt

""" Set up inputs and givens """
"-----------------------------------------------------------------------------"
# Geometry and physical constants:
L_pore = 5.0      # Diameter of pore in microns (or side length of square pore)
spacing = 10.0    # Horizontal spacing (center to center) of pores in microns
t_glass = 20.0    # Thickness of spin on glass layer in microns
Temp = 25.0       # Constant gas temperature [C]
Press = 101325.0  # Initial gas pressure [Pa]
phi_g = 0.5       # Porosity of glass layer
t_sim = 10.0      # Simulation time [s]

tau_g = phi_g**(-0.5)   # Tortuosity calculation via Bruggeman correlation

# Boundary conditions:
# Species in Simple Air: [O2', 'N2', 'AR']
# Constant concentrations of air will be supplied from ambient at inlet
# Constant flux of each species O2 will be calculated based on current
i_curr = 0.15     # [A/cm^2]  

# Switch inputs:
Diff_Mod = 0      # Diffusion model: 0 - Fick's, 1 - Dusty Gas
Geom = 1          # Lattice type: 1 - Square, 2 - Hexagonal

# Discretization for 2D mesh:
Nx = 5
Ny = 3

# Call cantera for gas-phase:
if Diff_Mod == 0:
    gas = ct.Solution('simple_air.cti')
elif Diff_Mod == 1:
    gas = ct.DustyGas('simple_air.cti')
    gas.porosity = phi_g
    gas.tortuosity = tau_g
    gas.mean_pore_radius = 1.5e-7 # length in [m]
    gas.mean_particle_diameter = 1.5e-6 # length in [m]
# 'simple_air.cti' was created from 'air.cti' by removing reactions and all 
# species except O2, N2, and Ar.
    
# Species name for contour plots:
# Species in Simple Air: ['O2', 'N2', 'AR']
plt_species = 'O2'

# Initialize solution vector:
Nspecies = gas.n_species
SV_0 = np.zeros(Nx*Ny*Nspecies)

# Given constants:
F = 96485333           # Faraday Constant [s-A/kmol]
R = ct.gas_constant    # Universal gas constant [J/kmol-K]

""" Pre-process variables/expressions """
"-----------------------------------------------------------------------------"
gas.TP = Temp+273.15, Press    # set gas state via temperature and pressure

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

# Set up initial values at each cell with ambient simple_air
SV_0 = np.tile(gas.Y*gas.density_mass/phi_g,Nx*Ny)
inlet_BC = np.tile(gas.Y*gas.density_mass/phi_g,BC_in)

# Solve for O2 flux at outlet BC based on i_curr and Faraday
iO2 = gas.species_index('O2')
MW_O2 = gas.molecular_weights[iO2]
J_O2_out = MW_O2 *i_curr *100**2 /F /4
J_BC = np.zeros(Nx*Nspecies)
J_BC[iO2::Nspecies] = J_O2_out
    
""" Sub-functions for integrator """
"-----------------------------------------------------------------------------"
if Diff_Mod == 0:
    from Ficks_func import Flux_Calc
elif Diff_Mod == 1:
    from DGM_func import Flux_Calc
    
def dSVdt_func(t,SV): # System of ODEs to solve
    dSVdt = np.zeros(Nx*Ny*Nspecies)
    Fluxes_X, Fluxes_Y = Flux_Calc(SV,Nx,dX,Ny,dY,Nspecies,inlet_BC,gas,phi_g,tau_g)
    Fluxes_Y[Ny*Nx*Nspecies:] = J_BC # Constant flux out BC
               
    # Initialize the fluxes into the first row (y-direction)
    Flux_Y_in = Fluxes_Y[0:Nx*Nspecies]
            
    # Vector math for inside domain all but last row of cells
    for j in range(Ny):
        ind1 = j*Nx*Nspecies # index for first cell in each row
        ind2 = (j+1)*Nx*Nspecies # index for last cell in each row

        Flux_Y_out = Fluxes_Y[(j+1)*Nx*Nspecies:(j+2)*Nx*Nspecies]
        
        Flux_X_in = Fluxes_X[j*(Nx+1)*Nspecies:(j+1)*(Nx+1)*Nspecies-Nspecies]
        Flux_X_out = Fluxes_X[j*(Nx+1)*Nspecies+Nspecies:(j+1)*(Nx+1)*Nspecies]
        
        dSVdt[ind1:ind2] = 1/phi_g*((Flux_Y_in + Flux_Y_out)/dY \
                         + (Flux_X_in + Flux_X_out)/dX)
        
        # The fluxes leaving the current row are the inlets to the next row
        Flux_Y_in = Flux_Y_out
        
    return dSVdt

""" Call ODE integrator and process results """
"-----------------------------------------------------------------------------"
sol = solve_ivp(dSVdt_func, (0, t_sim), SV_0) # Use ODE solver

plt_species_ind = gas.species_index(plt_species) # Extract index for plot

# Define variable vectors to be plotted
x_plt = np.linspace(0, rad+max_space, Nx) / 1e-6
y_plt = -1*np.linspace(0, t_glass, Ny)
SV = sol.y.T
sol_t = sol.t
SV_plt = np.reshape(SV[-1,plt_species_ind::Nspecies], (Ny,Nx))

# Create contour plot
plt.contourf(x_plt,y_plt,SV_plt)
plt.title('%s Density [kg/m^3 gas]' %plt_species)
plt.xlabel('Horizontal distance, x [micron]')
plt.ylabel('Glass thickness, y [micron]')
plt.colorbar()




















""" Comments and future steps """
"-----------------------------------------------------------------------------"
# Integrate calls for diffusion coefficients and properly incorporate into the
# ODE function. 

# Determine appropriate discretization in time and space and apply for a time
# that allows for a steady-state solution to be reached.

# Re-define flux parameters to be calculated from DGM.

# Exercise model for different geometries. 