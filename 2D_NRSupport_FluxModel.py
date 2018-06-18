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
L_pore = 4.0      # Diameter of pore in microns (or side length of square pore)
spacing = 10.0    # Horizontal spacing (center to center) of pores in microns
t_glass = 10.0    # Thickness of spin on glass layer in microns
Temp = 25.0       # Constant gas temperature [C]
Press = 101325.0  # Initial gas pressure [Pa]
phi_g = 0.5       # Porosity of glass layer
t_sim = 20.0      # Simulation time [s]
d_p = 1.5e-6      # Mean particle diameter [m]

tau_g = phi_g**(-0.5)   # Tortuosity calculation via Bruggeman correlation

# Boundary conditions:
# Species in Simple Air: [O2', 'N2', 'AR']
# Constant concentrations of air will be supplied from ambient at inlet
# Constant flux of each species O2 will be calculated based on current
i_curr = 0.0000000001     # [A/cm^2]  

# Switch inputs:
Diff_Mod = 0      # Diffusion model: 0 - Fick's, 1 - Dusty Gas
Geom = 1          # Lattice type: 1 - Square, 2 - Hexagonal
i_stiff = 1       # System stiff: 0 - Not stiff, 1 - Is stiff
movie = 0         # Save movie of transient solution: 0 - Off, 1 - On

# Discretization for 2D mesh:
Nx = 3
Ny = 3

# Call cantera for gas-phase:
gas_file = 'simple_air.cti'

if Diff_Mod == 0:
    gas = ct.Solution(gas_file)
elif Diff_Mod == 1:
    gas = ct.DustyGas(gas_file)
    gas.porosity = phi_g
    gas.tortuosity = tau_g
    gas.mean_pore_radius = L_pore/2 *1e-6
    gas.mean_particle_diameter = d_p
# 'simple_air.cti' was created from 'air.cti' by removing reactions and all 
# species except O2, N2, and Ar.
    
# Species name for contour plots:
# Species in Simple Air: ['O2', 'N2', 'AR']
plt_species = 'O2'
Nc = 50 # Number of contour levels for plot and movie

# Initialize solution vector:
Nspecies = gas.n_species
SV_0 = np.zeros(Nx*Ny*Nspecies)

# Given constants:
F = 96485333           # Faraday Constant [s-A/kmol]
R = ct.gas_constant    # Universal gas constant [J/kmol-K]

""" Pre-process variables/expressions """
"-----------------------------------------------------------------------------"
Temp = Temp + 273.15    # convert temperature from [C] -> [K]
gas.TP = Temp, Press    # set gas state via temperature and pressure

rad = 1e-6*L_pore/2.0   # length of top inlet [m]

# Determine half of max solid distance between pores for domain [m]
if Geom == 1: # square lattice
    max_space = 1e-6*np.sqrt(2)*(spacing - L_pore) / 2.0
elif Geom == 2: # hexagonal lattice
    max_space = 1e-6*(spacing - L_pore) / 2.0

BC_in = int(round(Nx*rad / (rad+max_space))) # x-nodes affected by inlet BC
if BC_in == 0:
    BC_in = 1
    
# Load method type based on stiffness
if i_stiff == 0:
    method = 'RK45'
elif i_stiff == 1:
    method = 'LSODA'

# Define differential distances
dX = (rad + max_space) / Nx
dY = 1e-6*t_glass / Ny

# Set up initial values at each cell with ambient simple_air
SV_0 = np.tile(gas.Y*gas.density_mass,Nx*Ny)
inlet_BC = gas.Y*gas.density_mass

# Solve for O2 flux at outlet BC based on i_curr and Faraday
iO2 = gas.species_index('O2')
MW_O2 = gas.molecular_weights[iO2]
J_O2_out = MW_O2 *i_curr *100**2 /F /4
J_BC = np.zeros(Nx*Nspecies)
J_BC[iO2::Nspecies] = J_O2_out

# Set up new constants to reduce use of division:
phi_inv = 1/phi_g
dY_inv = 1/dY
dX_inv = 1/dX
    
""" Sub-functions for integrator """
"-----------------------------------------------------------------------------"
if Diff_Mod == 0:
    from Ficks_func import Flux_Calc
elif Diff_Mod == 1:
    from DGM_func import Flux_Calc
    
def dSVdt_func(t,SV): # System of ODEs to solve
    dSVdt = np.zeros(Nx*Ny*Nspecies)
    Fluxes_X, Fluxes_Y = Flux_Calc(SV,Nx,dX,Ny,dY,Nspecies,BC_in,inlet_BC,gas,phi_g,tau_g,d_p)
    Fluxes_Y[Ny*Nx*Nspecies:] = J_BC # Constant flux out BC
    
#    print(np.reshape(Fluxes_X[iO2::Nspecies],(Ny,Nx+1)))
#    print(np.reshape(Fluxes_Y[iO2::Nspecies],(Ny+1,Nx)))
#    print(np.reshape(SV[iO2::Nspecies],(Ny,Nx)))
#    input('Press Enter to continue')
               
    # Initialize the fluxes into the first row (y-direction)
    Flux_Y_in = Fluxes_Y[0:Nx*Nspecies]
            
    # Vector math for inside domain all but last row of cells
    for j in range(Ny):
        ind1 = j*Nx*Nspecies # index for first cell in each row
        ind2 = (j+1)*Nx*Nspecies # index for last cell in each row

        Flux_Y_out = Fluxes_Y[(j+1)*Nx*Nspecies:(j+2)*Nx*Nspecies]
        
        Flux_X_in = Fluxes_X[j*(Nx+1)*Nspecies:(j+1)*(Nx+1)*Nspecies-Nspecies]
        Flux_X_out = Fluxes_X[j*(Nx+1)*Nspecies+Nspecies:(j+1)*(Nx+1)*Nspecies]
        
        dSVdt[ind1:ind2] = phi_inv*((Flux_Y_in - Flux_Y_out)*dY_inv \
                         + (Flux_X_in - Flux_X_out)*dX_inv)
        
        # The fluxes leaving the current row are the inlets to the next row
        Flux_Y_in = Flux_Y_out
        
    return dSVdt

""" Call ODE integrator and process results """
"-----------------------------------------------------------------------------"
sol = solve_ivp(dSVdt_func, [0, t_sim], SV_0, method=method) # Use ODE solver

plt_species_ind = gas.species_index(plt_species) # Extract index for plot

# Define variable vectors to be plotted
x_plt = np.linspace(0, rad+max_space, Nx) * 1e6
y_plt = -1*np.linspace(0, t_glass, Ny)
SV = sol.y.T
sol_t = sol.t
SV_plt = np.reshape(SV[-1,plt_species_ind::Nspecies], (Ny,Nx))

# Create contour plot
plt.contourf(x_plt,y_plt,SV_plt,
             levels=np.linspace(np.min(SV_plt),np.max(SV_plt),Nc))
plt.title('%s Density [kg/m^3 gas]' %plt_species)
plt.xlabel('Horizontal distance, x [micron]')
plt.ylabel('Glass thickness, y [micron]')
plt.colorbar()
plt.show()

# Create movie of entire transient solution over time interval
if movie == 1:
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set(xlim=(0, (rad+max_space)*1e6), ylim=(-t_glass, 0))
    
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    
    SV_cb = SV[:,plt_species_ind::Nspecies]
    
    ax.set_xlabel('Horizontal distance, x [micron]')
    ax.set_ylabel('Glass thickness, y [micron]')
    
    def animate(i):
        ax.collections = []
        SV_plt = np.reshape(SV[i,plt_species_ind::Nspecies],(Ny,Nx))
        cf = ax.contourf(x_plt,y_plt,SV_plt,
                         levels=np.linspace(np.min(SV_cb),np.max(SV_cb),Nc))
        cax.cla()
        fig.colorbar(cf, cax=cax)
        ax.set_title('Time (s): ' + str(round(sol_t[i])))
        
    anim = FuncAnimation(fig, animate, interval=10, frames=len(sol_t)-1)
    
    anim.save('animation.html')


















""" Comments and future steps """
"-----------------------------------------------------------------------------"
# Determine appropriate discretization in time and space and apply for a time
# that allows for a steady-state solution to be reached.

# Exercise model for different geometries. 