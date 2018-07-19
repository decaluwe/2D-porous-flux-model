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
        
    This model has also been expanded to track the diffusion of species in fuel
    cells. The user is able to define the reactive species at the constant flux
    outlet boundary condition so they are not limited to choosing fluids with 
    O2 as a species.
    
"""

""" Load any needed modeules """
"-----------------------------------------------------------------------------"
import os, time
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from shutil import copy2, rmtree
from scipy.integrate import solve_ivp

""" Set up inputs and givens """
"-----------------------------------------------------------------------------"
# Folder name for saving outputs (solution, gas_file, script, plot, animation):
folder_name = 'folder_name'

# Select gas file from Cantera or user-made cti files:
gas_file = 'simple_air.cti'
# 'simple_air.cti' was created from 'air.cti' by removing reactions and all 
# species except O2, N2, and Ar.

# Geometry and physical constants:
L_pore = 2.0      # Diameter of Si wafer pores [microns]
spacing = 6.0     # Horizontal spacing (center to center) of Si pores [microns]
d_p = 1.5e-6      # Mean particle diameter of glass porous medium [m]
r_p = 1.5e-7      # Mean pore radius of glass porous medium [m]
t_glass = 3.0     # Thickness of  glass layer [microns]
Temp = 25.0       # Constant gas temperature [C]
Press = 101325.0  # Initial gas pressure [Pa]
phi_g = 0.5       # Porosity of glass layer
t_sim = 120.0     # Simulation time [s]

# Boundary conditions:
# Constant concentrations of gas will be supplied from inlet user Temp, Press.
# Constant flux of specified species will be caluculated from current below.
i_curr = 0.1     # Current density at bottom boundary [A/cm^2]  

# Tolerance conditions for solver (better convergence, more computation time):
rtol = 1e-9       # Relative tolerance for IVP solver (default=1e-3)
atol = 1e-12      # Absolute tolerance for IVP solver (default=1e-6)

# Switch inputs:
Diff_Mod = 1      # Diffusion model: 1 - Fick's, 2 - Dusty Gas
Geom = 2          # Lattice type: 1 - Square, 2 - Hexagonal
ODE_method = 1    # Solver method: 1 - BDF, 2 - RK45, 3 - LSODA (for Stiff)
movie = 1         # Save movie of transient solution: 0 - Off, 1 - On
frames = 50       # For movie: 0 - frame at each time step, 
                  #     other - approximate # of frames to save generation time

# Discretization for 2D mesh:
Nx = 3      # Integer number of evenly-spaced discretizations in x-direction 
Ny = 3      # Integer number of evenly-spaced discretizations in y-direction 

# Reactive species at bottom boundary (i.e. species with constant flux out):
rxn_species = 'O2'
e_ratio = 4
# The e_ratio is used to calculate the constant flux boundary condition. It is 
# defined as the ratio of moles of electrons to 1 mol of rxn_species from the  
# redox reaction. For example... in O2 + 4H(+) + 4e(-) <--> 2H2O, the e_ratio  
# would be 4 with O2 as the rxn_species since there are 4 mol e(-) to 1 mol O2.
    
# Species name for contour plot/animation:
plt_species = 'O2'

""" Pre-process variables/expressions """
"-----------------------------------------------------------------------------"
# Create folder for any files/outputs to be saved:
if os.path.exists(folder_name):
    print('WARNING: folder_name already exists. Files will be overwritten.')
    input('Press "Enter" to continue and overwrite or "ctrl+c" to cancel.')
    if os.path.exists(folder_name + '/animation_frames'):
        rmtree(folder_name + '/animation_frames')
    if os.path.exists(folder_name + 'animation.html'):
        os.remove(folder_name + '/animation.html')     
else:
    os.makedirs(folder_name)

# Save the current script and cti file into new folder:
cwd = os.getcwd()
gas_path = ct.__path__[0]
copy2(os.path.basename(__file__), folder_name)
if os.path.exists(gas_file):
    copy2(gas_file, folder_name)
else:
    copy2(gas_path + '/data/' + gas_file, folder_name)

# Tortuosity calculation via Bruggeman correlation:
tau_g = phi_g**(-0.5)   

# Call cantera for gas-phase:
if Diff_Mod == 1:
    gas = ct.Solution(gas_file)
elif Diff_Mod == 2:
    gas = ct.DustyGas(gas_file)
    gas.porosity = phi_g
    gas.tortuosity = tau_g
    gas.mean_pore_radius = r_p
    gas.mean_particle_diameter = d_p
    
# Initialize solution vector:
Nspecies = gas.n_species
SV_0 = np.zeros(Nx*Ny*Nspecies)

# Given constants:
F = ct.faraday          # Faraday Constant [s-A/kmol]
R = ct.gas_constant     # Universal gas constant [J/kmol-K]

Temp = Temp + 273.15    # convert temperature from [C] -> [K]
gas.TP = Temp, Press    # set gas state via temperature and pressure

rad = 1e-6*L_pore/2.0   # length of top inlet [m]

# Determine half of max solid distance between pores for domain [m]
if Geom == 1: # square lattice
    max_space = 1e-6*(np.sqrt(2)*spacing - L_pore) / 2.0
elif Geom == 2: # hexagonal lattice
    max_space = 1e-6*(spacing - L_pore) / 2.0

# Number of x-nodes affected by inlet BC:
BC_in = int(round(Nx*rad / (rad+max_space))) 
if BC_in == 0:
    BC_in = 1
    
# Set ODE method type based on input switch:
if ODE_method == 1:
    method = 'BDF'
elif ODE_method == 2:
    method = 'RK45'
elif ODE_method == 3:
    method = 'LSODA'

# Define differential distances:
dX = (rad + max_space) / Nx
dY = 1e-6*t_glass / Ny

# Set up initial values at each cell with ambient simple_air:
SV_0 = np.tile(gas.Y*gas.density_mass,Nx*Ny)
inlet_BC = gas.Y*gas.density_mass

# Solve for flux at outlet BC based on i_curr, Faraday, and e_ratio:
i_rxnsp = gas.species_index(rxn_species)
MW_rxnsp = gas.molecular_weights[i_rxnsp]
J_rxnsp_out = MW_rxnsp *i_curr *100**2 /F /e_ratio
J_BC = np.zeros(Nx*Nspecies)
J_BC[i_rxnsp::Nspecies] = J_rxnsp_out

# Set up new constants to reduce use of division:
phi_inv = 1/phi_g
dY_inv = 1/dY
dX_inv = 1/dX
    
""" Sub-functions for integrator """
"-----------------------------------------------------------------------------"
# Load appropriate flux calculation function based on input switch:
if Diff_Mod == 1:
    from Ficks_func import Flux_Calc
elif Diff_Mod == 2:
    from DGM_func import Flux_Calc
    
# Function for system of ODEs to solve:
def dSVdt_func(t,SV): 
    dSVdt = np.zeros(Nx*Ny*Nspecies)
    Fluxes_X, Fluxes_Y = Flux_Calc(SV,Nx,dX,Ny,dY,Nspecies,BC_in,inlet_BC,gas,
                                   phi_g,tau_g,d_p)
    
    # Set constant flux out BC:
    Fluxes_Y[Ny*Nx*Nspecies:] = J_BC 
                   
    # Initialize the fluxes into the first row (y-direction):
    Flux_Y_in = Fluxes_Y[0:Nx*Nspecies]
            
    # Vector math for inside domain all but last row of cells:
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
print('\nBegin solving transient process.')
t1 = time.time()
sol = solve_ivp(dSVdt_func, [0, t_sim], SV_0, method=method,
                atol=atol, rtol=rtol) # Use ODE solver for IVP
t2 = time.time()
print('Solved', len(sol.t),'steps in', round(t2 - t1,2), 's.',
      'Wait while outputs are generated/saved.')

plt_species_ind = gas.species_index(plt_species) # Extract index for plot

# Define variable vectors to be plotted:
SV = sol.y.T
sol_t = sol.t
SV_plt = np.reshape(SV[-1,plt_species_ind::Nspecies], (Ny,Nx))
    
# Move into new folder to save all outputs:
os.chdir(folder_name)

# Save solution as matrix:
# First column gives time steps returned by IVP_solver. Each row is the entire
# domain solution for each of the species in the gas_file. The solution vector
# matrix is stored starting from the top left corner of the domain and moving
# across each row to the right before moving the the next row and starting 
# again at the left-most cell.
SV_save = np.concatenate((np.array([sol_t]).T, SV), axis=1)
np.savetxt('solution.csv', SV_save, delimiter=',')

# Create 2D pixel plot:
from mpl_toolkits.axes_grid1 import make_axes_locatable
delta_c = '{:.2e}'.format(np.max(SV_plt[-1,:]) - np.min(SV_plt[-1,:]))

fig = plt.figure(1)
ax = fig.add_subplot(111)

div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

ax.set_xlabel(r'Horizontal distance, x [$\mu$m]')
ax.set_ylabel(r'Glass thickness, y [$\mu$m]')
ax.set_title('Time: ' + str(sol_t[-1]) + ' s\n' r'$\Delta\rho_{max}$ outlet: ' 
             + delta_c + r' kg/m$^3$')
cf = ax.imshow(SV_plt,extent=[0,1e6*(rad+max_space),t_glass,0])

ticks = np.linspace(np.min(SV_plt),np.max(SV_plt),5)
cbar = fig.colorbar(cf, cax=cax, ticks=ticks, format='%.7f')
cbar.set_label(r'%s Density, $\rho$ [kg/m$^3$]' %plt_species)

# Add white space padding to avoid save cutoffs:
# Increasing wht_space allows more padding to be added around the figure or
# animation before it is saved so that the labels/values are not cutoff.
wht_space = 1

# Save the 2D pixel contour of the last time step:
fig.tight_layout(pad=wht_space)
plt.savefig('2D_pixel_tf.png')

# Create movie of entire transient solution over specified time interval:
if movie == 1:
    from matplotlib.animation import FuncAnimation
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    
    ax.set_xlabel(r'Horizontal distance, x [$\mu$m]')
    ax.set_ylabel(r'Glass thickness, y [$\mu$m]')
    
    if frames == 0:
        t_movie = sol_t
        SV_movie = SV
    elif len(sol_t) > frames:
        ind_movie = round(len(sol_t) / frames)
        if (len(sol_t)-1) % ind_movie == 0:
            t_movie = sol_t[0::ind_movie]
            SV_movie = SV[0::ind_movie,:]
        else:            
            t_movie = np.append(sol_t[0::ind_movie],sol_t[-1])
            SV_movie = np.append(SV[0::ind_movie,:],[SV[-1,:]],axis=0)      
    
    def animate(i):
        ax.collections = []
        SV_plt = np.reshape(SV_movie[i,plt_species_ind::Nspecies],(Ny,Nx))
        cf = ax.imshow(SV_plt,extent=[0,1e6*(rad+max_space),t_glass,0])
        delta_c = '{:.2e}'.format(np.max(SV_plt[-1,:]) - np.min(SV_plt[-1,:]))
   
        cax.cla()
        ticks = np.linspace(np.min(SV_plt), np.max(SV_plt),5)
        cbar = fig.colorbar(cf, cax=cax, ticks=ticks, format='%.7f')
        cbar.set_label(r'%s Density, $\rho$ [kg/m$^3$]' %plt_species)
        
        if t_movie[i] < 1.0:
            t_current = '{:.2e}'.format(t_movie[i])
        else:
            t_current = '{:.2f}'.format(t_movie[i])
        
        ax.set_title('Time: ' + t_current + ' s\n' 
                     r'$\Delta\rho_{max}$ outlet: ' + delta_c + r' kg/m$^3$')
        
        fig.tight_layout(pad=wht_space)
        
    anim = FuncAnimation(fig, animate, interval=500, frames=len(t_movie))
    
    anim.save('animation.html')
    
# Move back to original cwd after files are saved:
os.chdir(cwd)

""" Comments and future steps """
"-----------------------------------------------------------------------------"
# Add species complexity so that model can be extended to work for Fuel Cell
# processes.

# Determine appropriate discretization in x-y space for grid independence.
