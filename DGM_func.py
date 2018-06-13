"""
Author:
    Corey R. Randall (08 June 2018)
    
Description:
    This is an external function that calculates fluxes with the Dusty Gas 
    Model approach. It was written to be used in 2D_NRSupport_FluxModel.
"""

def Flux_Calc(SV,Nx,dX,Ny,dY,Nspecies,inlet_BC,gas,phi_g,tau_g):
    
    import numpy as np
    
    Fluxes_X = np.zeros((Nx+1)*Ny*Nspecies) # Fluxes in x-direction w/ 0's BC
    Fluxes_X_int = np.zeros((Nx-1)*Ny*Nspecies) # Interior x-direction fluxes
    Fluxes_Y = np.zeros(Nx*(Ny+1)*Nspecies) # Fluxes in y-direction 
    
    # Get molecular weights for mol -> mass conversions:
    MWs = gas.molecular_weights
    
    # Extract constant temperature [K] and density [kg/m^3]:
    T = gas.T
    D = gas.density_mass # Assumes incompressible flow
    
    # Initialize counters for flux loops:
    cnt_x = 0
    cnt_y = Nx*Nspecies
    
    # Calculate each x-direction flux:
    Y1 = SV[0:Nspecies]/sum(SV[0:Nspecies])
    
    for j in range(Ny):
        for i in range(1,Nx):
            Y2 = SV[(j+1)*i*Nspecies:(j+1)*i*Nspecies+Nspecies]\
               / sum(SV[(j+1)*i*Nspecies:(j+1)*i*Nspecies+Nspecies])

            molar_fluxes = gas.molar_fluxes(T,T,D,D,Y1,Y2,dX)
            Fluxes_X_int[cnt_x:cnt_x+Nspecies] = molar_fluxes*MWs
            
            Y1 = Y2 # Right cell becomes left when moving across row
            cnt_x = cnt_x + Nspecies
                    
    for j in range(Ny):
        ind1 = j*(Nx+1)*Nspecies + Nspecies # First non-zero x-flux of each row
        ind2 = ind1 + (Nx-1)*Nspecies # Last non-zero x-flux of each row
        
        Fluxes_X[ind1:ind2] = Fluxes_X_int[j*(Nx-1)*Nspecies:(j+1)*(Nx-1)*Nspecies]
    
    # Calculate each y-direction flux:
    for i in range(int(len(inlet_BC)/Nspecies)):
        Y1 = inlet_BC[i*Nspecies:(i+1)*Nspecies]
        Y2 = SV[i*Nspecies:(i+1)*Nspecies]
        
        molar_fluxes = gas.molar_fluxes(T,T,D,D,Y1,Y2,dY)
        Fluxes_Y[i*Nspecies:(i+1)*Nspecies] = molar_fluxes*MWs
    
    for j in range(Ny-2): 
        for i in range(Nx):
            Y1 = SV[j*Nx*Nspecies+i*Nspecies:j*Nx*Nspecies+(i+1)*Nspecies]
            Y2 = SV[(j+1)*Nx*Nspecies+i*Nspecies:(j+1)*Nx*Nspecies+(i+1)*Nspecies]
            
            molar_fluxes = gas.molar_fluxes(T,T,D,D,Y1,Y2,dY)
            Fluxes_Y[cnt_y:cnt_y+Nspecies] = molar_fluxes*MWs
            
            cnt_y = cnt_y + Nspecies
               
    return Fluxes_X, Fluxes_Y


















