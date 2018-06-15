"""
Author:
    Corey R. Randall (08 June 2018)
    
Description:
    This is an external function that calculates fluxes with the Dusty Gas 
    Model approach. It was written to be used in 2D_NRSupport_FluxModel.
"""

def Flux_Calc(SV,Nx,dX,Ny,dY,Nspecies,BC_in,inlet_BC,gas,phi_g,tau_g,d_p):
    
    import numpy as np
    
    Fluxes_X = np.zeros((Nx+1)*Ny*Nspecies) # Fluxes in x-direction w/ 0's BC
    Fluxes_X_int = np.zeros((Nx-1)*Ny*Nspecies) # Interior x-direction fluxes
    Fluxes_Y = np.zeros(Nx*(Ny+1)*Nspecies) # Fluxes in y-direction 
    
    # Get molecular weights for mol -> mass conversions:
    MWs = gas.molecular_weights
    
    # Set constant temperature from main function:
    T = gas.T
     
    # Initialize counters for flux loops:
    cnt_x = 0
    cnt_y = Nx*Nspecies
    
    # Calculate each x-direction flux: 
    for j in range(Ny):
        ind1 = j*Nx*Nspecies # First -> last index of cell on left
        ind2 = ind1 + Nspecies
        for i in range(1,Nx):
            ind3 = ind2 # First -> last index of cell on right
            ind4 = ind3 + Nspecies
            
            Y1 = SV[ind1:ind2] / sum(SV[ind1:ind2])
            D1 = sum(SV[ind1:ind2])
            Y2 = SV[ind3:ind4] / sum(SV[ind3:ind4])
            D2 = sum(SV[ind3:ind4])

            molar_fluxes = gas.molar_fluxes(T,T,D1,D2,Y1,Y2,dX)
            Fluxes_X_int[cnt_x:cnt_x+Nspecies] = molar_fluxes*MWs
            
            ind1 = ind3 # Index of right cell becomes index of left
            ind2 = ind1 + Nspecies
            cnt_x = cnt_x + Nspecies
            
        x1 = j*(Nx+1)*Nspecies + Nspecies # First non-zero x-flux of each row
        x2 = x1 + (Nx-1)*Nspecies # Last non-zero x-flux of each row
        
        Fluxes_X[x1:x2] = Fluxes_X_int[j*(Nx-1)*Nspecies:(j+1)*(Nx-1)*Nspecies]
    
    # Calculate each y-direction flux:
    for i in range(BC_in):
        Y1 = inlet_BC / sum(inlet_BC)
        D1 = sum(inlet_BC)
        Y2 = SV[i*Nspecies:(i+1)*Nspecies]\
           / sum(SV[i*Nspecies:(i+1)*Nspecies])
        D2 = sum(SV[i*Nspecies:(i+1)*Nspecies])
        
        molar_fluxes = gas.molar_fluxes(T,T,D1,D2,Y1,Y2,dY)
        Fluxes_Y[i*Nspecies:(i+1)*Nspecies] = molar_fluxes*MWs
    
    for j in range(Ny-1): 
        ind1 = j*Nx*Nspecies # First -> last index of cell on top
        ind2 = ind1 + Nspecies
        for i in range(Nx):
            Y1 = SV[ind1:ind2] / sum(SV[ind1:ind2])
            D1 = sum(SV[ind1:ind2])
            Y2 = SV[ind1+Nx*Nspecies:ind2+Nx*Nspecies]\
               / sum(SV[ind1+Nx*Nspecies:ind2+Nx*Nspecies])
            D2 = sum(SV[ind1+Nx*Nspecies:ind2+Nx*Nspecies])
            
            molar_fluxes = gas.molar_fluxes(T,T,D1,D2,Y1,Y2,dY)
            Fluxes_Y[cnt_y:cnt_y+Nspecies] = molar_fluxes*MWs
            
            ind1 = ind2
            ind2 = ind1 + Nspecies
            cnt_y = cnt_y + Nspecies
               
    return Fluxes_X, Fluxes_Y


















