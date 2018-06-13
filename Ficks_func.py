"""
Author:
    Corey R. Randall (08 June 2018)
    
Description:
    This is an external function that calculates fluxes via a Fick's diffusion
    model. It was written to be used in 2D_NRSupport_FluxModel.
"""

def Flux_Calc(SV,Nx,dX,Ny,dY,Nspecies,inlet_BC,gas,phi_g,tau_g):
    
    import numpy as np
        
    Fluxes_X = np.zeros((Nx+1)*Ny*Nspecies) # Fluxes in x-direction w/ 0's BC
    Fluxes_X_int = np.zeros((Nx-1)*Ny*Nspecies) # Interior x-direction fluxes
    Fluxes_Y = np.zeros(Nx*(Ny+1)*Nspecies) # Fluxes in y-direction
    
    # Initialize counters for flux loops:
    cnt_x = 0
    cnt_y = Nx*Nspecies
    
    # Calculate each x-direction flux:
    for j in range(Ny):
        ind1 = j*Nx*Nspecies # First -> last index of cell on left
        ind2 = ind1 + Nspecies
        for i in range(Nx-1):
            ind3 = ind2 # First -> last index of cell on right
            ind4 = ind3 + Nspecies
            
            Y1 = SV[ind1:ind2]/np.mean(SV[ind1:ind2])
            Y2 = SV[ind3:ind4]/np.mean(SV[ind3:ind4])
            
            gas.Y = np.mean([Y1,Y2],axis=0)
            D_AB = gas.mix_diff_coeffs_mass
            Delta_Y = Y2 - Y1 
                                
            Fluxes_X_int[cnt_x:cnt_x+Nspecies] = phi_g/(tau_g**2)*D_AB*Delta_Y
            
            ind1 = ind3 # Index of right cell becomes index of left
            ind2 = ind1 + Nspecies
            cnt_x = cnt_x + Nspecies
            
    for j in range(Ny):
        ind1 = j*(Nx+1)*Nspecies + Nspecies # First non-zero x-flux of each row
        ind2 = ind1 + (Nx-1)*Nspecies # Last non-zero x-flux of each row
        
        Fluxes_X[ind1:ind2] = Fluxes_X_int[j*(Nx-1)*Nspecies:(j+1)*(Nx-1)*Nspecies]
        
    # Calculate each y-direction flux:
    for i in range(int(len(inlet_BC)/Nspecies)):
        Y1 = inlet_BC[i*Nspecies:(i+1)*Nspecies]/np.mean(inlet_BC[i*Nspecies:(i+1)*Nspecies])
        Y2 = SV[i*Nspecies:(i+1)*Nspecies]/np.mean(SV[i*Nspecies:(i+1)*Nspecies])
        
        gas.Y = np.mean([Y1,Y2],axis=0)
        D_AB = gas.mix_diff_coeffs_mass
        Delta_Y = Y2 - Y1
        
        Fluxes_Y[i*Nspecies:(i+1)*Nspecies] = phi_g/(tau_g**2)*D_AB*Delta_Y
        
    for j in range(Ny-1):
        ind1 = j*Nx*Nspecies # First -> last index of cell on left
        ind2 = ind1 + Nspecies
        for i in range(Nx):
            Y1 = SV[ind1:ind2]/np.mean(SV[ind1:ind2])
            Y2 = SV[ind1+Nx*Nspecies:ind2+Nx*Nspecies]/np.mean(SV[ind1+Nx*Nspecies:ind2+Nx*Nspecies])
            
            gas.Y = np.mean([Y1,Y2],axis=0)
            D_AB = gas.mix_diff_coeffs_mass
            Delta_Y = Y2 - Y1
            
            Fluxes_Y[cnt_y:cnt_y+Nspecies] = phi_g/(tau_g**2)*D_AB*Delta_Y
            
            ind1 = ind2
            ind2 = ind1 + Nspecies
            cnt_y = cnt_y + Nspecies
                    
    return Fluxes_X, Fluxes_Y

