"""
Author:
    Corey R. Randall (08 June 2018)
    
Description:
    This is an external function that calculates fluxes via a Fick's diffusion
    model. It was written to be used in 2D_NRSupport_FluxModel.
"""

def Flux_Calc(SV,Nx,dX,Ny,dY,Nspecies,BC_in,inlet_BC,gas,phi_g,tau_g,d_p):
    
    import numpy as np
        
    Fluxes_X = np.zeros((Nx+1)*Ny*Nspecies) # Fluxes in x-direction w/ 0's BC
    Fluxes_X_int = np.zeros((Nx-1)*Ny*Nspecies) # Interior x-direction fluxes
    Fluxes_Y = np.zeros(Nx*(Ny+1)*Nspecies) # Fluxes in y-direction
    
    # Initialize counters for flux loops:
    cnt_x = 0
    cnt_y = Nx*Nspecies
    
    # Extract temperature for defining state at each point:
    Temp = gas.T
    
    # Constants to reduce use of division:
    phi_tau_sq = phi_g / (tau_g**2)
    K_g = 4*d_p**2*phi_g**3 / (72*tau_g**2*(1-phi_g)**2) # permeability [D]
    dX_inv = 1/dX
    dY_inv = 1/dY
    
    # Calculate each x-direction flux:
    for j in range(Ny):
        ind1 = j*Nx*Nspecies # First -> last index of cell on left
        ind2 = ind1 + Nspecies
        for i in range(Nx-1):
            ind3 = ind2 # First -> last index of cell on right
            ind4 = ind3 + Nspecies
            
            Y1 = SV[ind1:ind2] / sum(SV[ind1:ind2])
            Y2 = SV[ind3:ind4] / sum(SV[ind3:ind4])
            
            # Terms for diffusive flux:
            gas.TDY = Temp, sum(np.mean([Y1,Y2],axis=0)), np.mean([Y1,Y2],axis=0)
            D_AB = gas.mix_diff_coeffs_mass
            mu = gas.viscosity
            Delta_Y = Y2 - Y1 
            
            # Terms for convective flux:
            rho_k = np.mean([SV[ind1:ind2],SV[ind3:ind4]],axis=0)
            gas.TDY = Temp, sum(Y1), Y1
            P1 = gas.P
            gas.TDY = Temp, sum(Y2), Y2
            P2 = gas.P
            V_conv = -K_g*(P2 - P1)*dX_inv / mu
                                 
            Fluxes_X_int[cnt_x:cnt_x+Nspecies] = -phi_tau_sq*D_AB*Delta_Y\
                                               + rho_k*V_conv
            
            ind1 = ind3 # Index of right cell becomes index of left
            ind2 = ind1 + Nspecies
            cnt_x = cnt_x + Nspecies
            
        x1 = j*(Nx+1)*Nspecies + Nspecies # First non-zero x-flux of each row
        x2 = x1 + (Nx-1)*Nspecies # Last non-zero x-flux of each row
        
        Fluxes_X[x1:x2] = Fluxes_X_int[j*(Nx-1)*Nspecies:(j+1)*(Nx-1)*Nspecies]
                
        
    # Calculate each y-direction flux:
    for i in range(BC_in):
        Y1 = inlet_BC / sum(inlet_BC)
        Y2 = SV[i*Nspecies:(i+1)*Nspecies]\
           / sum(SV[i*Nspecies:(i+1)*Nspecies])
        
        # Terms for diffusive flux:
        gas.TDY = Temp, sum(np.mean([Y1,Y2],axis=0)), np.mean([Y1,Y2],axis=0)
        D_AB = gas.mix_diff_coeffs_mass
        mu = gas.viscosity
        Delta_Y = Y2 - Y1
        
        # Terms for convective flux:
        rho_k = np.mean([inlet_BC,SV[i*Nspecies:(i+1)*Nspecies]],axis=0)
        gas.TDY = Temp, sum(Y1), Y1
        P1 = gas.P
        gas.TDY = Temp, sum(Y2), Y2
        P2 = gas.P
        V_conv = -K_g*(P2 - P1)*dY_inv / mu
        
        Fluxes_Y[i*Nspecies:(i+1)*Nspecies] = -phi_tau_sq*D_AB*Delta_Y\
                                            + rho_k*V_conv
        
    for j in range(Ny-1):
        ind1 = j*Nx*Nspecies # First -> last index of cell on top
        ind2 = ind1 + Nspecies
        for i in range(Nx):
            Y1 = SV[ind1:ind2] / sum(SV[ind1:ind2])
            Y2 = SV[ind1+Nx*Nspecies:ind2+Nx*Nspecies]\
               / sum(SV[ind1+Nx*Nspecies:ind2+Nx*Nspecies])
            
            # Terms for diffusive flux:
            gas.TDY = Temp, sum(np.mean([Y1,Y2],axis=0)), np.mean([Y1,Y2],axis=0)
            D_AB = gas.mix_diff_coeffs_mass
            mu = gas.viscosity
            Delta_Y = Y2 - Y1
            
            # Terms for convective flux:
            rho_k = np.mean([SV[ind1:ind2],
                             SV[ind1+Nx*Nspecies:ind2+Nx*Nspecies]],axis=0)
            gas.TDY = Temp, sum(Y1), Y1
            P1 = gas.P
            gas.TDY = Temp, sum(Y2), Y2
            P2 = gas.P
            V_conv = -K_g*(P2 - P1)*dY_inv / mu
            
            Fluxes_Y[cnt_y:cnt_y+Nspecies] = -phi_tau_sq*D_AB*Delta_Y\
                                           + rho_k*V_conv
            
            ind1 = ind2
            ind2 = ind1 + Nspecies
            cnt_y = cnt_y + Nspecies
                    
    return Fluxes_X, Fluxes_Y

