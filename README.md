[![DOI](https://zenodo.org/badge/135629486.svg)](https://zenodo.org/badge/latestdoi/135629486)

# 2D-porous-flux-model
Modling 2D porous flux for NR flow through support

## Objective
This simulation models the difusion of O2 through a porous layer. 
The simulation is used to optimize the design of a support for 
flow-through Neutron Reflectometry (NR) measurements. The structure 
will support a membrane being measured via NR, while allowing a 
flux of species from one side to the other. The support structure 
must meet the following requirements:
* Extremely flat
* Porous
* Mechanically strong/structurally supporting
* Uniform species concentrations at the sample interface

## Geometry
The support being modeled has a two-layer structure: a structural 
layer (Si wafer with axial pores [i.e. 'pinholes'] etched through 
it), and a porous diffusion layer.  The structural layer gives
mechanical strength and a flat surface, while the porous diffusion
layer allows species to diffuse laterally so that the species 
concentrations are uniform by the time they reach the sample. An example
of this stucture can be seen in Figure 1.

![github image](https://user-images.githubusercontent.com/39809042/42965564-a609f95c-8b57-11e8-8642-38bd77588ee3.PNG)
<p align="center"> Figure 1: Visualization of discretization for support structure including top and side views

## Simulation Domain
The simulation is of the porous diffusion layer.
The pores are filled with air, and O2 is drawn out through the 
bottom boundary with a uniform and invariant flux that is 
calculated according to a user-input current density.  The conversion
assumes that a PEMFC fuel cell cathode exists at the bottom boundary,
and the flux is equal to that needed to support the equivalent oxygen
reduction reaction rate (written here as a global reaction):

     O2 + 4 H(+) + 4 e(-) <--> 2 H2O

The top boundary of the simulation is the polished Si wafer with axial
channels etched straight through it.  This boundary has two domains: 
Beneath the open axial channel, which is assumed to contain air with a 
constant temperature, pressure, and composition (Dirichlet boundary 
condition). The remaining portion of the top boundary is the Si, which 
imposes a zero-flux boundary condition.

The simulation is in 2-D, and the simulation domain spans from the 
mid-point of the axial pore to the point halfway between axial pores.
I.e., the lateral boundaries are symmetry planes and therefore have 
zero-flux boundary conditions.

## Simulation methods
The model uses a finite-volume approach, and models the diffusion of 
the O2 transiently, simulating out to a long enough time to reach 
steady state. At steady state, the maximum variation of the O2 
concentration at the sample (bottom) interface is calculated, in 
order to assess the suitability of the proposed structures.

## Running the Model
To run the model, after downloading or cloning this repository, one
executes the Python file `2D_FluxModel.py`.

Open and edit the file first, to adjust the user inputs for your own
particular application (all inputs are at the top of the file, and
are described in the section below).  Then save, close, and run the 
file using your preferred method for running Python files.

While the model calls either of the files Ficks_func.py or DGM_func.py
in order to evaluation mass flux rates, there should be no need for you
to ever interact directly with either of these two files.

## User inputs
The user inputs the geometry/microstructure for the porous diffusion 
layer and the boundary conditions:
* Uniform temperature for the entire simulation domain
* Pressure in axial pores
* Width of axial pores in Si and distance between axial pores
* Thickness of porous diffusion layer
* Diffusion layer porosity
* Avg. pore radius within porous diffusion layer
* Avg. particle size in porous diffusion layer
* Current density at PEMFC boundary
* Simulation time for transient process
* Absolute and relative tolerances for solver
* Discretization in the horizontal (x) and vertical (y) directions
* cti file containing species that will be tracked
* Name of species for which the plot/animation will be produced
* Reactive species name (i.e. the species that has a constant flux at the outlet)
* The ratio of moles of electrons per 1 mol of the reactive species from the redox reaction

## Additional Options/Switches
The user can also control different diffusive models, geometries, 
and solver methods by changing certain switch options:

* Diffusion model options: 1 - Advection-diffusion model, 2 - Dusty gas model
* Pore geometry options: 1 - Si pores arranged in squares, 2 - hexagonal arrangement
* Solver method options: 1 - Backward differencing, 2 - RK45, 3 - LSODA
* Animation options: 0 - does not produce/save animation, 1 - saves solution animation
* The animation can further be controlled with the 'frames' option. If the user 
chooses frames = 0 then a frame is created for each time step. This can end up 
taking a long time for solutions with a large number of time steps. To save time
 in generating this animation, the user can specify a number of frames to be saved.
 
## License

This tool is released under the BSD-3 clause license, see LICENSE for details.

## Citing the Model
 This model is versioned using Zenodo:
[![DOI](https://zenodo.org/badge/135629486.svg)](https://zenodo.org/badge/latestdoi/135629486)

If you use this tool as part of a scholarly work, please cite using:

> C.R. Randall and S.C. DeCaluwe. (2018) 2D Porous Flux Model v1.0 [software]. Zenodo. https://doi.org/10.5281/zenodo.1317600

A BibTeX entry for LaTeX users is

```TeX
@misc{2dPorousFlux,
    author = {Corey R. Randall and Steven C DeCaluwe},
    year = 2018,
    title = {2D Porous Flux Model v1.0},
    doi = {10.5281/zenodo.1317600},
    url = {https://github.com/decaluwe/2D-porous-flux-model},
}
```

In both cases, please update the entry with the version used. The DOI for the latest version is
given in the badge at the top, or alternately <https://doi.org/10.5281/zenodo.1317600> will
take you to the latest version (and generally represents all versions).

## Extending the model
While the model was written for a rather specific use case, it can be easily extended for other 2D porous flux simulations.  If you think it may be useful for your work, please download and edit as you see fit.  If you make a modification that you think others might also find useful, please consider submitting a pull request, and we are happy to incporate the changes (you are on GitHub, so chances are you already understand all this, but just in case...).

Perhaps the most obvious extension of this tool would be to look at 2D diffusion through the porous Gas Diffusion Layer (GDL) in a polymer electrolyte membrane fuel cell (PEMFC).  The PEMFC GDL has a similar domain, with metallic flow channels/current collectors at the top of the domain, and an (ideally) constant flux of O2 into the catalyst layer at the bottom boundary.  Keeping these current assumptions to model the GDL flow field would simply require changing some of the inputs to reflect the flow-field geometry.  

Relaxing the current assumptions, particularly to incorporate a non-uniform flux at the bottom boudnary should also be relatively straightforward.  The current density at a given node could be written as a function of the local gas concentration, or if one is feeling ambitious, an additional catalyst layer domain could be added.

