# Computational-Multiphase-Flow

Ideas for final project:
- Trailing suction dredging (https://en.wikipedia.org/wiki/Dredging)
- Jet lift (https://en.wikipedia.org/wiki/Dredging)
* https://dredgers.com.ua/en/drw-1600-25/
* https://www.trodatgroup.com/dredger/suction-dredger/6-8-10-12-inch-jet-lift-dredger.html
* ![image](https://user-images.githubusercontent.com/70904313/220875658-73720f7c-849f-4146-a726-9e739928c15c.png)
---

# Articles
Sediment:
https://drive.google.com/drive/folders/1duQF-7w8QmqLvBh9Hq5cknwRfabj9qSa

Venturi effect:
https://resources.system-analysis.cadence.com/blog/msa2022-venturi-effect-applications-in-cfd-simulations

Sediment in ship and seabed:
https://www.tandfonline.com/doi/full/10.1080/1064119X.2022.2059421

Two phase modeling of sediment plume:
https://link.springer.com/chapter/10.1007/978-981-15-2081-5_24

PDE's in python:
https://fenicsproject.org/pub/tutorial/html/._ftut1009.html

Traling suction pipe concentration:
https://iwaponline.com/jh/article/24/4/730/89187/CFD-simulation-and-model-predictive-control-of-the


# Project 1
*Steady fully-developed single-phase channel flow (same for all groups)*
 - Develop a routine to compute the velocity in a steady fully-developed single-phase
channel flow with variable viscosity, using the finite-volume method. 
- Assume that the viscosity profile is a prescribed input to the routine. 

Consider three possible wall boundary
conditions: 
1. prescribed velocity at the wall,
2. prescribed velocity-gradient at the wall,
3. prescribed relation between the shear-stress at the wall and the velocity field.

Consider two possible global boundary conditions: 
1. prescribed pressure-gradient
2. prescribed flow-rate.

---
