# Phase Transitions
This project consists of various modules I created through out my research
studying graviational waves from first order phase transitions.  It consisits
of a basic example using a toy model.

The main modules are in the phaseTransitions folder and are:
1. energy_budget.py
2. transition_dynamics.py
3. soundshell.py

Helper modules are:
1. parallel.py
2. toy_model.py

## Energy Budget
The module energy_budget.py consists of an implementation of https://arxiv.org/abs/1004.4187 to
study the hydrodynamics of bubble growth in first-order phase transitions. The code can solve the velocity profile and enthalpy density around single bubble and can calculate the efficiency of the transfer of vacuum energy to the bubble wall and the plasma.  The code can work in all regimes such as deflagration, hybrid, and detonation. 

To get the self similiar coordinate and the velocity, enthalpy density, and energy density, you can call 

    EB = energy_budget.EnergyBudget()
    xi, v, w, e = EB(vw=vw, alpha_n=alpha, nv=5001)

for a given bubble wall velocity and phase transition strength $\alpha$. A sample output is 

![profile plots](/figures/profiles.png)
