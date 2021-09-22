# Phase Transitions
This project consists of various modules I created through out my research
studying graviational waves from first order phase transitions.  It consisits
of a basic example using a toy model.

The main modules are in the phaseTransitions folder and are:
1. transition_dynamics.py
2. energy_budget.py
3. soundshell.py

Helper modules are:
1. parallel.py
2. toy_model.py

This code was written and tested in Python 3. A worked out example of the toy model effective potential can be found in main.py. 

## Table of Contents 
1. [Transition Dynamics](#Transition-Dynamics)
2. [Energy Budget](#Energy-Budget)
3. [Sound Shell Model](#Sound-Shell-Model)
4. [Toy Model](#Toy-Model)
5. [Main](#Main)
6. [Tests](#Tests)

## Transition Dynamics 
The module transition_dynamics.py consists of an implementation of https://arxiv.org/abs/2007.08537 to analyze in detail the dynamics of the phase transition during bubble nucleation, including the false vacuum fraction, bubble lifetime distribution, bubble number density, mean bubble separation, etc., for an expanding universe. The module will also calculate key temperature scales such as the nucleation temperature, percolation temperature, and the temperature when the phase transition ends. 

To get a dictionary of the interpolating functions of various quantities as well as the temperature scales you can call 

    TD = transition_dynamics.TransitionDynamics(
            T, 
            S_T, 
            dSoverTdT=dS_TdT, 
            delta_V=delta_V, 
            gstar=gstar, 
            vw=vw,
            kappa_m=0, 
            gamma=1,
            nT=101
        )
        dicTD = TD(T)

where T, S_T, dS_TdT, delta_V, gstar, vw, kappa_m, gamma=1, nT are an array of temperatures, array of the action/T, the derivative of action/T, the vacuum energy released during the phase transition, degrees of freedom, bubble wall velocity, fraction of matter if the phase transition occured during an Early Matter Dominated era, the factor that determines how the temperature scales with the scale factor in an expanding universe, and the number of points to use in internal computations such as integration. Note that dS_TdT, and delta_V can be either interpolating functions or numpy arrays. 

Sample output is 

<img src="/figures/transition_dynamics.png" alt="transition dynamics" width="500" height="500"/>

## Energy Budget
The module energy_budget.py consists of an implementation of https://arxiv.org/abs/1004.4187 to
study the hydrodynamics of bubble growth in first-order phase transitions. The code can solve the velocity profile and enthalpy density around single bubble and can calculate the efficiency of the transfer of vacuum energy to the bubble wall and the plasma.  The code can work in all regimes such as deflagration, hybrid, and detonation. 

To get the self similiar coordinate and the velocity, enthalpy density, and energy density interpolating functions you can call 

    EB = energy_budget.EnergyBudget()
    xi, v, w, e = EB(vw=vw, alpha_n=alpha, nv=5001)

for a given bubble wall velocity and phase transition strength $\alpha$. A sample output for the velocity and enthalpy density is 

<img src="/figures/profiles.png" alt="profiles" width="500" height="300"/>

## Sound Shell Model
The module soundshell.py will follow an implementation of https://arxiv.org/abs/1909.10040 to calculate the gravitational wave power spectra from first order early Universe phase transitions using the Sound Shell Model.

The module can be run with 

    SSM = soundshell.SoundShellModel(xi=xi, v=v, w=w, e=e, vw=vw)

    kR = np.logspace(-1, 3, 25, endpoint=True)
    Pgw = SSM.Pgw_prime(kR, njobs=5)

where xi, v, w, and e are the results of energy_budget.py and vw is the bubble wall velocity. njobs is a parameter that sets the number of cores during the calculation of Pgw as a function of kR.  This utilizes the helper function in phaseTranstions/parallel.py to use the multiprocess module. This will require downloading the multiprocess module.  The standard multiprocessing module in python is not used due to pickling.  If njobs=None, then multiprocess will not be imported and the code should run even if it is not installed. This will build the velocity power spectrum on a single core. 

A sample of the gravitational wave spectrum calculated in the sound shell model is 

<img src="/figures/Pgw_prime.png" alt="Pgw" width="500" height="300"/>

## Toy Model 
The module toy_model.py is used to trace the phase structure of a toy model effective potetnial and can output a dictionary of the temperature, action/T, derivitative of the action/T, vacuum energy released, the phase transtion strength as a function of temperature, the speed of sound as a function of temperature, and the scalar field vacuum expectation value as a function of temperature. See the appendix of https://arxiv.org/abs/2007.08537 for the definition of the potential. 

It can be called with 

    D, T0, E, lam = (0.1, 75, 0.1, 0.2)
    pT = toy_model.solveTransition(D, T0, E, lam)
    
where D, T0, E, and lam are the model paramters and pT is the resulting dictionary. 

Sample output of the results in the dictionary are 

<img src="/figures/toy_transition.png" alt="toy model" width="500" height="500"/>

## Main
An example implementation of the modules in phaseTransition for the toy model effective potential is given in main.py along with functions to output the sample functions and can be run with 

    python main.py
    
The plots are saved in the figures folder.

## Tests
Automated tests can be run with 

    python automated_tests.py
    
which will test the modules in phaseTransitions and load sample data in the tests folder.
