from phaseTransitions import (
    toy_model, transition_dynamics, energy_budget, soundshell
)
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def make_PT_plots(data, path=None):
    T = data.pop("T")
    _ = data.pop("d(S/T)/dT")
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for ax, (key, val) in zip(axs.ravel(), data.items()):
        ax.plot(T, val)
        ax.set(xlabel="$T$", ylabel=f"${key}$")
        if len(val.shape) > 1:
            ax.legend(["broken", "symmetric"])
        ax.grid()
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()

def make_TD_plots(data, path=None):
    labels = ["g(T)", "A_c(T)/\\beta_c", "n_b(T)/H^3", "HR_*(T)"]
    T = data["T"]
    a = data["T_f"]
    b = data["T_n"]
    T = T[(T>=a-0.2*(b-a)) & (T<= b)]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for (ax, l) in zip(axs.ravel(), labels): 
        func = data[l]
        ax.plot(T, func(T))
        ax.set(xlabel="$T$", ylabel=f"${l}$", yscale="log")
        ax.grid()
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()

def make_profile_plot(xi, v, w, path=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    axs[0].plot(xi, v(xi))
    axs[0].grid()
    axs[0].set(xlabel=r"$\xi$", ylabel=r"$v(\xi)$")
    axs[1].plot(xi, w(xi))
    axs[1].grid()
    axs[1].set(xlabel=r"$\xi$", ylabel=r"$\omega(\xi)$")
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()

def make_plot(x, y, path = None, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(**kwargs)
    ax.grid()
    if path:
        plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    gstar = 106.75
    vw = 0.92

    D, T0, E, lam = (0.1, 75, 0.1, 0.2)
    pT = toy_model.solveTransition(D, T0, E, lam)
    make_PT_plots(pT.copy(), path="figures/toy_transition.png")

    print("beginning PT calculation")
    TD = transition_dynamics.TransitionDynamics(
        pT["T"], 
        pT["S/T"], 
        dSoverTdT=pT["d(S/T)/dT"], 
        delta_V=pT["\Delta V(T)"], 
        gstar=gstar, 
        vw=vw,
        kappa_m=0, 
        gamma=1,
        nT=101
    )
    dicTD = TD(pT["T"])
    make_TD_plots(dicTD, path="figures/transition_dynamics.png")

    alphaT = interpolate.interp1d(pT["T"], pT[r"\alpha(T)"], kind="cubic")
    alpha = [alphaT(dicTD[t]) for t in ("T_n", "T_p", "T_f")]

    EB = energy_budget.EnergyBudget()
    xi, v, w, e = EB(vw=vw, alpha_n=alpha[0], nv=5001)
    make_profile_plot(xi, v, w, path="figures/profiles.png")

    print("beginning GW calculation")
    SSM = soundshell.SoundShellModel(xi=xi, v=v, w=w, e=e, vw=vw)

    kR = np.logspace(-1, 3, 25, endpoint=True)
    Pgw, Pv = SSM.Pgw_prime(kR, return_Pv=True, njobs=5)
    make_plot(
        kR, Pgw, 
        path="figures/Pgw_prime.png",
        xlabel="$kR_*$", ylabel=r"$(H R_*)^{-1}\mathcal{P}_{gw}(kR_*)$",
        xscale="log", yscale="log"
    )
