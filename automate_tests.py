import unittest
import json
import io
from contextlib import redirect_stdout

import numpy as np
from scipy import interpolate

from phaseTransitions import (
    toy_model, transition_dynamics, energy_budget, soundshell
)

class TestPhaseTransitions(unittest.TestCase):
    def setUp(self):
        self.njobs = 4
        xi, v = np.loadtxt(
            "tests/HD_vw_0.8_alpha_0.1.csv", delimiter=",", unpack=True
        )
        self.v = interpolate.interp1d(xi, v, kind="cubic")
        kR, Pv, Pgw = np.loadtxt(
            "tests/GW_vw_0.92_alpha_0.0046.csv", delimiter=",", unpack=True
        )
        self.Pgw = interpolate.interp1d(kR, Pgw, kind="cubic")
        return self

    def test_TransitionDynamics(self):
        Ts = {
            "T_n": 100.62343000794824,
            "T_p": 100.10565432236955,
            "T_f": 100.07717662134833
        }
        gstar = 106.75
        D, T0, E, lam = (0.1, 75, 0.1, 0.2)
        pT = toy_model.solveTransition(D, T0, E, lam)
        TD = transition_dynamics.TransitionDynamics(
            pT["T"], 
            pT["S/T"], 
            dSoverTdT=pT["d(S/T)/dT"], 
            delta_V=pT["\Delta V(T)"], 
            gstar=gstar, 
            vw=0.92,
            kappa_m=0, 
            gamma=1,
            nT=101
        )
        dicTD = TD(pT["T"])
        for key, val in dicTD.items():
            if key in Ts.keys():
                self.assertAlmostEqual(Ts[key], val, 1)
            else:
                continue
        return self

    def test_EnergyBudget(self, vw=0.8, alpha=0.1, _assert=True):
        EB = energy_budget.EnergyBudget()
        xi, v, w, e = EB(
            vw=vw, alpha_n=alpha, nv=5001
        )
        if _assert:
            self.assertAlmostEqual(self.v(0.76), v(0.76), 3)
        return xi, v, w, e

    def test_SoundShellModel(self):
        vw = 0.92
        alpha = 0.0046
        xi, v, w, e = self.test_EnergyBudget(
            vw=vw, alpha=alpha, _assert=False
        )
        SSM = soundshell.SoundShellModel(
            xi=xi, v=v, w=w, e=e, vw=vw
        )
        kR = np.logspace(0, 2, 32, endpoint=True)
        Pgw = SSM.Pgw_prime(kR, njobs=self.njobs)
        Pgw = interpolate.interp1d(kR, Pgw, kind="cubic")
        self.assertAlmostEqual(
            self.Pgw(10.0), Pgw(10.0), 12
        )
        return self

if __name__ == '__main__':
    unittest.main()
