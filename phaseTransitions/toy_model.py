import numpy as np
import matplotlib.pyplot as plt
import json

import scipy.interpolate as interpolate
import scipy.optimize as optimize

def Vtot(X,T,D,T0,E,lam):
    g = 106.75 
    a = np.pi**2/30.*g 
    return -1/3.*a*T**4 + D*(T**2 - T0**2)*X**2 - E*T*X**3 + lam/4*X**4

def S3tilde(alpha):
    return 4*4.85*(1+alpha/4.*(1 + 2.4/(1-alpha) + 0.26/(1-alpha)**2))

def action(T,D,T0,E,lam):
    M2 = 2*D*(T**2 - T0**2)
    alpha = lam*M2/(2*E**2*T**2)
    return M2**(3/2.)/(4*E**2*T**3)*S3tilde(alpha)

def dS3_TdT(T,SoverT):
    T = np.array(T)
    return np.gradient(SoverT,T)

def Tc(D,T0,E,lam):
    return np.sqrt(T0**2/(1-E**2/(lam*D)))

def hydrodynamics(T,*args):
    kwinterp = {"s":0,"k":3,"ext":3}
    ps,pb = [],[]
    phimins = []
    phimin0 = optimize.differential_evolution(Vtot,[[0,10000]],args=(0.0,)+args).x[0]
    F = lambda x,t,*args: Vtot(x,t,*args) + np.abs(Vtot(phimin0,0,*args))
    x = phimin0
    T = np.linspace(0.1*Tc(*args),0.999*Tc(*args),len(T))
    for t in T:
        x = optimize.minimize(Vtot,x,args=(t,)+args,method="Nelder-Mead").x[0]
        phimins.append(x)
        ps.append(-F(0,t,*args))
        pb.append(-F(x,t,*args))
    phimins = np.array(phimins)
    ps = np.array(ps)
    pb = np.array(pb)
    phimins = interpolate.UnivariateSpline(T,phimins,**kwinterp)
    ps = interpolate.UnivariateSpline(T,ps,**kwinterp)
    pb = interpolate.UnivariateSpline(T,pb,**kwinterp)
    es = lambda t: t*ps.derivative()(t) - ps(t)
    eb = lambda t: t*pb.derivative()(t) - pb(t)
    ws = lambda t: es(t) + ps(t)
    deltaV = lambda t: -(ps(t) - pb(t))
    ddeltaV = lambda t: -(ps.derivative()(t) - pb.derivative()(t))
    cs2s = lambda t: ps.derivative()(t)/(t*ps.derivative(n=2)(t))
    cs2b = lambda t: pb.derivative()(t)/(t*pb.derivative(n=2)(t))
    gamma = lambda t: ws(t)/es(t)
    alphabar = lambda t: 1/(3*ws(t))*((1+cs2b(t)**-1)*deltaV(t) - t*ddeltaV(t))
    g = 106.75 
    alpha = lambda t: 1/(np.pi**2/30*g*t**4)*(deltaV(t) - 1/4.*t*ddeltaV(t))
    return alpha,alphabar,deltaV,cs2s,cs2b,phimins

def _solveTransitions(D,T0,E,lam):
    with np.errstate(divide='ignore',invalid='ignore'):
        kwinterp = {"s":0,"k":3,"ext":3}
        g = 106.75 
        args = (D,T0,E,lam) 
        # compute the aciton and its derivative. estimate start and end temperature so that S(T) is between from 50 and 1000
        T = np.linspace(0.1*Tc(*args),0.99*Tc(*args),10000)
        S = action(T,*args)
        T = T[np.isfinite(S)]
        S = S[np.isfinite(S)]
        l = np.where((S>=50)&(S<=1000))
        S3_T = interpolate.UnivariateSpline(T[l],S[l],**kwinterp)
        dSdT = S3_T.derivative()
        Tstart = T[l][0]
        Tend = T[l][-1]
        T = np.linspace(Tstart,Tend,1000)
        alphabag,alphabar,deltaV,cs2,cb2,phimins = hydrodynamics(T,*args)
        c2 = np.column_stack((cb2(T), cs2(T)))
    return T, S3_T(T), dSdT(T), alphabag(T), c2, phimins(T), deltaV(T) 

def solveTransition(*args):
    res = _solveTransitions(*args)
    labels = [
        "T","S/T","d(S/T)/dT",r"\alpha(T)","c_s^2",r"\phi(T)",r"\Delta V(T)" 
    ]
    return dict(zip(labels, res))
