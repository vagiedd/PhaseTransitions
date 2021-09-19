import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import scipy.optimize as optimize

class TransitionDynamics:
    def __init__(self, T, SoverT, dSoverTdT=None, vw=0.92, kappa_m=0, 
                delta_V=None,gstar=106.75,gamma=1, nT=101):
        self.vw = vw
        self.kappa_m = kappa_m
        self.gamma = gamma
        self.nT = nT
       
        #calculations
        self.T = T
        self.Tc = T.max()

        self.SoverT = SoverT
        self.S3_T = interpolate.interp1d(T,SoverT,kind="cubic")
        if isinstance(dSoverTdT, np.ndarray):
            self.dS3_TdT = interpolate.interp1d(T, dSoverTdT, kind="cubic")
        else:
            self.dS3_TdT = interpolate.interp1d(T,np.gradient(SoverT,T),kind="cubic")
        if isinstance(delta_V, np.ndarray):
            self.delta_V = interpolate.interp1d(T, delta_V, kind="cubic")
        else:
            self.delta_V = lambda t: 0
        
        #constants
        self.p0bar = 1
        self.gstar = gstar
        self.ac = 1
        self.Mpl = 1.22e19 #GeV
        self.G = 1/self.Mpl**2

    def __call__(self,T,Tn=None):
        # store results
        dic = {}

        dic.update(
            {"T":self.T, "S_3(T)/T":self.S3_T, "d(S_3(T)/T)/dT":self.dS3_TdT}
        )

        #calculate g(T)
        I = np.vectorize(self.I,otypes=[np.float64])(T)
        T,I = T[np.isfinite(I)],I[np.isfinite(I)]
        g = self.g(T,I)
        _g = self.interp(T,g)
        dic["g(T)"] = _g
        if Tn:
            self.Tp = optimize.newton(lambda t: _g(t) - 0.7,Tn)
        else: 
            self.Tp = optimize.brentq(lambda t: _g(t) - 0.7,T[0],T[-1])
        dgdT = self.dgdT(T,g=g)

        #calculate beta_c
        _I = self.interp(T,I)
        try: 
            self.Tf = optimize.brentq(lambda t: _I(t) - 1,T[0],T[-1])
        except:
            self.Tf = optimize.newton(lambda t: _I(t) - 1,self.Tp)
        beta_c = self.a_ac(self.Tf)*self.beta(self.Tf,self.dS3_TdT)

        #calculate Ac/beta
        Ac = self.Ac(T,dgdT)
        mask = np.where(Ac > 0)[0]
        dic["A_c(T)/\\beta_c"] = self.interp(T[mask],Ac[mask]/beta_c,kind="cubic")

        #calculate nb
        nb = np.vectorize(self.nb,otypes=[np.float64])(T,g=dic["g(T)"])
        T,nb = T[np.isfinite(nb)],nb[np.isfinite(nb)]
        nb_H3 = self.interp(T,nb/self.H(T)**3,kind="linear")
        dic["n_b(T)/H^3"] = nb_H3

        #calculate Tn correctly
        try:
            self.Tn = optimize.newton(lambda t: nb_H3(t) - 1,self.Tp)
        except:
            self.Tn = optimize.brentq(lambda t: nb_H3(t) - 1,T[0],T[-1])

        #calculate Rstar
        mask = np.where(nb > 0)[0]
        Rstar = self.Rstar(nb[mask])
        _Rstar = self.interp(T[mask],Rstar,kind="cubic")
        dic["HR_*(T)"] = self.interp(T[mask],Rstar*self.H(T[mask]),kind="cubic")

        self.beta_Hf = (8*np.pi)**(1/3.)*self.vw/(self.H(self.Tf)*_Rstar(self.Tf))

        dic["T_n"] = self.Tn
        dic["T_p"] = self.Tp
        dic["T_f"] = self.Tf
        dic[r"\beta_H"] = self.beta_Hf

        return dic

    def interp(self,x,y,kind="cubic"):
        return interpolate.interp1d(
            x,y,
            kind=kind,
            bounds_error=False,
            fill_value=(0,0)
        )

    def suppression_factor(self,vw,Uf,beta_H):
        HRstar = (8*np.pi)**(1/3.)*vw/(beta_H)
        tauH = HRstar/Uf
        if tauH > 1.0: 
            return {"tauH":1.0, "upsilon":1 - 1/np.sqrt(1 + 2*tauH)}
        else: 
            return {"tauH":tauH, "upsilon":1 - 1/np.sqrt(1 + 2*tauH)}

    def a_ac(self,T):
        return (self.Tc/T)**(1/self.gamma)
        
    def H(self,T):
        delta_V = self.delta_V(T)
        rhostar= np.pi**2/30*self.gstar*self.Tc**4
        ptot = (rhostar + delta_V)/(1-self.kappa_m)
        Hstar = np.sqrt(8*np.pi*self.G/3*ptot)
        y = self.a_ac(T)
        return Hstar*np.sqrt(self.kappa_m/y**3 + (1-self.kappa_m)/y**4 + delta_V/ptot)
        
    def r(self,Tpp):
        with np.errstate(divide='ignore',invalid='ignore'):
            integrand = integrate.simps(
                1/Tpp*1/(self.gamma*self.H(Tpp))*(self.Tc/Tpp)**(-1/self.gamma),Tpp)
        r = self.vw/self.ac*integrand
        r[np.isnan(r)] = 0.0
        return r
    
    def p(self,T):
        return self.p0bar*T**4*np.exp(-self.S3_T(T))
    
    def I(self,T,r=None):
        def f(Tp,Tc,T):
            r = self.r(np.linspace(T, Tp, self.nT))
            f = 1/Tp*1/(self.gamma*self.H(Tp))*self.p(Tp)*(Tc/Tp)**(3/self.gamma)*(self.ac*r)**3
            return f
        Tp = np.linspace(T,self.Tc,self.nT)
        with np.errstate(divide='ignore',invalid='ignore'):
            integrand = integrate.simps(f(Tp,self.Tc,T),Tp)
        I = 4*np.pi/3*integrand
        return I

    def g(self,T,I=None):
        if I is None: 
            g = np.exp(-self.I(T))
        else : 
            g = np.exp(-I)
        return g

    def dgdT(self,T,g=None,dT=1e-5):
        if g is None:
            return (self.g(T+dT) - self.g(T-dT))/(2*dT)
        else:
            return np.gradient(g,T)
    
    def beta(self,Tf,dS3_TdT):
        return self.gamma*self.H(Tf)*Tf*dS3_TdT(Tf)
    
    def Ac(self,T,dgdT):
        Ac = self.a_ac(T)/self.vw*(self.H(T)*self.gamma*T)*dgdT
        return Ac

    def nb(self,T,g=None): 
        def f(Tp,Tc,g):
            f = 1/Tp*1/(self.gamma*self.H(Tp))*self.p(Tp)*g(Tp)*(Tc/Tp)**(3/self.gamma)
            return f
        if g is None: 
            integrand,err = integrate.quad(f,T,self.Tc,args=(self.Tc,self.g),
                    limit=1000)
        else:
            Tp = np.linspace(T, self.Tc, 10*self.nT)
            g = g(Tp)
            f = 1/Tp*1/(self.gamma*self.H(Tp))*self.p(Tp)*g*(self.Tc/Tp)**(3/self.gamma)
            not_nan = np.isfinite(f)
            with np.errstate(divide='ignore',invalid='ignore'):
                integrand = integrate.simps(f[not_nan],Tp[not_nan]) 
        return (T/self.Tc)**(3/self.gamma)*integrand

    def Rstar(self,nb):
        return 1/nb**(1/3.)
