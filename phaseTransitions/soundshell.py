import numpy as np
from scipy import integrate
from scipy import interpolate

class SoundShellModel:

    def __init__(self, xi=None, v=None, w=None, e=None, 
                    vw = None, cs=1/np.sqrt(3)):
        assert xi is not None
        assert v is not None
        assert w is not None
        assert e is not None
        assert vw is not None
        self.xi = xi
        self.v = v
        self.w = w
        self.vw = vw
        self.cs = cs
        self.betaR = (8*np.pi)**(1/3.)*vw
        self.T_tilde = np.logspace(-2, 1.301, 1000)

        mean_e = e(xi)[-1]
        mean_w = w(xi)[-1]
        self.lam = lambda x: (e(x) - mean_e)/mean_w
        self.Gamma = mean_w/mean_e

    def dfdz(self, z):
        z = z[:, None]
        I = lambda x: 1/z**2*self.v(x)*(z*x*np.cos(z*x) - np.sin(z*x))
        return 4*np.pi*integrate.simps(I(self.xi), self.xi)

    def l(self, z):
        z = z[:, None]
        I = lambda x: 1/z*self.lam(x)*x*np.sin(z*x)
        return 4*np.pi*integrate.simps(I(self.xi), self.xi)

    def A2(self, z):
        return 0.25*(self.dfdz(z)**2 + (self.cs*self.l(z))**2)

    def nu(self, T):
        return np.exp(-T)

    def Uf1d(self, Z):
        I = lambda z: z**2/(2*np.pi**2)*self.dfdz(z)**2
        UF2 = 3/(4*np.pi*self.vw**3)*integrate.simps(I(Z),Z, axis=-1)
        return UF2**0.5

    def Uf(self, Z):
        I1 = lambda t: self.nu(t)*t**3
        I2 = lambda z: z**2/(2*np.pi**2)*self.A2(z)
        UF2 = 2/(self.betaR)**3*(
            integrate.simps(I1(Z),Z, axis=-1)*integrate.simps(I2(Z),Z, axis=-1)
        )
        return UF2**0.5

    def _Pv(self, kR):
        I = lambda t: self.nu(t)*t**6*self.A2(t*kR/self.betaR)
        Pv = 2/self.betaR**3*1/(2*np.pi**2)*(kR/self.betaR)**3*(
            integrate.simps(I(self.T_tilde), self.T_tilde, axis=-1)
        )
        return Pv

    def Pv(self, kR, njobs=None):
        if njobs:
            from phaseTransitions import parallel
            f = parallel.Parallel(self._Pv, cores=njobs)
            Pv = f(kR)
        else:
            Pv = [self._Pv(_kR) for _kR in kR]
        return Pv

    def _Pgw_prime(self, y, Pv, Uf, nz):
        zm = y*(1.0 - self.cs)/(2.0*self.cs)
        zp = y*(1.0 + self.cs)/(2.0*self.cs)
        z = np.logspace(np.log10(zm), np.log10(zp), nz, endpoint=True)
        Pv_bar = lambda kR: np.pi**2/(kR**3*Uf**2)*Pv(kR)
        I = lambda z: 1.0/z*(z-zp)**2*(z-zm)**2/(zp + zm - z)*Pv_bar(z)*Pv_bar(zp+zm-z)
        Pgw = 1.0/(4*np.pi*y*self.cs)*((1.0 - self.cs**2)/(self.cs**2))**2*integrate.simps(I(z),z)
        return 3*(self.Gamma*Uf**2)**2*y**3/(2*np.pi**2)*Pgw

    def Pgw_prime(self, kR, return_Pv = False, **kwargs):
        zm = kR[0]*(1.0 - self.cs)/(2.0*self.cs)
        zp = kR[-1]*(1.0 + self.cs)/(2.0*self.cs)
        z = np.logspace(np.log10(zm), np.log10(zp), len(kR), endpoint=True)
        Pv = interpolate.interp1d(z, self.Pv(z, **kwargs), kind="cubic")
        Uf1d = self.Uf1d(kR)
        Pgw_prime = [self._Pgw_prime(_kR, Pv, Uf1d, len(kR)) for _kR in kR]
        return (Pgw_prime, Pv(kR)) if return_Pv else Pgw_prime
