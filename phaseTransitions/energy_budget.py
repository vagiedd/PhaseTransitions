import numpy as np
from scipy.integrate import solve_ivp,ode,odeint,simps,fixed_quad,quad,quadrature,romberg,cumtrapz,trapz
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline,BarycentricInterpolator
from scipy import optimize

class interp:
    def __init__(self,spl,bounds=[],fill_value=None):
        self.spl = spl
        self.bounds = bounds
        self.fill_value=None

    def __call__(self,x):
        spl = self.spl
        if(type(spl)==tuple): 
            a = self.bounds[0]
            b = self.bounds[1]
            c = self.bounds[2]
            d = self.bounds[3]
            if(self.fill_value is not None):
                f1 = self.fill_value[0]
                f2 = self.fill_value[1]
            else:
                f1 = 0.0
                f2 = 0.0
            spl1 = spl[0]
            spl2 = spl[1]
            x = np.array([x])
            pw = np.piecewise(x,[x<a,np.logical_and(x>=a,x<=b),np.logical_and(x>b,x<c),np.logical_and(x>=c,x<=d),x>d],[f1,lambda x: spl1(x),lambda x: (spl2(c)-spl1(b))/(c-b)*x,lambda x:spl2(x),f2])
            return pw[0]
        else: 
            a = self.bounds[0]
            b = self.bounds[1]
            if(self.fill_value is not None):
                f1 = self.fill_value[0]
                f2 = self.fill_value[1]
            else:
                f1 = 0.0
                f2 = 0.0
            x = np.array([x])
            pw = np.piecewise(x,[x<a,x>b,np.logical_and(x>=a,x<=b)],[f1,f2,lambda x: spl(x)])
            return pw[0]

class EnergyBudget:
    def __init__(self,n=0):
        self.cs = 1/np.sqrt(3.)
        self.profile_tol = 0.001
        self.find_zero_tol = 1.0e-8

        self.n = n

        self.profiles = {}
        self.types = {}
        self.cs_tol = 0.01
        self.interp_xi = None
        self.interp_v = lambda x: np.nan
        self.interp_w = None

        self.type_store = ""

        self.sh = []

        self.vj = None

    def __call__(self,vw=None,alpha_p=None,alpha_n=None,nv=1001):
        if(alpha_p is None): 
            alpha_p = self.get_alpha(vw,alpha_n,nv=nv)
        if(alpha_p == np.nan): 
            return np.nan,np.nan,{}
        profiles = self.get_profile(vw,alpha_p=alpha_p,nv=nv)
        x = np.linspace(0,1.0,nv)
        w = np.vectorize(self.w_over_wn,otypes=[np.float64])(x,vw)
        w = interp1d(x,w,kind='cubic',bounds_error=False,fill_value=(1.0,))
        eps = 3/4.*alpha_n*1.0
        e = 3/4.*w(x[x<=profiles["xi"][-1]])
        e = interp1d(
            x[x<=profiles["xi"][-1]],
            e,
            kind='cubic',bounds_error=False,fill_value=(3/4. + eps,)
        )
        return x, self.interp_v, w, e

    def mu(self,xi,v): return (xi - v)/(1. - xi*v)

    def gamma(self,v): 
        return 1./(np.sqrt(1. - v**2))

    def kinetic_efficiency_factor(self,xi,v,w,vw,alpha,xish): 
        I1 = lambda x: x**2*w(x)*self.gamma(v(x))**2*v(x)**2
        ximax = max([xish,vw])
        mean_w = 1.0
        eps = 3/4.*mean_w*alpha 
        kappa = 3/(eps*vw**3)*quad(I1,0,ximax,limit=1000)[0]
        return kappa

    def thermal_efficiency_factor(self,xi,w,vw,alpha,xish):
        ximax = max([xish,vw])
        I2 = lambda x: x**2*3/4.*(w(x) - 1.0)
        mean_w = 1.0
        eps = 3/4.*mean_w*alpha 
        W = 3/(eps*vw**3)*quad(I2,0,ximax,limit=1000)[0] 
        return W

    def ek(self,xi,v,w,ximax):
        I1 = lambda x: x**2*w(x)*self.gamma(v(x))**2*v(x)**2
        tol = {"epsrel":1e-10,"epsabs":1e-10}
        ek = 4*np.pi*quad(I1,0,ximax,limit=1000,**tol)[0]
        return ek

    def Uf2(self,xi,v,w,vw,xish):
        ximax = max([xish,vw])
        tol = {"epsrel":1e-10,"epsabs":1e-10}
        mean_w = 1.0
        Uf2 = 3*self.ek(xi,v,w,ximax)/(4*np.pi*mean_w*vw**3)
        return Uf2

    def shock_front(self,xi): 
        return (self.cs**2/xi - xi)/(self.cs**2 - 1.)

    def rarefraction_front(self,xi): 
        return (self.cs - xi)/(xi*self.cs - 1.)

    def jouget_velocity(self,alpha_plus): 
        return 1./(np.sqrt(3.)*(1. + alpha_plus))*(np.sqrt(alpha_plus*(2. + 3.*alpha_plus))+1.)

    def diffy_q(self,t,y):
        n = self.n
        v = t
        xi = y[0]
        f = (xi*(-1 + v*xi)*(1 - 3*v**2 + 4*v*xi + (-3 + v**2)*xi**2))/(-2*v*(-1 + v**2)*(-1 + v*xi)**2)
        return f

    def jac(self,t,y):
        n = self.n
        v = t
        xi = y[0]
        jac = (1 - 3*v**2 + xi*(8*v - xi*(9 + v**2 + 2*v*(-3 + v**2)*xi)))/(2.*v*(-1 + v**2)*(-1 + v*xi)**2)
        return [[jac]]

    def event(self,t,y):
        xi = y[0]
        v = t
        return  (xi - v)/(1.0 - xi*v + 1.0e-100)*xi - self.cs**2

    def velocity_profile_shock(self,xi_i,v_i,v_f,nv=1000,solver="Radau"):
        teval=np.linspace(v_i,v_f,nv,endpoint=True)
        sol = solve_ivp(self.diffy_q,[v_i,v_f],[xi_i],t_eval=teval,events=self.event,dense_output=True,method=solver,jac=self.jac)
        vsh = sol.t_events[0]
        xi = sol.y[0,...]
        v = sol.t
        if(xi[0]>xi[-1]):
            xi = np.flipud(xi)
            v = np.flipud(v)
        if(len(vsh)==0 or np.any(np.array(vsh)==None) ):
            sh = [self.cs,0.0]
        else:
            mask = np.where(v>=max(vsh))[0]
            xi = xi[mask]
            v = v[mask]
            try:
                sh = [optimize.brentq(lambda xi,vsh: xi*(xi - vsh)/(1.0 - xi*vsh) - self.cs**2,1/np.sqrt(3),1.0,maxiter=1000,args=(vsh.max(),)),vsh.max()]
            except:
                sh = [optimize.newton(lambda xi,vsh: xi*(xi - vsh)/(1.0 - xi*vsh) - 1/np.sqrt(3),1/np.sqrt(3)*1.2,maxiter=1000,args=(vsh.max(),)),vsh.max()]
        self.sh = sh
        return xi,v,sh

    def v_plus(self,alpha_plus,v_minus):
        #computes the two branches of v_plus
        f0 = 1/(1.0 + alpha_plus)
        f1 = v_minus/2. + 1/(6.*v_minus)
        f2 = np.sqrt((v_minus/2. + 1/(6.*v_minus))**2 + alpha_plus**2 + 2/3.*alpha_plus - 1/3.)
        v_plus = [f0*(f1 + f2), f0*(f1-f2)]
        if(v_minus > 1/np.sqrt(3)): return v_plus[0]
        else: return v_plus[1]

    def find_v_minus(self,alpha_plus,v_p,tol=1.0e-8,print_guess=False):
        #finds v_minus by taking in alpha_plus and v_plus
        def v_minus(alpha_plus,v_p):
            vm_plus = 1/2.*(((1 + alpha_plus)*v_p + (1 - 3*alpha_plus)/(3*v_p)) + np.sqrt(((1 + alpha_plus)*v_p + (1 - 3*alpha_plus)/(3*v_p))**2 - 4/3.))
            vm_minus = 1/2.*(((1 + alpha_plus)*v_p + (1 - 3*alpha_plus)/(3*v_p)) - np.sqrt(((1 + alpha_plus)*v_p + (1 - 3*alpha_plus)/(3*v_p))**2 - 4/3.))
            return [vm_plus,vm_minus]
        if(v_p < 1/np.sqrt(3)): v_m = v_minus(alpha_plus,v_p)[1]
        else: 
            v_m = v_minus(alpha_plus,v_p)[0]
        return v_m

    def w_over_wn(self,xi_0,v_w):
        v_p = self.profiles["v_plus"]
        v_m = self.profiles["v_minus"]
        profile_type = self.profile_type(v_p,v_m)
        I1 = lambda v,xi,cs: self.gamma(v)**2*self.mu(xi(v),v)*(1.0/cs**2+1.0)
        I2 = lambda xi,v,cs,n: n*v(xi)/(1.-xi*v(xi))*(1/cs**2 + 1.0)
        if self.profiles["type"] == "Detonation" :
            if(xi_0 > v_w):
                return 1
            else:
                r = v_w/(1. - v_w**2)*(1. - v_m**2)/v_m*np.exp(-fixed_quad(I1,self.interp_v(xi_0),self.interp_v(v_w),args=(self.interp_xi,self.cs,))[0] + fixed_quad(I2,xi_0,v_w,args=(self.interp_v,self.cs,self.n,))[0])
                return r
        elif self.profiles["type"] == "Deflagration" :
            xish = self.profiles["xish"]
            vsh = self.profiles["vsh"]
            if(xi_0 > xish): 
                return 1
            elif(xi_0 >= v_w):
                r = xish/(1.0-xish**2)*(1.0-self.mu(xish,vsh)**2)/self.mu(xish,vsh)*np.exp(-fixed_quad(I1,self.interp_v(xi_0),vsh,args=(self.interp_xi,self.cs,))[0] + fixed_quad(I2,xi_0,xish,args=(self.interp_v,self.cs,self.n,))[0])
                return r
            else:
                r = xish/(1.0-xish**2)*(1.0-self.mu(xish,vsh)**2)/self.mu(xish,vsh)*np.exp(-fixed_quad(I1,self.interp_v(v_w),vsh,args=(self.interp_xi,self.cs,))[0] + fixed_quad(I2,v_w,xish,args=(self.interp_v,self.cs,self.n,))[0])
                r = r*self.mu(v_w,self.interp_v(v_w))/(1. - self.mu(v_w,self.interp_v(v_w))**2)*(1. - v_w**2)/v_w
                return r
        else:
            xish = self.sh[0]
            vsh = self.sh[1]
            if(xi_0 > xish): 
                return 1
            elif(xi_0 >= v_w):
                r = xish/(1.0-xish**2)*(1.0-self.mu(xish,vsh)**2)/self.mu(xish,vsh)*np.exp(-fixed_quad(I1,self.interp_v(xi_0),vsh,args=(self.interp_xi,self.cs,))[0] + fixed_quad(I2,xi_0,xish,args=(self.interp_v,self.cs,self.n,))[0])
                return r
            else:
                r = xish/(1.0-xish**2)*(1.0-self.mu(xish,vsh)**2)/self.mu(xish,vsh)*np.exp(-fixed_quad(I1,self.interp_v(v_w),vsh,args=(self.interp_xi,self.cs,))[0] + fixed_quad(I2,v_w,xish,args=(self.interp_v,self.cs,self.n,))[0])
                r = r*v_p/(1.0 - v_p**2)*(1.0 - self.cs**2)/self.cs*np.exp(-fixed_quad(I1,self.interp_v(xi_0),self.mu(v_w,v_m),args=(self.interp_xi,self.cs,))[0] + fixed_quad(I2,xi_0,v_w,args=(self.interp_v,self.cs,self.n,))[0])
                return r

    def profile_type(self,v_plus,v_minus):
        #Determine the profile type from v_plus and v_minus
        tol = self.profile_tol
        if(v_plus > v_minus):
            if(self.cs > v_minus): 
                profile = "Strong detonation"
            if(self.cs < v_minus): 
                profile = "Weak detonation" 
            if(np.abs(v_minus - self.cs) <= tol): 
                profile = "Jouget detonation" 
        elif(v_plus < v_minus):
            self.types["deflagration"] = True
            if(self.cs < v_minus ): 
                profile = "Strong deflagration"
            if(self.cs > v_minus): 
                profile = "Weak deflagration" 
            if(np.abs(v_minus - self.cs) <= tol): 
                profile = "Jouget deflagration" 
        else: profile = "forbidden"

        return profile
    
    def detonation_profile(self,v_w,alpha_p,nv=1000):
        v_plus = v_w
        vm = self.find_v_minus(alpha_p,v_plus)
        xi_i = v_plus 
        v_i = self.mu(v_plus,vm)
        v_f = 1.0e-6#0.00001
        xi,v,sh=self.velocity_profile_shock(xi_i,v_i,v_f,nv=nv,solver="Radau")
        loc = np.where(v<=self.rarefraction_front(xi))[0]
        if(len(loc)>0): 
            v = v[loc]
            xi = xi[loc]
        self.interp_xi = interp1d(v,xi,kind='linear',bounds_error=False,fill_value=(0,))
        pos = np.argsort(xi)
        xi = xi[pos]
        v = v[pos]
        spl = UnivariateSpline(xi,v,s=0)
        self.interp_v = interp(spl,bounds=[xi[0],xi[-1]])
        return xi,self.interp_v,sh

    def deflagration_profile(self,v_w,alpha_p,nv=1000):
        v_minus = v_w
        vp = self.v_plus(alpha_p,v_minus)
        xi_i = v_minus
        v_i = self.mu(v_minus,vp)
        v_f = 1.0e-6
        xi,v,sh=self.velocity_profile_shock(xi_i,v_i,v_f,nv=nv,solver="Radau")
        pos = np.argsort(xi)
        xi = xi[pos]
        v = v[pos]
        spl = UnivariateSpline(xi,v,s=0)
        self.interp_v = interp(spl,bounds=[xi[0],xi[-1]])
        self.interp_xi = interp1d(v,xi,bounds_error=False,fill_value=(0,))
        return xi,self.interp_v,sh
    
    def hybrid_profile(self,v_w,alpha_p,nv=1000):
        v_minus = self.cs
        vp = self.v_plus(alpha_p,v_minus)
        xi_i = v_w
        v_i = self.mu(v_w,v_minus)
        v_f = 1e-6
        xi_1,v_1,sh=self.velocity_profile_shock(xi_i,v_i,v_f,nv=nv,solver="Radau")
        xi_i = v_w
        v_i = self.mu(v_w,vp)
        xi_2,v_2,sh=self.velocity_profile_shock(xi_i,v_i,v_f,nv=nv,solver="Radau")
        xi = np.hstack((xi_1,xi_2))
        v = np.hstack((v_1,v_2))
        pos = np.argsort(xi_1)
        xi_1 = xi_1[pos]
        v_1 = v_1[pos]
        pos = np.argsort(xi_2)
        xi_2 = xi_2[pos]
        v_2 = v_2[pos]
        try:
            spl1 = UnivariateSpline(xi_1,v_1,s=0)
            spl2 = UnivariateSpline(xi_2,v_2,s=0)
            self.interp_v = interp((spl1,spl2,),bounds=[xi_1[0],xi_1[-1],xi_2[0],xi_2[-1]])
        except:
            self.interp_v = interp1d(xi,v,kind="linear",bounds_error=False,fill_value=(0,))
        self.interp_xi = interp1d(v,xi,kind="linear",bounds_error=False,fill_value=(0,))
        return xi,self.interp_v,sh

    def get_alpha(self,v_w,alpha_n,nv=1000):
        I1 = lambda v,xi,cs: self.gamma(v)**2*self.mu(xi(v),v)*(1.0/cs**2+1.0)
        I2 = lambda xi,v,cs,n: n*v(xi)/(1.-xi*v(xi))*(1/cs**2 + 1.0)
        if v_w >= self.jouget_velocity(alpha_n) :
            alpha_p = alpha_n
            self.type_store = "Detonation"
        elif( v_w >= self.cs and v_w < self.jouget_velocity(alpha_n) ):
            def fun(alpha,v_w,alpha_n):
                self.xi,self.v,sh = self.hybrid_profile(v_w,alpha)
                xish = sh[0]
                vsh = sh[1]
                xi_0 = v_w
                kwargs = dict(n=5)
                args = (self.interp_xi,self.cs,)
                I = fixed_quad(I1,self.interp_v(xi_0),vsh,args=args,**kwargs)[0]
                w_over_wn = xish/(1.0-xish**2)*(1.0-self.mu(xish,vsh)**2)/self.mu(xish,vsh)*np.exp(-I)
                return alpha*w_over_wn - alpha_n
            self.type_store = "Hybrid"
            try:
                alpha_p = optimize.brentq(fun,1e-6,alpha_n*1.2,args=(v_w,alpha_n,))
            except:
                alpha_p = optimize.newton(fun,alpha_n/2.,args=(v_w,alpha_n,),maxiter=1000)
            self.xi,self.v,sh = self.hybrid_profile(v_w,alpha_p)
            xi_0 = v_w + 1.0e-8 
            xish = sh[0]
            vsh = sh[1]
            kwargs = dict(n=5)
            args = (self.interp_xi,self.cs,)
            I = fixed_quad(I1,self.interp_v(xi_0),vsh,args=args,**kwargs)[0]
            w_over_wn = xish/(1.0-xish**2)*(1.0-self.mu(xish,vsh)**2)/self.mu(xish,vsh)*np.exp(-I)
        elif( v_w < self.cs ):
            def fun(alpha,v_w,alpha_n):
                self.xi,self.v,sh = self.deflagration_profile(v_w,alpha)
                xish = sh[0]
                vsh = sh[1]
                xi_0 = v_w  + 1.0e-8
                kwargs = dict(n=5)
                args = (self.interp_xi,self.cs,)
                I = fixed_quad(I1,self.interp_v(xi_0),vsh,args=args,**kwargs)[0]
                w_over_wn = xish/(1.0-xish**2)*(1.0-self.mu(xish,vsh)**2)/self.mu(xish,vsh)*np.exp(-I)

                return alpha*w_over_wn - alpha_n
            self.type_store = "Deflagration"
            try:
                alpha_p = optimize.brentq(fun,1e-6,alpha_n*1.2,args=(v_w,alpha_n,))
            except:
                alpha_p = optimize.newton(fun,alpha_n/2.,args=(v_w,alpha_n,),maxiter=1000)
        else :
            return np.nan
        return alpha_p

    def get_profile(self,v_w,alpha_p=None,nv=1001):
        self.profiles = {"type":"","xi":None,"v":None,"v_plus":None,"v_minus":None,"xish":None,"vsh":None,"w_over_wn":None,"alpha_n":None,"alpha_p":None}
        if(self.type_store == ""):
            if(v_w >= self.jouget_velocity(alpha_p) ):
                self.type_store = "Detonation"
            elif( v_w >= self.cs and v_w < self.jouget_velocity(alpha_p) ):
                self.type_store = "Hybrid"
            elif( v_w < self.cs ):
                self.type_store = "Deflagration"
        #DETONATION
        if(self.type_store == "Detonation"):
            v_plus = v_w
            vm = self.find_v_minus(alpha_p,v_plus)
            xi,v,sh = self.detonation_profile(v_w,alpha_p,nv=nv)
            self.profiles["type"] = "Detonation"
            self.profiles["v_plus"] = (v_plus)
            self.profiles["v_minus"] = (vm)
            self.profiles["xi"] = (xi)
            self.profiles["v"] = (v)
            w_over_wn = self.w_over_wn(v_w+1.0e-10,v_w)
            self.profiles["w_over_wn"] = (w_over_wn)
            self.profiles["alpha_n"] = (alpha_p*w_over_wn)
            self.profiles["alpha_p"] = (alpha_p)
            self.profiles["xish"] = sh[0]
            self.profiles["vsh"] = sh[1]
        #HYBRID
        elif(self.type_store == "Hybrid"):
            v_minus = self.cs
            vp = self.v_plus(alpha_p,v_minus)
            xi,v,sh = self.hybrid_profile(v_w,alpha_p,nv=nv)
            self.profiles["type"] = ("Hybrid")
            self.profiles["v_plus"] = (vp)
            self.profiles["v_minus"] = (v_minus)
            self.profiles["xi"] = (xi)
            self.profiles["v"] = (v)
            self.profiles["xish"] = sh[0]
            self.profiles["vsh"] = sh[1]
            w_over_wn = self.w_over_wn(v_w+1.0e-10,v_w)
            self.profiles["w_over_wn"] = (w_over_wn)
            self.profiles["alpha_n"] = (alpha_p*w_over_wn)
            self.profiles["alpha_p"] = (alpha_p)
        #DEFLAGRATION
        elif(self.type_store == "Deflagration"):
            v_minus = v_w
            vp = self.v_plus(alpha_p,v_minus)
            xi,v,sh = self.deflagration_profile(v_w,alpha_p,nv=nv)
            self.profiles["type"] = "Deflagration"
            self.profiles["v_plus"] = (vp)
            self.profiles["v_minus"] = (v_minus)
            self.profiles["xi"] = (xi)
            self.profiles["v"] = (v)
            self.profiles["xish"] = sh[0]
            self.profiles["vsh"] = sh[1]
            w_over_wn = self.w_over_wn(v_w+1.0e-8,v_w)
            self.profiles["w_over_wn"] = (w_over_wn)
            self.profiles["alpha_n"] = (alpha_p*w_over_wn)
            self.profiles["alpha_p"] = (alpha_p)
        else:
            self.profiles["type"] = (np.nan)
            self.profiles["v_plus"] = (np.nan)
            self.profiles["v_minus"] = (np.nan)
            self.profiles["xi"] = (np.nan)
            self.profiles["v"] = (np.nan)
            self.profiles["xish"] = np.nan
            self.profiles["vsh"] = np.nan
            self.profiles["w_over_wn"] = (np.nan)
            self.profiles["alpha_n"] = (np.nan)
            self.profiles["alpha_p"] = (np.nan)
        return self.profiles

    def interp(self,x,y,k=3):
        #calls the interpolation class at the top
        return interp(x,y,k=k)

    def plot_v_plus(self,alpha_plus,**plot):
        import matplotlib.pylab as plt
        v_minus = np.linspace(0.01,1,100)

        vp = self.v_plus(alpha_plus,v_minus)

        vp_0 = np.array(vp)[0,...]
        vp_1 = np.array(vp)[1,...]

        plt.plot(v_minus,vp_0,color='blue',**plot)
        plt.plot(v_minus,vp_1,color='red',**plot)
        plt.xlim(0,1.)
        plt.ylim(0,1.)
        plt.xlabel("$v_{-}$")
        plt.ylabel("$v_{+}$")


    def plot_velocity_profile(self,xi_i,v_i,v_f,set_zero=False,**plot):
        import matplotlib.pylab as plt
        xi,v = self.velocity_profile(xi_i,v_i,v_f)
        if(len(xi)>1 and set_zero): 
            xi0 = xi[0]
            v0 = 0
            xi = np.insert(xi,0,xi0)
            v = np.insert(v,0,v0)
            xi = np.append(xi,xi[-1])
            v = np.append(v,0)

        plt.rcParams.update({'font.size':16})
        plt.plot(xi,v,**plot)
        plt.xlim(0,1.)
        plt.ylim(0,1.)
        plt.xlabel(r"$\xi$")
        plt.ylabel(r"$v(\xi)$")

        self.plot_fronts()

    def plot_fronts(self):
        import matplotlib.pylab as plt
        xi = np.linspace(self.cs,1.)
        sh = self.shock_front(xi)
        rf = self.rarefraction_front(xi)

        plt.plot(xi,sh,color="black",linestyle="--")
        plt.plot(xi,rf,color="black",linestyle="--")
        plt.fill_between(xi,sh,rf,facecolor="grey")
