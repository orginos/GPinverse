import numpy as np
import scipy.integrate as integrate
from torch.special import gammaln
import torch as tr
from orthogonal_poly import legendre_01

class FE_Integrator:
    def __init__(self,x):
        self.N = x.shape[0]
        xx = np.append(x,2.0*x[self.N-1] - x[self.N-2])
        self.x = np.append(0,xx)
        self.eI = 0

        self.Norm = np.empty(self.N)
        for i in range(self.N):
            self.Norm[i] = self.ComputeI(i, lambda x : 1)
            
    def pulse(self,x,x1,x2):
        return np.heaviside(x-x1,0.5)* np.heaviside(x2-x,0.5)
    
    def f(self,x,i):
 ##       if(i==0):
 ##           R=(x- self.x[2])/(self.x[1] -self.x[2])*np.heaviside(x-self.x[0],1.0)* np.heaviside(self.x[2]-x,0.5)

            #R= self.pulse(x,self.x[0],self.x[1])
            #R= (x- self.x[0])/(self.x[1] -self.x[0])*self.pulse(x,self.x[0],self.x[1])
            #R+=(x- self.x[2])/(self.x[1] -self.x[2])*self.pulse(x,self.x[1],self.x[2])
            #R+=(x- self.x[1])/(self.x[0] -self.x[1])*self.pulse(x,self.x[0],self.x[1]) 
##            return R
        ii=i+1
        R = (x- self.x[ii-1])/(self.x[ii] -self.x[ii-1])*self.pulse(x,self.x[ii-1],self.x[ii  ])
        R+= (x- self.x[ii+1])/(self.x[ii] -self.x[ii+1])*self.pulse(x,self.x[ii  ],self.x[ii+1])

       # if(i==0):
       #     R *=2
        return R
    
    def set_up_integration(self,Kernel = lambda x: 1):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.ComputeI(i,Kernel)
        return res
   
    # assume symmetrix function F(x,y) = F(y,x)
    # for efficiency 
    def set_up_dbl_integration(self,Kernel = lambda x,y: 1):
        res = np.empty([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                res[i,j] = self.ComputeIJ(i,j,Kernel)
                res[j,i]  = res[i,j]
        #res[0,:] *=2
        #res[:,0] *=2
        return res
        
    def ComputeI(self,i,Kernel):
        I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[i], self.x[i+2])
        self.eI += eI
        return I
    
    def ComputeIJ(self,i,j,Kernel):
        I,eI = integrate.dblquad(lambda x,y: self.f(x,i)*Kernel(x,y)*self.f(y,j), self.x[j], self.x[j+2],self.x[i], self.x[i+2])
        self.eI += eI
        return I
    
    
# quadratic finite elements are more complicated...
# ... but now it works!
# also I should try the qubic ones too
class FE2_Integrator:
    def __init__(self,x):
        self.N = x.shape[0]
        xx = np.append(x,[2.0*x[self.N-1] - x[self.N-2], 3.0*x[self.N-1]-2*x[self.N-2],0] )
        #self.x = np.append([-x[0],0],xx)
        self.x = np.append(0,xx)
        self.eI = 0

        self.Norm = np.empty(self.N)
        for i in range(self.N):
            self.Norm[i] = self.ComputeI(i, lambda x : 1)
            
    def pulse(self,x,x1,x2):
        return np.heaviside(x-x1,0.5)* np.heaviside(x2-x,0.5)
    
    def f(self,x,i):
        R=0.0
        if(i==0):
            #R=self.pulse(x,self.x[0],self.x[1])
            #R=self.pulse(x,self.x[1],self.x[2])
        #    R+=(x- self.x[2])/(self.x[1] -self.x[2])*self.pulse(x,self.x[1],self.x[2])

            R+=(x- self.x[2])*(x- self.x[3])/((self.x[1] -self.x[3])*(self.x[1] -self.x[2]))**np.heaviside(x-self.x[0],1.0)* np.heaviside(self.x[3]-x,0.5)
            #self.pulse(x,self.x[0],self.x[3])
            return R
        ii =i+1
        if(ii%2==0):
            R  += (x- self.x[ii-1])*(x- self.x[ii+1])/((self.x[ii] -self.x[ii+1])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-1],self.x[ii+1])
            return R
        else:
            R += (x- self.x[ii-2])*(x- self.x[ii-1])/((self.x[ii] -self.x[ii-2])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-2],self.x[ii  ])
            R += (x- self.x[ii+1])*(x- self.x[ii+2])/((self.x[ii] -self.x[ii+2])*(self.x[ii] -self.x[ii+1]))*self.pulse(x,self.x[ii  ],self.x[ii+2])
            return R
    
        return R
    
    def set_up_integration(self,Kernel = lambda x: 1):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.ComputeI(i,Kernel)
        return res
        
    # assume symmetrix function F(x,y) = F(y,x)
    # for efficiency 
    def set_up_dbl_integration(self,Kernel = lambda x,y: 1):
        res = np.empty([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                res[i,j] = self.ComputeIJ(i,j,Kernel)
                res[j,i]  = res[i,j]
        return res
    
    def ComputeI(self,i,Kernel):
        #if(i==0):
        #    I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,0), self.x[0], self.x[3])
        #    self.eI += eI
        #    return I
        ii=i+1
        if(ii%2==0):
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[ii-1], self.x[ii+1])
            self.eI += eI
        else:
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[ii-2], self.x[ii+2])
            self.eI += eI
        return I
    
    def ComputeIJ(self,i,j,Kernel):
        # I need to fix the i=0 case
        ii=i+1
        jj=j+1
        if(ii%2==0):
            xx = (self.x[ii-1], self.x[ii+1])
        else:
            xx = (self.x[ii-2], self.x[ii+2])
        if(jj%2==0):
            yy = (self.x[jj-1], self.x[jj+1])
        else:
            yy = (self.x[jj-2], self.x[jj+2])
        
        I,eI = integrate.dblquad(lambda x,y: self.f(x,i)*Kernel(x,y)*self.f(y,j), yy[0], yy[1],xx[0], xx[1])
        self.eI += eI

        return I

def interp(x,q,fe):
    S = 0*x
    for k in range(fe.N):
        S+= fe.f(x,k)*q[k]
    return S

class simple_PDF():
    def __init__(self,a,b,g): 
        self.a=a
        self.b=b
        self.g=g
        self.r = 1.0
        self.F = lambda y: (y**a*(1-y)**b*(1 + g*np.sqrt(y)))/self.r
        self.r,e = integrate.quad(self.F,0.0,1.0)  

def simplePDFnormed(x,a,b):
    return x**a*(1-x)**b*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

def very_simplePDFnormed(x,b):
    return (1-x)**b*tr.exp(gammaln(b+2) - gammaln(b+1))

def KrbfMat(x,s,w):
    return s*s*tr.exp(-0.5*((x.view(1,x.shape[0]) - x.view(x.shape[0],1))/w)**2)

def KrbfMat(x,s,w):
    return s*s*tr.exp(-0.5*((x.view(1,x.shape[0]) - x.view(x.shape[0],1))/w)**2)

def KlegeMat(x,*d):
    deg = len(d) -1
    L = legendre_01(x,deg)
    for i in range(len(d)):
        L[:,i]*=d[i]
    return L@tr.diag((2*tr.arange(0,deg+1,dtype=x.dtype)+1))@L.T

def KlegeMat_v0(x,d):
    deg = d.shape[0]-1
    L = legendre_01(x,deg)
    return L@tr.diag(d**2*(2*tr.arange(0,deg+1,dtype=x.dtype)+1))@L.T

class PolyKer():
    def __init__(self,degree=10):
        self.deg =degree

    def KerMat(self,x,s):
        L = legendre_01(x,self.deg)
        return L@tr.diag(s*s*(2*tr.arange(0,self.deg+1,dtype=x.dtype)+1))@L.T
        
class splitRBFker():
    def __init__(self,sp,scale=1):
        self.sp =sp
        self.scale = scale
    def KerMat(self,x,s1,w1,s2,w2):
        K2 = KrbfMat(x,s2,w2) # linear
        K1 = KrbfMat(tr.log(x),s1,w1)
        sig = tr.diag(tr.special.expit(self.scale*(x-self.sp)))
        sigC = tr.eye(x.shape[0])-sig
        return sig@K2@sig + sigC@K1@sigC
        
    
# General GP V1       
class GaussianProcess():
    def __init__(self,x_grid,V,Y,Gamma,Pd= lambda x : 2.*(1.-x) ,Ker = lambda x: tr.outer(x,x),**args):
        self.x_grid = tr.tensor(x_grid)
        self.N = x_grid.shape[0]
        self.V = tr.tensor(V)
        self.Y = tr.tensor(Y)
        self.Gamma = tr.tensor(Gamma) # data covariance
        self.Pd = Pd # the default model function. It must work in torch for training?
        self.Ker = Ker # the Kernel
        self.pd_args = tuple([tr.tensor([a]) for a in args["Pd_args"]])
        self.ker_args = tuple([tr.tensor([a]) for a in args["Ker_args"]])
        self.Npd_args = len(self.pd_args)
        self.Nker_args = len(self.ker_args)
        
    def ComputePosterior(self): # computes the covariance matrix of the posterior
        K = self.Ker(self.x_grid,*self.ker_args)
        Pd = self.Pd(self.x_grid,*self.pd_args)
        Chat = self.Gamma + self.V@K@self.V.T
        iChat = tr.linalg.inv(Chat)
        VK = self.V@K
        #print(K)
        self.CpMat = K - VK.T@iChat@VK
        self.Pm = Pd +VK.T@iChat@(self.Y-self.V@Pd)
        return self.Pm,self.CpMat
    
    def nlpEvidence(self,p_x,k_x):
        # p = [a,b,sig,w]
        pd_args = tuple(p_x)
        k_args = tuple(k_x)
        K = self.Ker(self.x_grid,*k_args)
        Pd = self.Pd(self.x_grid,*pd_args)
        Chat = self.Gamma + self.V@K@self.V.T
        iChat = tr.linalg.inv(Chat)
        #print(Y,self.V,Pd)
        D = self.Y - self.V@Pd
        # no need for D.T@iChat@D D.@iChat@D does the job...
        nlp = 0.5*(D@iChat@D + tr.log(tr.linalg.det(Chat)))
        return nlp
    
    def train(self,Nsteps=10,lr=0.1,mode="all"):
        p_x=tr.tensor(self.pd_args ,requires_grad=True)
        k_x=tr.tensor(self.ker_args,requires_grad=True)
        optim = tr.optim.Adam([p_x,k_x], lr=lr) # train everything
        if mode=="kernel" :
            optim = tr.optim.Adam([k_x], lr=lr) # train only kernel
            print("Training kernel only")
        elif mode=="mean" :
            print("Training mean only")
            optim = tr.optim.Adam([p_x], lr=lr) # train only default model
        else:
            print("Training everything")
        losses = []
        for i in range(Nsteps):
            optim.zero_grad()
            loss = self.nlpEvidence(p_x,k_x)
            loss.backward()
            optim.step()
            losses.append(loss.detach().item()+self.Gamma.shape[0]/2.0*np.log(2.0*np.pi))
        self.pd_args=tuple(p_x.detach())
        self.ker_args=tuple(k_x.detach())
        return losses
   
if __name__ == "__main__":
    import argparse as arg
    import matplotlib.pyplot as plt
    import torch as tr
    def test_single():
        #Nx = 128
        #Nx = 256
        #Nx = 1024
        Nx=64
        x = np.concatenate((np.logspace(np.log10(1e-5),np.log10(1e-2),Nx),np.linspace(1.001e-2,1,Nx)))
        pdf = simple_PDF(-0.3,2,0.1)
        q_d = pdf.F(x)
        fe = FE_Integrator(x)
        fe2 = FE2_Integrator(x)
        S = fe.set_up_integration(Kernel=lambda x: np.sqrt(x))
        S2 = fe2.set_up_integration(Kernel=lambda x: np.sqrt(x))
        Ife = S@q_d
        Ife2 = S2@q_d
        Ie,_ = integrate.quad(lambda x : np.sqrt(x)*pdf.F(x),0,1)
        print("The integral is: ",Ie)
        print("Linear Interpolator integral and diff:", Ife, Ife-Ie)
        print("Quadratic Interpolator integral and diff:", Ife2, Ife2-Ie)
        
    def test_double():
        Nx=64
        x = np.concatenate((np.logspace(np.log10(1e-5),np.log10(1e-2),Nx),np.linspace(1.001e-2,1,Nx)))
        pdf = simple_PDF(-0.3,2,0.1)
        q_d = pdf.F(x)
        fe1 = FE_Integrator(x)
        fe2 = FE2_Integrator(x)
        GP = simpleGP(width=0.3)
        
        SS1 = fe1.set_up_dbl_integration(lambda x,y : GP.K(x,y))
        SS2 = fe2.set_up_dbl_integration(lambda x,y : GP.K(x,y))

        IIe,_ = integrate.dblquad(lambda x,y : pdf.F(x)*GP.K(x,y)*pdf.F(y),0,1,0,1)
        II1 = q_d@SS1@q_d
        II2 = q_d@SS2@q_d
        print("Linear: ",II1,IIe-II1)
        print("Quadratic: ",II2,IIe-II2)
        print("Exact: ",IIe)
    
    def test_prior():
        Nx=256
        x_grid = np.concatenate((np.logspace(-12,-1,np.int32(Nx/2)),np.linspace(0.1+1e-4,1,np.int32(Nx/2))))
        #x_grid = np.logspace(np.log10(1e-4),np.log10(1),Nx)
        #x_grid = np.linspace(1e-8,1,Nx)
        fe = FE2_Integrator(x_grid)
        lam = 1e-5   # soften the constrants
        lam_c = 1e-4
        B0 = fe.set_up_integration(Kernel=lambda x: 1)
        B1 = np.zeros_like(B0) 
        B1[-1] = 1.0 # x=1 is at the end...
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:]))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        #Y = np.concatenate(([1,0],mM))
        Y = np.array([1.,0.])
        myGP    = GaussianProcess(x_grid,V,Y,Gamma,Pd=simplePDFnormed, Ker=KrbfMat,Pd_args=(-0.3,2.0),Ker_args=(10.0,0.6) )
        Kmat = myGP.Ker(myGP.x_grid,*myGP.ker_args)
        fig, ax = plt.subplots()
        im = ax.imshow(Kmat.numpy())
        plt.show()
        Chat = myGP.Gamma + myGP.V@Kmat@myGP.V.T
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(Chat)
        plt.show()
        p,Cp = myGP.ComputePosterior()
        Cp = 0.5*(Cp+Cp.T) +1e-7*tr.eye(Cp.shape[0])
        Mc=tr.distributions.MultivariateNormal(p,covariance_matrix=Cp)
        fig3, ax3 = plt.subplots()
        im3 = ax3.imshow(Cp)
        plt.show()
        plt.plot(x_grid,p,x_grid,myGP.Pd(myGP.x_grid,*myGP.pd_args).numpy())
        plt.ylim([-0.1,4])
        plt.show()
        pdfMc = Mc.sample((100,))
        _=plt.plot(x_grid,pdfMc.T.numpy(),alpha=0.1,color='orange')
        _=plt.plot(x_grid,p.numpy(),color='red',label='reconstructed')
        #_=plt.plot(x[1:],q[1:],color='blue',label='original')
        plt.legend()
        _=plt.title(r"Prior samples of $\mathcal{P}(x)$ ")
        plt.ylim([-2,5])
        plt.show()
        S = fe.set_up_integration(Kernel=lambda x: 1)
        print("Test mean   norm      : ",      S@(myGP.Pd(myGP.x_grid,*myGP.pd_args)).numpy())
        print("Test sample norm      : ",      S@pdfMc.numpy()[10,:])
        print("Test sample constraint: ",pdfMc.numpy()[10,Nx-1])
        plt.plot(x_grid,myGP.Pd(myGP.x_grid,*myGP.pd_args).numpy())
        plt.ylim([-0.1,4])
        plt.show()
        ii,ee =integrate.quad(lambda x : myGP.Pd(tr.tensor(x),*myGP.pd_args),0,1)
        print("\n mean norm from quad:",ii)
        
    
    def test_posterior():
        def pseudo_data(nu,a,b,g,da,db,dg,N):
            sa = np.random.normal(a,da,N)
            sb = np.random.normal(b,db,N)
            sg = np.random.normal(g,dg,N)
    
            D = np.zeros((N,nu.shape[0]))
            Norm=1.0
            for k in range(N):
                for i in range(nu.shape[0]):
                    F =  lambda y: y**sa[k]*(1-y)**sb[k]*(1 + sg[k]*np.sqrt(y)-0.1*y)*np.cos(nu[i]*y) 
                    r,e = integrate.quad(F,0.0,1.0) 
                    D[k,i] = r
                    if i==0:
                        Norm = r
                    D[k,i] = D[k,i]/Norm
            #add additional gaussian noise to break correlations
            NN = np.random.normal(0,1e-2,np.prod(D.shape)).reshape(D.shape)
            return D+NN
        
        Nnu = 12
        nu = np.linspace(0,13,Nnu)
        #create fake data
        jM = pseudo_data(nu,-0.3,2.9,1.0,0.02,.2,0.2,1000)
        M = np.mean(jM,axis=0)
        eM = np.std(jM,axis=0)
        print("Check the zero point:",nu[0],M[0],eM[0])
        plt.errorbar(nu,M,eM,marker='o')
        plt.show()
        #chop off the nu = 0
        jM = jM[:,1:]
        n = nu[1:]
        M = np.mean(jM,axis=0)
        eM = np.std(jM,axis=0)
        
        print("jM shape: ",jM.shape)

        CovD = np.cov(jM.T)   
        CovD =(CovD + CovD.T)/2.0
        U,S,V = np.linalg.svd(CovD)
        #print("Data Cov: ",CovD)
        print("Data Cov S:",S)

        Nx=256
        x_grid = np.concatenate((np.logspace(-12,-1,np.int32(Nx/2)),np.linspace(0.1+1e-4,1,np.int32(Nx/2))))
        fe = FE2_Integrator(x_grid)
        lam = 1e-5   # soften the constrants
        lam_c = 1e-5
        B0 = fe.set_up_integration(Kernel=lambda x: 1)
        B1 = np.zeros_like(B0) 
        B1[-1] = 1.0 # x=1 is at the end...
        n # is the nu values at current z
        B = np.zeros((n.shape[0],Nx))
        for k in np.arange(n.shape[0]):
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(n[k]*x))
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        Gamma[2:,2:] = CovD
        Y = np.concatenate(([1,0],M))
        # now construct the GP
        myGP    = GaussianProcess(x_grid,V,Y,Gamma,Pd=simplePDFnormed, Ker=KrbfMat,Pd_args=(0.0,1.0),Ker_args=(20.0,0.4) )
        Kmat = myGP.Ker(myGP.x_grid,*myGP.ker_args)
        fig, ax = plt.subplots()
        im = ax.imshow(Kmat.numpy())
        plt.show()
        Chat = myGP.Gamma + myGP.V@Kmat@myGP.V.T
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(Chat)
        plt.show()
        p,Cp = myGP.ComputePosterior()
        Cp = 0.5*(Cp+Cp.T) +1e-7*tr.eye(Cp.shape[0])
        Mc=tr.distributions.MultivariateNormal(p,covariance_matrix=Cp)
        fig3, ax3 = plt.subplots()
        im3 = ax3.imshow(Cp)
        plt.show()
        plt.plot(x_grid,p,x_grid,myGP.Pd(myGP.x_grid,*myGP.pd_args).numpy())
        plt.ylim([-0.1,4])
        plt.show()
        pdfMc = Mc.sample((100,))
        plt.plot(x_grid,pdfMc.T.numpy(),alpha=0.1,color='orange')
        plt.plot(x_grid,p.numpy(),color='red',label='reconstructed')
        #_=plt.plot(x[1:],q[1:],color='blue',label='original')
        plt.legend()
        plt.title(r"Posterior samples of $\mathcal{P}(x)$ ")
        plt.ylim([-0.1,5])
        plt.show()
        S = fe.set_up_integration(Kernel=lambda x: 1)
        print("Test prior mean norm     : ",      S@(myGP.Pd(myGP.x_grid,*myGP.pd_args)).numpy())
        print("Test posterior mean norm : ",      S@p.numpy())
        print("Test sample norm         : ",      S@pdfMc.numpy()[10,:])
        print("Test sample constraint   : ",pdfMc.numpy()[10,Nx-1])
        ii,ee =integrate.quad(lambda x : myGP.Pd(tr.tensor(x),*myGP.pd_args),0,1)
        print("\n mean norm from quad:",ii)
        nn = np.linspace(0,15,128)
        iB = np.zeros((nn.shape[0],Nx))
        for k in range(nn.shape[0]):
            iB[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(nn[k]*x))
        ttQ = pdfMc.numpy()@iB.T
        mttQ = ttQ.mean(axis=0)
        plt.plot(nn,ttQ.T,color="orange",alpha=0.1)
        plt.plot(nn,mttQ,color="red",label='reconstructed')
        plt.errorbar(n,M,yerr=eM,linestyle='none',marker='o',markersize=4,label='data')
        plt.show()

        l_hist = myGP.train(2000,lr=1e-2,mode="kernel")
        print("Hyper parameters after training: (a,b,sig,w)=\n",*myGP.pd_args,*myGP.ker_args)

        plt.plot(l_hist)
        plt.xlabel('epoch')
        plt.ylabel('Evidence')
        plt.title('Training history')
        plt.show()

        hp,hCp = myGP.ComputePosterior()
        # symmetrize and regularize
        hCp = 0.5*(hCp+hCp.T)+1e-7*tr.eye(hCp.shape[0])
        hMc=tr.distributions.MultivariateNormal(hp,covariance_matrix=hCp)
        h_pdfMc = hMc.sample((100,))
        h_ttQ = h_pdfMc.numpy()@iB.T
        mh_ttQ = h_ttQ.mean(axis=0)
        plt.plot(x_grid,h_pdfMc.T.numpy(),alpha=0.1,color='orange')
        plt.plot(x_grid,hp.numpy(),color='red',label='reconstructed')
        plt.legend()
        plt.title(r"Posterior samples of $\mathcal{P}(x)$ after training ")
        plt.ylim([-0.1,5])
        plt.show()
        plt.plot(nn,ttQ.T,color="orange",alpha=0.1)
        plt.plot(nn,mttQ,color="red",label='reconstructed')
        plt.errorbar(n,M,yerr=eM,linestyle='none',marker='o',markersize=4,label='data')
        plt.plot(nn,h_ttQ.T,color="cyan",alpha=0.1)
        plt.plot(nn,mh_ttQ,color="blue",label='trained reconstruction')
        plt.legend()
        plt.title(r'Posterior samples of $\mathcal{Q}(\nu)$ after training')
        plt.show()
        
    parser = arg.ArgumentParser()
    #parser.add_argument('--help', help='foo help')
    parser.add_argument('--single',action='store_true')
    parser.add_argument('--double',action='store_true')
    parser.add_argument('--prior',action='store_true')
    parser.add_argument('--post',action='store_true')
    args = parser.parse_args()
    if(args.single):
        test_single()
    elif(args.double):
        test_double()
    elif(args.prior):
        test_prior()
    elif(args.post):
        test_posterior()
    else:
        print("Nothing to test!")
        
