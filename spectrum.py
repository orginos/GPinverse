import numpy as np
import scipy as sc

def gevp_real(C,t0=4,t_ref=5):
    #t_ref is used a as a reference to sort eigenvectors
    #sorting is  not implemented
    Nt = C.shape[0]
    # make sure the matrix is symmetric
    mC = np.zeros_as(C)
    for t in range(Nt):
        mC[t] = 0.5*(C[t]+ C[t].T)
    L=np.empty(C.shape[0:2]) # principal correlators
    V=np.empty(C.shape) # eigenvectors
    for t in range(Nt):
        (ll,V[t])=sc.linalg.eig(mC[t],mC[t0])
        L[t]=np.real(ll)

    return (L,V)

def gevp(C,t0=4,t_ref=5):
    #t_ref is used a as a reference to sort eigenvectors
    #sorting is  not implemented
    Nt = C.shape[0]
    # make sure the matrix is symmetric
    mC = np.zeros_like(C)
    for t in range(Nt):
        mC[t] = 0.5*(C[t]+ np.conjugate(C[t].T))
    L=np.empty(C.shape[0:2]) # principal correlators
    V=np.empty(C.shape,dtype=C.dtype) # eigenvectors
    for t in range(Nt):
        (ll,V[t])=sc.linalg.eig(mC[t],mC[t0])
        L[t]=np.real(ll)

    return (L,V)


# logarithmic effective mass
def eff_mass(C,d=1):
    return np.log(C/np.roll(C,-d))

# mesonic effective mass
def eff_mass_cosh(C,d=1):
    R = (np.roll(C,d) + np.roll(C,-d))/(2.0*C)
    #print(R)
    m = np.zeros(R.shape)
    for t in range(d,R.shape[0]-d):
        if(R[t]>=1.0):
            m[t] = np.math.acosh(R[t])
    return m
