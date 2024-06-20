import numpy as np
import h5py
import struct

def read_itd_bin(file):
    f=open(file, mode='rb')
    c = f.read()
    (nm,nz,Ncnf) = struct.unpack("iii", c[:12])
    print("Nnu =",nm," Nz= ",nz, "Ncnf= ",Ncnf)
    nu = np.empty((nm,nz))
    rMj = np.empty((Ncnf,nm,nz))
    iMj = np.empty((Ncnf,nm,nz))
    #rM,iM,rMe,iMe
    fpos=12
    for z in range(nz):
        for p in range(nm):
            z2=struct.unpack("d", c[fpos:fpos+8])[0]
            fpos=fpos+8
            v=struct.unpack("d", c[fpos:fpos+8])[0]
            nu[p,z]=v 
            fpos=fpos+8
                
            for k in range(Ncnf):
                d=struct.unpack("d", c[fpos:fpos+8])[0]
                fpos=fpos+8
                rMj[k,p,z] = d
            for k in range(Ncnf):
                d=struct.unpack("d", c[fpos:fpos+8])[0]
                fpos=fpos+8
                iMj[k,p,z] = d

    rM  = np.mean(rMj,axis=0)
    iM  = np.mean(iMj,axis=0)
    rMe = np.std(rMj,axis=0)*np.sqrt(Ncnf-1)
    iMe = np.std(iMj,axis=0)*np.sqrt(Ncnf-1)
    
    return nu,rMj,iMj,rM,iM,rMe,iMe
   
def read_itd_h5(file,type):
    hf = h5py.File(file, 'r')
    group = hf.get(type)
    Np=np.array(hf.get(type)).shape[0]
    Nz=np.array(group.get('pz1/jack/1/')).shape[0]
    Nj=np.array(group.get('pz1/jack/1/0/pitd')).shape[0]

    rMj = np.empty((Nj,Np,Nz),dtype=float)
    iMj = np.empty((Nj,Np,Nz),dtype=float)
    rM = np.empty((Np,Nz),dtype=float)
    iM = np.empty((Np,Nz),dtype=float)
    rMe = np.empty((Np,Nz),dtype=float)
    iMe = np.empty((Np,Nz),dtype=float)
    nu =  np.empty((Np,Nz),dtype=float)

    for p in range(1,Np+1):
        ip = p-1
        for z in range(0,Nz):
            keyR = 'pz'+str(p)+'/jack/1/'+str(z)+'/pitd'
            keyI = 'pz'+str(p)+'/jack/2/'+str(z)+'/pitd'
            rr = np.array(group.get(keyR))
            ii = np.array(group.get(keyI))
            rMj[:,ip,z] = rr[:,1] #np.array(group.get(keyR))[:,1]
            iMj[:,ip,z] = ii[:,1]
            nu[ip,z] =  rr[0,0]
            keyR = 'pz'+str(p)+'/ensemble/1/'+str(z)+'/pitd'
            keyI = 'pz'+str(p)+'/ensemble/2/'+str(z)+'/pitd'
            rr = np.array(group.get(keyR))
            ii = np.array(group.get(keyI))
            rM[ip,z] = rr[0,1] #np.array(group.get(keyR))[:,1]
            iM[ip,z] = ii[0,1]
            rMe[ip,z] = rr[0,2] #np.array(group.get(keyR))[:,1]
            iMe[ip,z] = ii[0,2]
    return nu,rMj,iMj,rM,iM,rMe,iMe 

#Colin Egerer to Everyone (4:04 PM)
#/b_b0xDA__J0_A1pP/zsep<Z>/pz<PZ>/jack/<Re,Im>/pitd
#pitd : ( 349, 2 )
#For example, (137,0): 4.71239, 0.260395,  ===>  <Ioffe-time>  <matelem for this jack>
def read_itd_h5_v2(file,type):
    hf = h5py.File(file, 'r')
    group = hf.get(type)
    Nz=np.array(hf.get(type)).shape[0]
    #print("Nz= ",Nz)
    Np=np.array(group.get('zsep0/')).shape[0]
    #print("Np= ",Np)
    
    Nj=np.array(group.get('zsep0/pz1/jack/Re/pitd')).shape[0]
    #print("Nj= ", Nj)
    
    rMj = np.empty((Nj,Np,Nz),dtype=float)
    iMj = np.empty((Nj,Np,Nz),dtype=float)
    nu =  np.empty((Np,Nz),dtype=float)

    for p in range(1,Np+1):
        ip = p-1
        for z in range(0,Nz):
            keyR = 'zsep'+str(z)+'/pz'+str(p)+'/jack/Re/pitd'
            keyI = 'zsep'+str(z)+'/pz'+str(p)+'/jack/Im/pitd'
            rr = np.array(group.get(keyR))
            #print(keyR,rr)
            ii = np.array(group.get(keyI))
            #print(keyI,ii)
            rMj[:,ip,z] = rr[:,1] #np.array(group.get(keyR))[:,1]
            iMj[:,ip,z] = ii[:,1]
            nu[ip,z] =  rr[0,0]
    rM  = np.mean(rMj,axis=0)
    rMe = np.std (rMj,axis=0)*np.sqrt(Nj-1)
    iM  = np.mean(iMj,axis=0)
    iMe = np.std (iMj,axis=0)*np.sqrt(Nj-1)
    
    return nu,rMj,iMj,rM,iM,rMe,iMe 



def read_systematic_err(file):
    f = open(file, "r")
    sigL = []
    for line in f.readlines():
        fields = line.split(' ')
        #print(fields)
        z = float(fields[0])
        p = float(fields[1])
        reE = float(fields[2])
        imE = float(fields[3])
        sigL.append([z,p,reE,imE])
    f.close()

    return sigL

if __name__ == "__main__":
    # execute only if run as a script
    sig = read_systematic_err('data/PDF/Nf2/N5/fit_sys_N5.dat')
    print(sig)
