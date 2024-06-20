import numpy as np
import tarfile

## this reader only reads the real part

def read_proplist(file):
    C=[]
    Ncnf=0
    Nt=0
    with open(file,'r') as f:
        header=1  
        for line in f: 
            #read the header
            if (header==1):
                data = line.split()
                Ncnf=int(data[0])
                Nt=int(data[1])
                header=0
            else:
                [t,re,im] = line.split()
                C.append(float(re))
                #print(t,re)
    #print(C)
    return np.reshape(C,(Ncnf,Nt))

# this reader preserves the imaginary part
def read_cproplist(file):
    C=[]
    Ncnf=0
    Nt=0
    with open(file,'r') as f:
        header=1  
        for line in f: 
            #read the header
            if (header==1):
                data = line.split()
                Ncnf=int(data[0])
                Nt=int(data[1])
                header=0
            else:
                [t,re,im] = line.split()
                C.append(float(re)+1j*float(im))
                #print(t,re)
    #print(C)
    return np.reshape(C,(Ncnf,Nt))

# this reader preserves the imaginary part
def read_cproplist_tar(targzf,file):
    tar = tarfile.open(targzf, "r:gz")
    C=[]
    Ncnf=0
    Nt=0
    f = tar.extractfile(file)
    header=1  
    for line in f: 
        #read the header
        if (header==1):
            data = line.split()
            Ncnf=int(data[0])
            Nt=int(data[1])
            header=0
        else:
            [t,re,im] = line.split()
            C.append(float(re)+1j*float(im))
            #print(t,re)
            #print(C)
    return np.reshape(C,(Ncnf,Nt))

def fold(C,s):
    (Ncnf,Nt)=C.shape
    for t in range(1,Nt):
        C[:,t] = 0.5*(C[:,t]+s*C[:,Nt-t])
    return C[:,0:int(Nt/2)+1]

# folding for complex correlators
def cfold(C):
    (Ncnf,Nt)=C.shape
    for t in range(1,Nt):
        C[:,t] = 0.5*(C[:,t]+np.conjugate(C[:,Nt-t]))
    return C[:,0:int(Nt/2)+1]

