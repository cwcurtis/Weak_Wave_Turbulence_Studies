import numpy as np
import matplotlib.pyplot as plt
import my_fft2 as fft2
from numba import jit
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer

Pi = np.pi


def nonlin(u,f0,KT,ftrans):
    #uphys = np.fft.ifft2(u.reshape(KT,KT))
    #unl = -1j*np.fft.fft2(uphys*uphys*uphys.conj())
    
    uphys = ftrans.ifft2(u.reshape(KT,KT))
    unl = -1j*ftrans.fft2(uphys*uphys*uphys.conj())
    
    phik = -1j*np.random.uniform(0.,2.*Pi)
    forc = f0*np.exp(phik)
    return (unl+forc).flatten() #forcing in that range

@jit("c16[:](c16[:,:],c16[:],int8,int8,f8)")
def time_stepper(f0,Eop,KT,Nsteps,dt):
    KTT = KT**2
    un = np.zeros(KTT,dtype=np.complex128) # Initialize the solution vector and associated parts of multi-step solver.
    uavg = np.zeros(KTT,dtype=np.float64)
    ftrans = fft2.my_fft2(KT)
    iterav = 10
    avcnt = 0
    
    for jj in xrange(Nsteps): # Run the time stepper i.e. how we get from time t_n = ndt to time t_(n+1)
        k1 = dt*nonlin(un,f0,KT,ftrans) 
        k2 = dt*nonlin(Eop*(un+k1),f0,KT,ftrans)
        un = Eop*(un+.5*k1) + .5*k2
        if Nsteps > 2000 and np.mod(jj,iterav)==0: # Do not average over temporally transient phenomena
            uavg += (np.abs(un*un.conj())).real
            avcnt += 1
    if avcnt > 0:
        uavg /= float(avcnt)
        return np.concatenate((un,uavg))
    else:
        return un

def mat_avg(M,K):
    
    KT = 2*K
    
    dk = Pi/K
    krad = np.linspace(dk,Pi,K)
    kavg = np.zeros(krad.shape)
    
    inds = np.arange(0,KT)
    indsr = np.kron(inds,np.ones(KT))
    indsc = np.kron(np.ones(KT),inds)
       
    Mrads = Pi - dk/np.sqrt(2.)*np.sqrt(indsr**2. + indsc**2.)
    Mflt = M.flatten()
    
    for ll in xrange(0,K-1):
        indsh = Mrads < krad[ll+1]
        indsl = Mrads >= krad[ll]
        indsc = indsh*indsl
        totparts = np.sum(indsc)
        if totparts > 0:
            kavg[ll] = np.sum(Mflt[indsc])/totparts
            
    return [krad,kavg]

def nls_solver(K,Llx,tf):
    
    # Total number of modes in simulation is KT=2*K
    # Domain is over [-Llx,Llx]x[-Llx,Llx]
    # Simulation is run to time tf
    
    dt = .1 #time-step as in Nazarenko paper
    Nsteps = int(np.round(tf/dt))
    n = 8.
    f0c = 2.1e-3 #forcing coefficient
    nuh = 2e-6
    nul = 1e-18
    Kmask = K/2
    KT = 2*K
    KTT = KT**2
    
    Xmesh = np.linspace(-Llx,Llx,KT,endpoint=False)
    Xxmesh, Yymesh = np.meshgrid(Xmesh, Xmesh)

    Dds = 1j*Pi/Llx*np.concatenate((np.arange(K+1),np.arange(-K+1,0,1,)),0)
    Dy = np.kron(Dds,np.ones(KT))
    Dx = np.kron(np.ones(KT),Dds)
    Dx2 = Dx**2.
    Dy2 = Dy**2.
    
    hmask = np.zeros((KT,KT),dtype=np.complex128)
    hmask[Kmask:KT-Kmask,:] = 1. + 1j*0.
    hmask[:,Kmask:KT-Kmask] = 1. + 1j*0.
    
    #Dhyp = ((-(Dx2+Dy2))**(n))*hmask.flatten()
    Dhyp = (-(Dx2+Dy2))**(n)
    rDhyp = np.zeros(KTT,dtype=np.complex128)
    rDhyp[1:] = 1./Dhyp[1:]
    Lap = 1j*(Dx2 + Dy2) - nul*rDhyp - nuh*Dhyp
    #Lap = 1j*(Dx2 + Dy2) - nuh*Dhyp
    
    Eop = np.exp(dt*Lap)
    
    f0 = np.ones((KT,KT),dtype=np.complex128) # Build out the forcing function
    
    Kl = 4
    Kh = 6
    
    f0[:(K-1-Kh),:] = 0. + 1j*0.
    f0[(K+Kh):,:] = 0. + 1j*0.
    f0[:,:(K-1-Kh)] = 0. + 1j*0.
    f0[:,(K+Kh):] = 0. + 1j*0.
       
    for ll in xrange(K-Kl,K-1+Kl):
        f0[K-Kl:K-1+Kl,ll] = 0. + 1j*0.    
    
    f0 = f0c*np.fft.fftshift(f0)
    
    usol = time_stepper(f0,Eop,KT,Nsteps,dt)
    un = usol[:KTT]
    flag = False
    if usol.size > KTT:
        uavg = usol[KTT:]
        flag = True
        
    ufin = np.fft.ifft2(un.reshape(KT,KT))
    reprt = ufin.real
    mag = np.abs(ufin)
    
    fig = plt.figure(figsize=(8, 6))
    plt.pcolor(Xxmesh, Yymesh, mag, cmap='RdBu')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("$|\psi|$")
    plt.colorbar()
    plt.show()
    
    if flag:
        uavg = np.fft.fftshift(uavg.reshape(KT,KT))
        avres = mat_avg(uavg,K)
      
        fig = plt.figure(figsize=(8, 6))
        plt.plot(np.log10(np.abs(avres[0])),np.log10(np.abs(avres[1])))
        plt.xlabel("$k$")
        plt.ylabel("$n_{k}$")
        plt.show()
    
start = timer()
nls_solver(128,10,10000.) #negative sigma - defocusing case (dark NLS) #nu=#nu=2*10^-6 #10^5 Nsteps #tf=10000
end = timer()
ttot = np.str(end - start)
print('Elpased time is ' + ttot + 'sec')