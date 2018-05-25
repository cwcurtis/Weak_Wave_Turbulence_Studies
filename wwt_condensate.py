import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer

Pi = np.pi

def nonlin(u,f0,KT):
    uphys = np.fft.ifft2(u.reshape(KT,KT))   #what if I make phi(k) a variable like uphys
    unl = -1j*np.fft.fft2(uphys*uphys*uphys.conj())
    phik = -1j*np.random.uniform(0,2.*Pi)
    forc = f0*np.exp(phik)
    return (unl+forc).flatten() #forcing in that range

def mat_avg(M,K):
    
    KT = 2*K
    
    dk = Pi/K
    krad = np.linspace(dk,Pi,K)
    kavg = np.zeros(krad.shape)
    
    Mtl = M[0:K-1,0:K-1]
    Mtr = np.fliplr(M[0:K-1,K:KT-1])
    Mbl = np.flipud(M[K:KT-1,0:K-1])
    Mbr = np.flipud(np.fliplr(M[K:KT-1,K:KT-1]))
    Mtl += Mtr + Mbl + Mbr
    Mavg = np.triu(Mtl,0)+(np.tril(Mtl,-1)).T
    inds = np.triu_indices(K-1)
    Mrem = Mavg[inds]
    Mrads = Pi - dk/np.sqrt(2.)*np.sqrt(inds[0]**2. + inds[1]**2.)
    
    for ll in xrange(0,K-1):
        indsh = Mrads < krad[ll+1]
        indsl = Mrads >= krad[ll]
        indsc = indsh*indsl
        totparts = np.sum(indsc)
        if totparts > 0:
            kavg[ll] = np.sum(Mrem[indsc])/(8*totparts)
            
    return [krad,kavg]

def nls_solver(K,Llx,tf):
    
    # Total number of modes in simulation is KT=2*K
    # Domain is over [-Llx,Llx]x[-Llx,Llx]
    # Simulation is run to time tf
    
    dt = .1 #time-step as in Nazarenko paper
    Nsteps = int(np.round(tf/dt))
    #Nsteps = 110000 #from updated paper
    n = 8.
    f0c = 2.1e-3 #forcing coefficient
    nuh = 2e-18
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
    #Lap = 1j*(Dx2 + Dy2) - nul*rDhyp - nuh*Dhyp
    Lap = 1j*(Dx2 + Dy2) - nuh*Dhyp
    
    Lop = 1./(np.ones(KTT)-3.*dt/4.*Lap)
    
    f0 = np.zeros((KT,KT),dtype=np.complex128) # Build out the forcing function
    Klow = 4
    Khigh = 6
    for ll in xrange(Klow,Khigh+1):
        f0[Klow:Khigh,ll] = f0c
        f0[Klow:Khigh,KT-ll] = f0c
        f0[KT-Khigh:KT-Klow,ll] = f0c
        f0[KT-Khigh:KT-Klow,KT-ll] = f0c
        
    wn = np.zeros(KTT,dtype=np.complex128) # Initialize the solution vector and associated parts of multi-step solver.
    wnm1 = np.zeros(KTT,dtype=np.complex128)
    wnp1 = np.zeros(KTT,dtype=np.complex128)

    nln = np.zeros(KTT,dtype=np.complex128)
    nlnm1 = np.zeros(KTT,dtype=np.complex128)
    nlnm2 = np.zeros(KTT,dtype=np.complex128)
    nlnm3 = np.zeros(KTT,dtype=np.complex128)

    uavg = np.zeros(KTT,dtype=np.float64)
    
    iterav = 10
    avcnt = 0
    
    for jj in xrange(Nsteps): # Run the time stepper i.e. how we get from time t_n = ndt to time t_(n+1)
        nln[:] = nonlin(wn,f0,KT) # Computation of nonlinearity
        wnp1[:] = Lop*(wn + wnm1/3. + dt*(55./24.*nln - 59./24.*nlnm1 + 37./24.*nlnm2 - 3./8.*nlnm3)) - wnm1/3.
        wnm1[:] = wn
        wn[:] = wnp1
    
        nlnm3[:] = nlnm2
        nlnm2[:] = nlnm1
        nlnm1[:] = nln
        
        if Nsteps > 1000 and np.mod(jj,iterav)==0: # Do not average over temporally transient phenomena
            uavg += np.abs(wn*wn.conj())
            avcnt += 1
        
    uavg /= float(avcnt)    
    uavg = np.fft.fftshift(uavg.reshape(KT,KT))
    avres = mat_avg(uavg,K)
    
    ufin = np.fft.ifft2(wnp1.reshape(KT,KT))
    reprt = ufin.real
    mag = np.abs(ufin)
    
    fig = plt.figure(figsize=(8, 6))
    plt.pcolor(Xxmesh, Yymesh, mag, cmap='RdBu')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("$|\psi|$")
    plt.colorbar()
    plt.show()
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(np.log10(np.abs(avres[0])),np.log10(np.abs(avres[1])))
    plt.xlabel("$k$")
    plt.ylabel("$n_{k}$")
    plt.show()
    
start = timer()
nls_solver(128,10,1000) #negative sigma - defocusing case (dark NLS) #nu=#nu=2*10^-6 #10^5 Nsteps #tf=10000
end = timer()
ttot = np.str(end - start)
print('Elpased time is ' + ttot + 'sec')