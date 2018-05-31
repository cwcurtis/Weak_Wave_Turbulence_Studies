function wwt_maker(K,Llx,tf)

    tic

    dt = .1; 
    Nsteps = tf/dt;
    n = 8;
    f0c = 2.1e-3; 
    nuh = 2e-6;
    nul = 1e-18;
    Kmask = K/2;
    KT = 2*K;
    KTT = KT^2;

    Kl = 4;
    Kh = 6;

    Dd = 1i*pi/Llx*[0:K -K+1:-1]';
    Dx = kron(Dd,ones(KT,1));
    Dy = kron(ones(KT,1),Dd);
    Dx2 = Dx.^2;
    Dy2 = Dy.^2;
    Dhyp = (-(Dx2+Dy2)).^(n);
    iDhyp = 1./Dhyp;
    iDhyp(1) = 0;
    Lap = 1i*(Dx2+Dy2) - nuh*Dhyp - nul*iDhyp;
    Eop = exp(dt*Lap);

    %{
    f0 = ones(KT);
    f0(1:K-Kh-1,:) = 0;
    f0(K+Kh+1:end,:) = 0;
    f0(:,1:K-Kh-1) = 0;
    f0(:,K+Kh+1:end) = 0;

    for ll = K-Kl:K+Kl
        f0(K-Kl:K+Kl,ll) = 0;
    end
    %}
    
    f0 = zeros(KT);
    
    ksq = sqrt( ( (kron(ones(KT,1),(-K+1:K)')).^2 + (kron((-K+1:K)',ones(KT,1))).^2)/2 );
    indsl = ksq >= Kl;
    indsh = ksq <= Kh;
    indsc = logical(indsl.*indsh);
    f0(indsc) = 1;
    f0 = reshape(f0,KT,KT);
    f0 = f0c*fftshift(f0);
   
    %{
    un = abs(randn(KT));
    phase = exp(1i*2*pi*rand(KT));
    un = un.*phase;
    un = 1e-4*un/norm(un);
    un = un(:);
    %}
    un = zeros(KT^2,1);
    uavg = zeros(KT^2,1);
    Nstart = 10000;
    Nint = 1000;
    acnt = 0;
    
    for jj=1:Nsteps
        k1 = dt*nonlin(un,f0,KT);
        k2 = dt*nonlin(Eop.*(un+k1),f0,KT);
        un = Eop.*(un+k1/2) + k2/2;
        
        if jj> Nstart && mod(jj,Nint)==0
            uavg = uavg + KT^4*abs(un.*conj(un));
            acnt = acnt + 1;
        end
    end
    
    if acnt > 0
        uavg = fftshift(reshape(uavg/acnt,KT,KT));
        [krad,kavg] = mat_avg(uavg,K);
        figure(1)
        plot(log10(pi*krad/Llx),log10(2*Llx*krad.*kavg),'k-','LineWidth',2)
    end
    
    Xmesh = linspace(-Llx,Llx,KT+1);
    Xmesh = Xmesh(1:KT)';
    ufin = ifft2(reshape(un,KT,KT));
    
    figure(2)
    imagesc(Xmesh,Xmesh,abs(ufin))
    
    figure(3)
    imagesc(Xmesh,Xmesh,angle(ufin))
        
    toc
end

function uout = nonlin(un,f0,KT)
    uphys = ifft2(reshape(un,KT,KT));
    unl = -1i*fft2(uphys.*uphys.*conj(uphys));
    phi = exp(-1i*2*pi*rand);
    unl = unl + phi*f0;
    uout = unl(:);
end

function [krad,kavg] = mat_avg(M,K)
    KT = 2*K;
    inds = -K+1:K;
    figure(4)
    surf(inds,inds,log10(M),'LineStyle','none')
    
    krad = (1:K)';
    kavg = M(K,K+1:KT)';
    %{
    indsr = kron(ones(KT,1),inds');
    indsc = kron(inds',ones(KT,1));
    M = M(:);
    kavg = zeros(length(krad),1);
    mrads = sqrt((indsr.^2+indsc.^2)/2);
    for jj=1:length(krad)-1
       indsl = mrads >= krad(jj);
       indsh = mrads < krad(jj+1);
       indsc = logical(indsl.*indsh);
       totparts = sum(indsc);
       
       if totparts > 0
          kavg(jj) = sum(M(indsc))/totparts; 
       end
    end
    %}
    
end
