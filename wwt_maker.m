function wwt_maker(K,Llx,tf)

    tic

    dt = .1; 
    dx = Llx/K;
    Nsteps = tf/dt;
    n = 8;
    Kmask = K/2;
    KT = 2*K;
    KTT = KT^2;
    f0c = KTT*2.1e-3; 
    nuh = 2e-6;
    nul = 1e-18;
    Xmesh = linspace(-Llx,Llx,KT+1);
    Xmesh = Xmesh(1:KT)';
    
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

    f0 = zeros(KT);
    
    ksq = sqrt( ( (kron(ones(KT,1),(-K+1:K)')).^2 + (kron((-K+1:K)',ones(KT,1))).^2) );
    indsl = ksq >= Kl;
    indsh = ksq <= Kh;
    indsc = logical(indsl.*indsh);
    f0(indsc) = 1;
    f0 = reshape(f0,KT,KT);
    f0 = f0c*fftshift(f0);
   
    un = zeros(KT^2,1);
    uavg = zeros(KT^2,1);
    Ncnt = [];
    
    Nstart = 1e3;
    Nvstart = 5e3;
    Nint = 1e2;
    acnt = 0;
    uvels = zeros(KT,KT,3);
    vvels = zeros(KT,KT,3);
    dxr = dx/100;
    dtl = 2*dt;
    Ktot = KT + 2*(KT-2) + 2;
    xpaths = zeros(Ktot);
    ypaths = zeros(Ktot);
    Xpts = zeros(Ktot,1);
    Xrght = Xmesh(1:KT-1)+dxr;
    Xlft = Xmesh(2:KT)-dxr;
    Xpts(1:3:Ktot) = Xmesh;
    Xpts(2:3:Ktot) = Xrght;
    Xpts(3:3:Ktot) = Xlft;
    [xpaths(:,:),ypaths(:,:)] = meshgrid(Xpts);
    tftle = 0;
    
    for jj=1:Nsteps
        k1 = dt*nonlin(un,f0,KT);
        k2 = dt*nonlin(Eop.*(un+k1),f0,KT);
        un = Eop.*(un+k1/2) + k2/2;
        if jj>=Nstart 
            uphys = ifft2(reshape(un,KT,KT));            
            if jj>=Nvstart
                fac = conj(uphys)./(uphys.^2+1e-5);
                Dux = ifft2(reshape(Dx.*un(:),KT,KT));
                Duy = ifft2(reshape(Dy.*un(:),KT,KT));
                cind = mod(jj-Nvstart,3)+1;
                uvels(:,:,cind) = imag(Dux.*fac);
                vvels(:,:,cind) = imag(Duy.*fac);        
                if cind==3            
                    [xpaths,ypaths] = ftle_finder_rk4(xpaths,ypaths,uvels,vvels,Xmesh,dtl);
                    tftle = tftle + dtl;
                end
            end
            if mod(jj,Nint)==0
                uavg = uavg + abs(un.*conj(un))/KT^4;
                uphys = ifft2(reshape(un,KT,KT));
                nint = sum(sum(real(uphys.*conj(uphys))))*(1/KT)^2;
                Ncnt = [Ncnt nint];
                acnt = acnt + 1;
            end
        end
    end
    sigfield = ftle_finder_sub_grid(xpaths,ypaths,dxr,K,tftle);    
    
    if acnt > 0
        uavg = fftshift(reshape(uavg/acnt,KT,KT));
        [krad,kavg] = mat_avg(uavg,K);
        figure(1)
        plot(log10(pi*krad/Llx),log10(2*pi*(pi*krad/Llx).*kavg),'k-','LineWidth',2)
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$\log_{10}|k|$','Interpreter','LaTeX','FontSize',30)
        ylabel('$N(|k|)$','Interpreter','LaTeX','FontSize',30)
        
        figure(2)
        plot(dt*Nint*(1:acnt)+dt*Nstart,log10(Ncnt),'k-','LineWidth',2)
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$\tilde{t}$','Interpreter','LaTeX','FontSize',30)
        ylabel('$\log_{10} N_{p}$','Interpreter','LaTeX','FontSize',30)      
    end
    
    ufin = ifft2(reshape(un,KT,KT));
    
    figure(3)
    imagesc(Xmesh,Xmesh,abs(ufin))
    h = set(gca,'FontSize',30);
    set(h,'Interpreter','LaTeX')
    xlabel('$x$','Interpreter','LaTeX','FontSize',30)
    ylabel('$y$','Interpreter','LaTeX','FontSize',30)
    
    figure(4)
    imagesc(Xmesh,Xmesh,angle(ufin))
    h = set(gca,'FontSize',30);
    set(h,'Interpreter','LaTeX')
    xlabel('$x$','Interpreter','LaTeX','FontSize',30)
    ylabel('$y$','Interpreter','LaTeX','FontSize',30)    
    
    figure(5)
    imagesc(Xmesh(2:KT-1),Xmesh(2:KT-1),sigfield)
    h = set(gca,'FontSize',30);
    set(h,'Interpreter','LaTeX')
    xlabel('$x$','Interpreter','LaTeX','FontSize',30)
    ylabel('$y$','Interpreter','LaTeX','FontSize',30)    
    
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
    %inds = -K+1:K;
    
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
