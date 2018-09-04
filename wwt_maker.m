function wwt_maker(K,Llx,tf)

    tic

    dt = .1; 
    dx = Llx/K;
    n = 8;
    Kmask = K/2;
    KT = 2*K;
    KTT = KT^2;

    f0c = KTT*2.1e-3; 
    nuh = 2e-6;
    nul = 0;

    Xmesh = linspace(-Llx,Llx,KT+1);
    Xmesh = Xmesh(1:KT)';
    dreg = 1e-2;
    
    Kl = 4;
    Kh = 6;

    Dd = 1i*pi/Llx*[0:K -K+1:-1]';
    Dx = kron(Dd,ones(KT,1));
    Dy = kron(ones(KT,1),Dd);
    Dx2 = Dx.^2;
    Dy2 = Dy.^2;
    Dhyp = (-(Dx2+Dy2)).^(n);
    iDhyp = 1./Dhyp;
    indsrmv = isinf(iDhyp);
    iDhyp(indsrmv) = 0;
    Lap = 1i*(Dx2+Dy2) - nuh*Dhyp - nul*iDhyp;
    Eop = exp(dt*Lap);

    f0 = zeros(KT^2,1);
    ksq = ( (kron(ones(KT,1),[0:K -K+1:-1]')).^2 + (kron([0:K -K+1:-1]',ones(KT,1))).^2) ;
    indsl = ksq >= Kl^2;
    indsh = ksq <=  Kh^2;
    indsc = logical(indsl.*indsh);
    f0(indsc) = 1;
    f0 = reshape(f0,KT,KT);
    f0 = f0c*f0;
    
    un = zeros(KT^2,1);
    uavg = zeros(KT^2,1);
    
    Nsteps = tf/dt;
    Ncnt = [];
    Nstart = floor(.99*Nsteps);
    Nint = 20;
    
    acnt = 0;
    pcnt = 2;
    
    DMDmat = zeros(KT^2,floor((Nsteps-Nstart)/Nint));
    rsdl = zeros(floor((Nsteps-Nstart)/Nint),1);
    for jj=1:Nsteps
        k1 = dt*nonlin(un,f0,KT);
        k2 = dt*nonlin(Eop.*(un+k1),f0,KT);
        un = Eop.*(un+k1/2) + k2/2;
        if jj>=Nstart 
            if mod(jj,Nint)==0
                uphys = ifft2(reshape(un,KT,KT));                   
                if jj == Nstart
                    ptnzr = uphys;
                end
                uavg = uavg + abs(un.*conj(un))/KT^4;
                nint = sum(sum(real(uphys.*conj(uphys))))*(1/KT)^2;
                Ncnt = [Ncnt nint];
                DMDmat(:,acnt+1) = abs(uphys(:));
                if acnt > 0
                    avec = DMDmat(:,1:acnt)\DMDmat(:,acnt+1);
                    rsdl(acnt) = log10(norm(DMDmat(:,1:acnt)*avec - DMDmat(:,acnt+1)));                    
                end
                acnt = acnt + 1;
            end
        end
    end
    
    if acnt > 0
        uavg = fftshift(reshape(uavg/acnt,KT,KT));
        [krad,kavg] = mat_avg(uavg,K);
        
        V1 = DMDmat(:,1:end-1);
        V2 = DMDmat(:,2:end);
        [U,Sig,W] = svd(V1,'econ');
        idSig = 1./diag(Sig);
        indsrmv = isinf(idSig);
        idSig(indsrmv) = 0;
        iSig = diag(idSig);
        
        V2 = U'*V2*W*iSig;
        [evecs,evals] = eigs(V2,acnt-1);
        devals = diag(evals);
        evecs = U*evecs;
        
        bspread = evecs\ptnzr(:);
        mdmags = (devals.^(dt*Nint*acnt)).*bspread;
        [~,Iinds] = sort(abs(mdmags),'descend');        
              
        for mm=1:pcnt     
            figure(mm)
            surf(Xmesh,Xmesh,abs(mdmags(Iinds(mm))*reshape(evecs(:,Iinds(mm)),KT,KT)),'LineStyle','none')
            h = set(gca,'FontSize',30);
            set(h,'Interpreter','LaTeX')
            xlabel('$x$','Interpreter','LaTeX','FontSize',30)
            ylabel('$y$','Interpreter','LaTeX','FontSize',30)        
        end
       
        figure(pcnt+1)
        plot(1:(acnt-1),rsdl,'k','LineWidth',2)
        
        figure(pcnt+2)
        plot(1:length(Iinds),log10(abs(devals(Iinds))),'k','LineWidth',2)
        
        figure(pcnt+3)
        plot(1:length(Iinds),bspread(Iinds),'k','LineWidth',2)
        
    end
    
    ufin = ifft2(reshape(un,KT,KT));
    
    figure(pcnt+4)
    surf(Xmesh,Xmesh,abs(ufin),'LineStyle','none')
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
    krad = (1:K)';
    kavg = M(K,K+1:KT)';    
end
