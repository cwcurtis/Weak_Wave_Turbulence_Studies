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
    nul = 1e-18;

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
    fprintf('Total Number of Time Steps: %1.4e \n', Nsteps);
    
    Nstart = floor(.99*Nsteps);
    fprintf('Starting Number of Time Step: %1.4e \n', Nstart);
    
    Ncnt = [];
    
    Nint = 50;
    NDMD = 5;
    acnt = 0;
    dcnt = 0;
    
    DMDmat = zeros(KT^2,floor((Nsteps-Nstart)/NDMD));
    rsdl = zeros(floor((Nsteps-Nstart)/NDMD),1);
    
    for jj=1:Nsteps
        k1 = dt*nonlin(un,f0,KT);
        k2 = dt*nonlin(Eop.*(un+k1),f0,KT);
        un = Eop.*(un+k1/2) + k2/2;
        if jj>=Nstart 
            
            if mod(jj,Nint)==0
                uphys = ifft2(reshape(un,KT,KT));                   
                uavg = uavg + abs(un.*conj(un))/KT^4;
                nint = sum(sum(real(uphys.*conj(uphys))))*(1/KT)^2;
                Ncnt = [Ncnt nint];                
                acnt = acnt + 1;
            end
                        
            if mod(jj,NDMD) == 0
                uphys = ifft2(reshape(un,KT,KT));                                    
                if jj == Nstart
                   ptnzr = abs(uphys(:));
                end                 
                DMDmat(:,dcnt+1) = abs(uphys(:));
                %if dcnt > 0
                %   avec = DMDmat(:,1:dcnt)\DMDmat(:,dcnt+1);
                %   rsdl(dcnt) = log10(norm(DMDmat(:,1:dcnt)*avec - DMDmat(:,dcnt+1))/norm(DMDmat(:,dcnt+1)));                    
                %end                 
                dcnt = dcnt + 1;
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
        [evecs,evals] = eigs(V2,dcnt-1);
        devals = diag(evals);
        evecs = U*evecs*diag(1./vecnorm(evecs));
        
        bspread = evecs\ptnzr;
        cdevals = log(devals)/(NDMD*dt);
        svdinds = real(cdevals) >= -.02;
        gvinds = real(cdevals) > 0.01;
        rminds = imag(cdevals) >= 0;
        tinds = logical(rminds.*svdinds.*(1-gvinds));
        
        nopsv = sum(gvinds);
        
        brm = bspread(tinds);
        dvrm = cdevals(tinds);
        rvcs = evecs(:,tinds);
        
        pcnt = sum(log10(abs(brm))>=1);
                
        [~,maxinds] = sort(abs(brm),'descend');
        
        figure(1)
        scatter(real(dvrm),log10(abs(brm)),20,'filled')
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$\mbox{Re}(\lambda_{j})$','Interpreter','LaTeX','FontSize',30)
        ylabel('$\log_{10}|b_{j}|$','Interpreter','LaTeX','FontSize',30) 
        
        figure(2)
        scatter(imag(dvrm),log10(abs(brm)),20,'filled')                
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$\mbox{Im}(\lambda_{j})$','Interpreter','LaTeX','FontSize',30)
        ylabel('$\log_{10}|b_{j}|$','Interpreter','LaTeX','FontSize',30) 
        
        for mm=1:pcnt
            figure(mm+2)
            surf(Xmesh,Xmesh,abs(brm(maxinds(mm))*reshape(rvcs(:,maxinds(mm)),KT,KT)),'LineStyle','none')
            h = set(gca,'FontSize',30);
            set(h,'Interpreter','LaTeX')
            xlabel('$x$','Interpreter','LaTeX','FontSize',30)
            ylabel('$y$','Interpreter','LaTeX','FontSize',30) 
            eval = dvrm(maxinds(mm));
            txt = sprintf('$$\\lambda: %1.4f + %1.4f i$$',real(eval),imag(eval));
            title(txt,'interpreter','latex')
        end
        %{
        if nopsv > 0
            gvpinds = logical(rminds.*gvinds);
            prvals = cdevals(gvpinds);
            brvals = bspread(gvpinds);
            rpevecs = evecs(:,gvpinds);
            for mm=1:sum(gvpinds)
                figure(mm+pcnt+2)
                surf(Xmesh,Xmesh,abs(brvals(mm)*reshape(rpevecs(:,mm),KT,KT)),'LineStyle','none')
                h = set(gca,'FontSize',30);
                set(h,'Interpreter','LaTeX')
                xlabel('$x$','Interpreter','LaTeX','FontSize',30)
                ylabel('$y$','Interpreter','LaTeX','FontSize',30) 
                eval = prvals(mm);
                txt = sprintf('$$\\lambda: %1.4f + %1.4f i$$',real(eval),imag(eval));
                title(txt,'interpreter','latex')
            end
        end
        %}
        
        figure(pcnt+2+1)
        scatter(real(cdevals),imag(cdevals),10,'filled')        
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$\mbox{Re}(\lambda_{j})$','Interpreter','LaTeX','FontSize',30)
        ylabel('$\mbox{Im}(\lambda_{j})$','Interpreter','LaTeX','FontSize',30) 
        
    end
    
    ufin = ifft2(reshape(un,KT,KT));
    
    figure(pcnt+2+2)
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
