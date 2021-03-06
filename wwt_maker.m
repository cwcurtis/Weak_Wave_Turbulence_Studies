function wwt_maker(K,Llx,tf)

    tic
    scl = 1;
    
    tf = 1/scl*tf;
    
    dt = scl*.1; 
    dx = Llx/K;
    n = 8;
    Kmask = K/2;
    KT = 2*K;
    KTT = KT^2;

    f0c = KTT*2.1e-3; 
    nuh = 1/scl*2e-6;
    nul = 0;

    Xmesh = linspace(-Llx,Llx,KT+1);
    Xmesh = Xmesh(1:KT)';
    dreg = 1e-2;
    
    Kl = 4;
    Kh = 6;

    Dds = 1i*pi/Llx*[0:K-1 0 -K+1:-1]'; 
    Dxs = kron(Dds,ones(KT,1));
    Dys = kron(ones(KT,1),Dds);
    
    Dxs2 = Dxs.^2;
    Dys2 = Dys.^2;
        
    Dd = 1i*pi/Llx*[0:K -K+1:-1]';
    Dx = kron(Dd,ones(KT,1));
    Dy = kron(ones(KT,1),Dd);
    
    Dx2 = Dx.^2;
    Dy2 = Dy.^2;
    
    Dhyp = (-(Dx2+Dy2)).^(n);
    iDhyp = 1./Dhyp;
    indsrmv = isinf(iDhyp);
    iDhyp(indsrmv) = 1;
    
    Lap = 1i*(Dxs2+Dys2) - nuh*Dhyp - nul*iDhyp;
    Eop = exp(dt*Lap);

    f0 = zeros(KT^2,1);
    ksq = ( (kron(ones(KT,1),[0:K -K+1:-1]')).^2 + (kron([0:K -K+1:-1]',ones(KT,1))).^2) ;
    indsl = ksq >= Kl^2;
    indsh = ksq <=  Kh^2;
    indsc = logical(indsl.*indsh);
    f0(indsc) = 1;
    f0 = reshape(f0,KT,KT);
    f0 = f0c*f0;
    f0 = fft2(real(ifftshift(ifft2(f0))));
                
    un = zeros(KT^2,1);
    uavg = zeros(KT^2,1);
    
    Nsteps = tf/dt;
    fprintf('Total Number of Time Steps: %1.4e \n', Nsteps);
    
    Nstart = floor(.98*Nsteps);
    fprintf('Starting Number of Time Step: %1.4e \n', Nstart);
    
    tscale = dt*(Nsteps-Nstart);
    fprintf("Time scale for sampling is: %1.5f\n\n",tscale)
        
    Ncnt = [];
    
    Nint = 50;
    NDMD = floor(1/scl*5);
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
                        
            if mod(jj,NDMD)==0 || jj==Nstart
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
        nmbrmds = length(devals);
        evecs = U*evecs*diag(1./vecnorm(evecs));
        
        bspread = evecs\ptnzr;
        cdevals = log(devals)/(NDMD*dt);
        coff = -.02;
        
        svdinds = real(cdevals) >= coff;
        gvinds = real(cdevals) > 0.01;
        rminds = imag(cdevals) >= 0;
        tinds = logical(rminds.*svdinds.*(1-gvinds));
        
        %nopsv = sum(gvinds);
        
        brm = bspread(tinds);
        dvrm = cdevals(tinds);
        rvcs = evecs(:,tinds);
                        
        %[~,maxinds] = sort(abs(brm),'descend');
        
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
        
        epow = ones(length(devals),1);
        mdcnts = zeros(dcnt-1,1);
        tsamp = zeros(dcnt-1,1);
        mincnt = zeros(dcnt-2,1);
        maxcnt = zeros(dcnt-2,1);
        
        for mm=1:dcnt-1
            bamp = epow.*bspread;
            [~,mxindslc] = sort(abs(bamp),'descend');
                 
            indl = 0;
            err = 1;
            approx = 0;
            while err > .1
                indl = indl + 1;
                approx = approx + evecs(:,mxindslc(indl))*bamp(mxindslc(indl));
                err = norm(approx-DMDmat(:,mm))/norm(DMDmat(:,mm));                      
            end
            mdcnts(mm) = indl;
            tsamp(mm) = dt*(Nstart + (mm-1)*NDMD);
            new_inds = mxindslc(1:indl);
            if mm>1
               rcnt = length(old_inds);
               ccnt = length(new_inds);
               diffmat = (repmat(new_inds',rcnt,1)-repmat(old_inds,1,ccnt))==0;  
               loctot = sum(sum(diffmat));                                                         
               %mincnt(mm-1) = loctot/mn;
               maxcnt(mm-1) = loctot/(rcnt+ccnt-loctot);
            end
            if mm == dcnt-1
                maxvals = cdevals(mxindslc(1:indl));
                redinds = imag(maxvals) >= 0;
                revals = maxvals(redinds);
                rvecs = evecs(:,mxindslc(1:indl));
                brm = bamp(mxindslc(1:indl));
                modes = abs(rvecs(:,redinds)*diag(brm(redinds)));
                pcnt = sum(redinds);
            end            
            epow = epow.*devals;
            old_inds = new_inds;            
        end
        Kvec = -K+1:K;
        if pcnt >= 4
            for ll=1:4            
                cmode = reshape(modes(:,ll),KT,KT);
                eval = revals(ll);
            
                figure(2*(ll-1)+1+2)
                surf(Xmesh,Xmesh,cmode,'LineStyle','none')
                h = set(gca,'FontSize',30);
                set(h,'Interpreter','LaTeX')
                xlabel('$x$','Interpreter','LaTeX','FontSize',30)
                ylabel('$y$','Interpreter','LaTeX','FontSize',30) 
                txt = sprintf('$$\\lambda: %1.4f + %1.4f i$$',real(eval),imag(eval));
                title(txt,'interpreter','latex')
            
                figure(2*(ll-1)+2+2)
                surf(Kvec,Kvec,log10(abs(fftshift(fft2(cmode)/KT))),'LineStyle','none')
                h = set(gca,'FontSize',30);
                set(h,'Interpreter','LaTeX')
                xlabel('$K_x$','Interpreter','LaTeX','FontSize',30)
                ylabel('$K_y$','Interpreter','LaTeX','FontSize',30) 
                txt = sprintf('$$\\lambda: %1.4f + %1.4f i$$',real(eval),imag(eval));
                title(txt,'interpreter','latex')            
            end
        end
        %{
        errvec = zeros(dcnt-1,1);
        epow = ones(length(devals),1);
        for mm=1:dcnt-1
            approx = evecs(:,svdinds)*(epow(svdinds).*bspread(svdinds));
            errvec(mm) = norm(approx-DMDmat(:,mm),'inf')/norm(DMDmat(:,mm),'inf');
            tsamp(mm) = dt*(Nstart + (mm-1)*NDMD);
            epow = epow.*devals;
        end
        %}
        
        figure(11)
        scatter(real(cdevals),imag(cdevals),10,'filled')        
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$\mbox{Re}(\lambda_{j})$','Interpreter','LaTeX','FontSize',30)
        ylabel('$\mbox{Im}(\lambda_{j})$','Interpreter','LaTeX','FontSize',30) 
        
        figure(12)
        plot(tsamp,mdcnts/nmbrmds,'k-','LineWidth',2)
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$t^{(s)}_{n}$','Interpreter','LaTeX','FontSize',30)
        ylabel('$C_{r}(n)$','Interpreter','LaTeX','FontSize',30)  
        
        figure(13)
        plot(tsamp(2:end),maxcnt,'k-','LineWidth',2)
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$t^{(s)}_{n}$','Interpreter','LaTeX','FontSize',30)        
        ylabel('$\mathcal{J}_{i}(n)$','Interpreter','LaTeX','FontSize',30)        
    end
    
    ufin = ifft2(reshape(un,KT,KT));
    
    figure(14)
    surf(Xmesh,Xmesh,abs(ufin),'LineStyle','none')
    h = set(gca,'FontSize',30);
    set(h,'Interpreter','LaTeX')
    xlabel('$x$','Interpreter','LaTeX','FontSize',30)
    ylabel('$y$','Interpreter','LaTeX','FontSize',30)
    
    figure(15)
    surf(Kvec,Kvec,log10(abs(fftshift(reshape(un,KT,KT))/KT)),'LineStyle','none')
    h = set(gca,'FontSize',30);
    set(h,'Interpreter','LaTeX')
    xlabel('$K_x$','Interpreter','LaTeX','FontSize',30)
    ylabel('$K_y$','Interpreter','LaTeX','FontSize',30)
        
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
