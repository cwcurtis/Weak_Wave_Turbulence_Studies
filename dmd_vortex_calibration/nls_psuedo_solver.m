function nls_psuedo_solver(K,Llx,tf,Nv)

    tic

    dt = .1; 
    Nsteps = tf/dt;
    KT = 2*K;
    
    Xmesh = linspace(-Llx,Llx,KT+1);
    Xmesh = Xmesh(1:KT)';
    
    Dd = 1i*pi/Llx*[0:K -K+1:-1]';
    Dx = kron(Dd,ones(KT,1));
    Dy = kron(ones(KT,1),Dd);
    Dx2 = Dx.^2;
    Dy2 = Dy.^2;
    Lap = 1i*(Dx2+Dy2);
    Eop = exp(dt*Lap);

    Nstart = round(.6*Nsteps);
    tscale = dt*(Nsteps-Nstart);
    fprintf("Time scale for sampling is: %1.5f\n\n",tscale)
    Nint = 5;
    acnt = 0;
    
    DMDmat = zeros(KT^2,(Nsteps-Nstart)/Nint);
    
    sgns = ones(Nv);
    
    Xvmesh = linspace(-4*Nv,4*Nv,Nv+1);
    Xvmesh = Xvmesh(1:Nv)+1/2;
    [Xxvmesh,Yyvmesh] = meshgrid(Xvmesh);
    xcs = [Xxvmesh(:),Yyvmesh(:)];
    
    for jj=1:Nv
       if mod(jj,2) == 0
           sgns(jj,1:2:Nv-1) = -1;
       else
           sgns(jj,2:2:Nv) = -1;
       end
    end
    sgns = sgns(:);
        
    un = steady_gen_box(Llx,K,xcs,sgns);
    un = fft2(un);
    un = un(:);
    
    for jj=1:Nsteps
        k1 = dt*nonlin(un,KT);
        k2 = dt*nonlin(Eop.*(un+k1),KT);
        un = Eop.*(un+k1/2) + k2/2;
        if jj>=Nstart 
            if mod(jj,Nint)==0
                uphys = ifft2(reshape(un,KT,KT));                   
                if jj == Nstart
                    ptnzr = abs(uphys(:));
                end
                DMDmat(:,acnt+1) = abs(uphys(:));
                acnt = acnt + 1;
            end
        end
    end
    
    if acnt > 0
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
        evecs = U*evecs*diag(1./vecnorm(evecs));
        
        bspread = evecs\ptnzr;
        cdevals = log(devals)/(Nint*dt);
        coff = -.05;
        svdinds = real(cdevals) >= coff;
        fprintf("Real part cutoff is: %1.5f\n\n",coff);
        gvinds = real(cdevals) > 0.05;
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
        
        %{
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
        %}
        
        errvec = zeros(acnt-1,1);
        tsamp = zeros(acnt-1,1);
        epow = ones(length(devals),1);
        for mm=1:acnt-1
            approx = evecs(:,svdinds)*(epow(svdinds).*bspread(svdinds));
            errvec(mm) = norm(approx-DMDmat(:,mm),'inf')/norm(DMDmat(:,mm),'inf');
            tsamp(mm) = dt*(Nstart + (mm-1)*Nint);
            epow = epow.*devals;
        end
        
        figure(3)
        scatter(real(cdevals),imag(cdevals),10,'filled')        
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$\mbox{Re}(\lambda_{j})$','Interpreter','LaTeX','FontSize',30)
        ylabel('$\mbox{Im}(\lambda_{j})$','Interpreter','LaTeX','FontSize',30) 
        
        figure(4)
        plot(tsamp,log10(errvec),'k-','LineWidth',2)
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$t_{s}$','Interpreter','LaTeX','FontSize',30)
        ylabel('$E(t_{s})$','Interpreter','LaTeX','FontSize',30) 
                
    end
    
    ufin = ifft2(reshape(un,KT,KT));
   
    figure(5)
    surf(Xmesh,Xmesh,abs(ufin),'LineStyle','none')
    h = set(gca,'FontSize',30);
    set(h,'Interpreter','LaTeX')
    xlabel('$x$','Interpreter','LaTeX','FontSize',30)
    ylabel('$y$','Interpreter','LaTeX','FontSize',30)
    
    toc
end

function uout = nonlin(un,KT)
    uphys = ifft2(reshape(un,KT,KT));
    unl = -1i*fft2(uphys.*uphys.*conj(uphys));
    uout = unl(:);
end
