function nls_psuedo_solver(K,Llx,tf,Nv)

    tic

    dt = .01; 
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

    Nstart = 1000;
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
                    ptnzr = abs(uphys);
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
        evecs = U*evecs;
        
        bspread = evecs\ptnzr(:);
        mdmags = (devals.^(dt*Nint*acnt)).*bspread;
        [~,Iinds] = sort(abs(mdmags),'descend');        
              
        efin1 = reshape(evecs(:,Iinds(1)),KT,KT);
        efin2 = reshape(evecs(:,Iinds(2)),KT,KT);
        
        figure(1)
        surf(Xmesh,Xmesh,abs(mdmags(Iinds(1))*efin1),'LineStyle','none')
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$x$','Interpreter','LaTeX','FontSize',30)
        ylabel('$y$','Interpreter','LaTeX','FontSize',30)    
    
        figure(2)
        surf(Xmesh,Xmesh,abs(mdmags(Iinds(2))*efin2),'LineStyle','none')
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$x$','Interpreter','LaTeX','FontSize',30)
        ylabel('$y$','Interpreter','LaTeX','FontSize',30)    
        
        figure(3)
        plot(1:length(bspread),log10(abs((devals(Iinds)).^(dt*Nint*acnt).*bspread(Iinds))),'k','LineWidth',2)
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$n$','Interpreter','LaTeX','FontSize',30)
        ylabel('$\mbox{log}_{10}|b_{n}|$','Interpreter','LaTeX','FontSize',30)    
        
        figure(4)
        plot(1:length(devals),log10(abs(devals(Iinds))),'k','LineWidth',2)
        h = set(gca,'FontSize',30);
        set(h,'Interpreter','LaTeX')
        xlabel('$n$','Interpreter','LaTeX','FontSize',30)
        ylabel('$\mbox{log}_{10}|\mu_{n}|$','Interpreter','LaTeX','FontSize',30)               
    end
    
    ufin = ifft2(reshape(un,KT,KT));
   
    figure(5)
    surf(Xmesh,Xmesh,abs(ufin),'LineStyle','none')
    h = set(gca,'FontSize',30);
    set(h,'Interpreter','LaTeX')
    xlabel('$x$','Interpreter','LaTeX','FontSize',30)
    ylabel('$y$','Interpreter','LaTeX','FontSize',30)    
    
    figure(6)
    surf(Xmesh,Xmesh,log10(abs(abs(ufin)-abs(mdmags(Iinds(1))*efin1))),'LineStyle','none')
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
