function ucoher = wavelet_decomp(u,mxlvl)

    qval = 2.8;
    nmax = 300;
    
    %omode = dwtmode('status','nodisplay');
    [Cv,Sv] = wavedec2(abs(u),mxlvl,'coif4');
    steps = Sv(:,1).*Sv(:,2);
    lvlset = mxlvl - 5;
    
    % Work over the fine scales to determine the coherent structures.
    pinds = steps(1) + 3*sum(steps(2:lvlset-1));
    
    for jj = lvlset:mxlvl
        nstep = steps(jj);
        for mm=1:3
            cinds = pinds+1+nstep*(mm-1):pinds + mm*nstep;
            Cvjl = Cv(cinds);
            % Begin interval selection
            theta0 = 1e6;
            sig0 = sqrt(sum((Cvjl).^2)/nstep);
            cnt = 0;
            while abs(qval*sig0-theta0) > 1e-14*theta0 && cnt<nmax
                theta0 = qval*sig0;
                linds = abs(Cvjl) <= theta0;
                ntheta = sum(linds);
                sig0 = sqrt(sum(Cvjl(linds)).^2/ntheta);         
                cnt = cnt + 1;
            end
            %disp(theta0)
            Cvjl(linds) = 0;
            Cv(cinds) = Cvjl;        
        end
        pinds = pinds + 3*nstep;
    end
    ucoher = waverec2(Cv,Sv,'coif4');
    %dwtmode(omode);