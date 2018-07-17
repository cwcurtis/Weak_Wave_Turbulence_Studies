function sigfield = ftle_finder_sub_grid(xpaths,ypaths,dxr,K,tf)
    KT = 2*K;
    Ktot = KT + 2*(KT-2) + 2;
    
    dfux = zeros(KT-2);
    dfuy = zeros(KT-2);
    dfvx = zeros(KT-2);
    dfvy = zeros(KT-2);
    
    tvec = 4:3:Ktot-3;
    
    for jj=2:KT-1
        jval = 3*(jj-1)+1;
        dfux(jj-1,:) = (xpaths(jval+1,tvec)-xpaths(jval-1,tvec))/(2*dxr);
        dfvx(jj-1,:) = (ypaths(jval+1,tvec)-ypaths(jval-1,tvec))/(2*dxr);
        dfuy(:,jj-1) = (xpaths(tvec,jval+1)-xpaths(tvec,jval-1))/(2*dxr);
        dfvy(:,jj-1) = (ypaths(tvec,jval+1)-ypaths(tvec,jval-1))/(2*dxr);       
    end
    
    a = dfux.^2 + dfvx.^2;
    b = dfux.*dfuy + dfvx.*dfvy;
    c = dfuy.^2 + dfvy.^2;
    
    sigfield = 1/tf*log( (a + c + sqrt((a-c).^2 + 4*b.^2))/2 );