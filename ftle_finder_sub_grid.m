function sigfield = ftle_finder_sub_grid(uvels,vvels,Xmesh,dt,dx)
    KT = length(Xmesh);
    dxr = dx/100;
    ntot = length(uvels(1,1,:));
    ninds = ntot/2-1;
    dtl = 2*dt;
    Ktot = KT + 2*(KT-2) + 2;
    xpaths = zeros(Ktot,Ktot,ninds);
    ypaths = zeros(Ktot,Ktot,ninds);
    Xpts = zeros(Ktot,1);
    Xrght = Xmesh(1:KT-1)+dxr;
    Xlft = Xmesh(2:KT)-dxr;
    Xpts(1:3:Ktot) = Xmesh;
    Xpts(2:3:Ktot) = Xrght;
    Xpts(3:3:Ktot) = Xlft;
    
    [Xmgrid,Ymgrid] = meshgrid(Xmesh);    
    [xpaths(:,:,1),ypaths(:,:,1)] = meshgrid(Xpts);
    
    for mm=1:ninds-1               
        k1x = dtl*interp2(Xmgrid,Ymgrid,uvels(:,:,2*mm-1),xpaths(:,:,mm),ypaths(:,:,mm));
        k1y = dtl*interp2(Xmgrid,Ymgrid,vvels(:,:,2*mm-1),xpaths(:,:,mm),ypaths(:,:,mm));                
                
        xup = xpaths(:,:,mm) + k1x/2;
        yup = ypaths(:,:,mm) + k1y/2;
                
        k2x = dtl*interp2(Xmgrid,Ymgrid,uvels(:,:,2*mm),xup,yup);
        k2y = dtl*interp2(Xmgrid,Ymgrid,vvels(:,:,2*mm),xup,yup);
                
        xup = xpaths(:,:,mm) + k2x/2;
        yup = ypaths(:,:,mm) + k2y/2;
                                
        k3x = dtl*interp2(Xmgrid,Ymgrid,uvels(:,:,2*mm),xup,yup);
        k3y = dtl*interp2(Xmgrid,Ymgrid,vvels(:,:,2*mm),xup,yup);
                
        xup = xpaths(:,:,mm) + k3x;
        yup = ypaths(:,:,mm) + k3y;
        
        k4x = dtl*interp2(Xmgrid,Ymgrid,uvels(:,:,2*mm+1),xup,yup);
        k4y = dtl*interp2(Xmgrid,Ymgrid,vvels(:,:,2*mm+1),xup,yup);
                
        xpaths(:,:,mm+1) = xpaths(:,:,mm) + (k1x + 2*k2x + 2*k3x + k4x)/6;
        ypaths(:,:,mm+1) = ypaths(:,:,mm) + (k1y + 2*k2y + 2*k3y + k4y)/6;                        
    end
    
    dfux = zeros(KT-2,KT-2);
    dfuy = zeros(KT-2,KT-2);
    dfvx = zeros(KT-2,KT-2);
    dfvy = zeros(KT-2,KT-2);
    
    tvec = 4:3:Ktot-3;
    
    for jj=2:KT-1
        jval = 3*(jj-1)+1;
        dfux(jj-1,:) = (xpaths(jval+1,tvec,end)-xpaths(jval-1,tvec,end))/(2*dxr);
        dfvx(jj-1,:) = (ypaths(jval+1,tvec,end)-ypaths(jval-1,tvec,end))/(2*dxr);
        dfuy(:,jj-1) = (xpaths(tvec,jval+1,end)-xpaths(tvec,jval-1,end))/(2*dxr);
        dfvy(:,jj-1) = (ypaths(tvec,jval+1,end)-ypaths(tvec,jval-1,end))/(2*dxr);       
    end
    
    a = dfux.^2 + dfvx.^2;
    b = dfux.*dfuy + dfvx.*dfvy;
    c = dfuy.^2 + dfvy.^2;
    
    sigfield = 1/(dt*ntot)*log( (a + c + sqrt((a-c).^2 + 4*b.^2))/2 );