function [sigfield,skip] = ftle_finder(uvels,vvels,Xmesh,dt,dx)
    KT = length(Xmesh);
    skip = 1;
    dxl = skip*dx;
    ntot = length(uvels(1,1,:));
    ninds = ntot/2-1;
    dtl = 2*dt;
    xpaths = zeros(KT/skip,KT/skip,ninds);
    ypaths = zeros(KT/skip,KT/skip,ninds);
        
    [Xmgrid,Ymgrid] = meshgrid(Xmesh);
    [xpaths(:,:,1),ypaths(:,:,1)] = meshgrid(Xmesh(1:skip:KT));
        
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
    
    dfux = zeros(KT/skip-2,KT/skip-2);
    dfuy = zeros(KT/skip-2,KT/skip-2);
    dfvx = zeros(KT/skip-2,KT/skip-2);
    dfvy = zeros(KT/skip-2,KT/skip-2);
    tvec = 2:KT/skip-1;
    
    for jj=2:KT/skip-1
        dfux(jj-1,:) = (xpaths(jj+1,tvec,end)-xpaths(jj-1,tvec,end))/(2*dxl);
        dfvx(jj-1,:) = (ypaths(jj+1,tvec,end)-ypaths(jj-1,tvec,end))/(2*dxl);
        dfuy(:,jj-1) = (xpaths(tvec,jj+1,end)-xpaths(tvec,jj-1,end))/(2*dxl);
        dfvy(:,jj-1) = (ypaths(tvec,jj+1,end)-ypaths(tvec,jj-1,end))/(2*dxl);       
    end
    
    a = dfux.^2 + dfvx.^2;
    b = dfux.*dfuy + dfvx.*dfvy;
    c = dfuy.^2 + dfvy.^2;
    
    sigfield = 1/(dt*ntot)*log( (a + c + sqrt((a-c).^2 + 4*b.^2))/2 );