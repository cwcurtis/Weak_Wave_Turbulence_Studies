function [xpaths,ypaths] = ftle_finder_rk4(xpaths,ypaths,uvels,vvels,Xmesh,dtl)
    
    [Xmgrid,Ymgrid] = meshgrid(Xmesh);    
    
    k1x = dtl*interp2(Xmgrid,Ymgrid,uvels(:,:,1),xpaths,ypaths);
    k1y = dtl*interp2(Xmgrid,Ymgrid,vvels(:,:,1),xpaths,ypaths);                
                
    xup = xpaths + k1x/2;
    yup = ypaths + k1y/2;
                
    k2x = dtl*interp2(Xmgrid,Ymgrid,uvels(:,:,2),xup,yup);
    k2y = dtl*interp2(Xmgrid,Ymgrid,vvels(:,:,2),xup,yup);
                
    xup = xpaths + k2x/2;
    yup = ypaths + k2y/2;
                                
    k3x = dtl*interp2(Xmgrid,Ymgrid,uvels(:,:,2),xup,yup);
    k3y = dtl*interp2(Xmgrid,Ymgrid,vvels(:,:,2),xup,yup);
                
    xup = xpaths + k3x;
    yup = ypaths + k3y;
       
    k4x = dtl*interp2(Xmgrid,Ymgrid,uvels(:,:,3),xup,yup);
    k4y = dtl*interp2(Xmgrid,Ymgrid,vvels(:,:,3),xup,yup);
                
    xpaths = xpaths + (k1x + 2*k2x + 2*k3x + k4x)/6;
    ypaths = ypaths + (k1y + 2*k2y + 2*k3y + k4y)/6;                            