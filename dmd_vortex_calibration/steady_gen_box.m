function [uinterp] = steady_gen_box(Llx,K,xcs,sgns)

KT = 2*K;

[r,uplus,~] = steady_solve(2*Llx);

Ndata = length(r);

x=(-Llx:Llx/K:Llx-Llx/K); y=x;

[X,Y]=meshgrid(x,y);

theta = linspace(0,2*pi,Ndata+1);
theta = theta(1:Ndata);

[tgrid,rgrid] = meshgrid(theta,r);
[Xscat,Yscat] = pol2cart(tgrid,rgrid);

uinterp = ones(KT);
uextend = kron(ones(1,Ndata),uplus');

Nvorts = length(xcs(:,1));

for mm=1:Nvorts
    Xdat = (Xscat - xcs(mm,1));
    Ydat = (Yscat - xcs(mm,2));
    [xx,yy] = meshgrid(x-xcs(mm,1),y-xcs(mm,2));
    ang_surf = exp(-1i.*sgns(mm).*atan2(yy,xx));  
    uinterp = uinterp.*griddata(Xdat,Ydat,uextend,X,Y).*ang_surf;  
end
