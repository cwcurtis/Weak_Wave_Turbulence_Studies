
% Finding a single vortex steady state using radial ODE and BVP solver:
%       i u_t = -(1/2) (u_xx+u_yy) + alpha*(|u|^2) u + V(x) u
% alpha = 1 is defocusing
% alpha =-1 is focusing
%
% Examples:
%
% Ex1: Vortex inside MT
% S=1;mu=1;Omega=0.05;
% [r,u,up] = SteadyStateBVP(mu,S,Omega);
%
% Ex2: Vortex on a homogeneous background
% S=1;mu=1;Omega=0;
% [r,u,up] = SteadyStateBVP(mu,S,Omega);
%

function [r,u,up] = steady_solve(Llx)

    r0 = 1e-8;
    
    rmax = sqrt(2)*Llx;
        
    Nr = 400;
    rini = linspace(r0,rmax,Nr);
    
    u0 = rini.*tanh(rini)./(1+rini);        
    u0p = (tanh(rini)./(1+rini).^2 + rini.*sech(rini).^2./(1+rini));
    
    solinit.x = rini;
    solinit.y(1,:) = u0;
    solinit.y(2,:) = u0p;

    options=bvpset('AbsTol',1e-10,'RelTol',1e-5,'Nmax',1000);

    sol = bvp4c(@vortexode,@vortexbc,solinit,options);

    r  = sol.x;
    u  = sol.y(1,:);
    up = sol.yp(1,:);

    
%---------Boundary Condition--------%
function res=vortexbc(ua,ub)
    %ua:left, ub:right, ua(1);u, ua(2):u'
    res = [ua(1) ; ub(1) - 1];

%---------ODE function-------------%
function yprime = vortexode(r,u)
    yprime = [  u(2,:) ; -u(2,:)./r + u(1,:)./(r.^2) - (1  -  u(1,:).^2).*u(1,:) ];





