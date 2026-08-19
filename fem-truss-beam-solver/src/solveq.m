% Solver, with Dir. boundary conditions
function [d,Q]=solveq(K,F,bc)
%------------------------------------------------------------- 
[nd,nd]=size(K);
fdof=[1:nd]';
%
d=zeros(size(fdof));
Q=zeros(size(fdof));
%
pdof=bc(:,1);
dp=bc(:,2);
fdof(pdof)=[];
%
re=F(fdof)-K(fdof,pdof)*dp;
s=K(fdof,fdof)\(F(fdof)-K(fdof,pdof)*dp);
% solution vector
d(pdof)=dp;
d(fdof)=s;
% reaction forces
Q = K*d - F;
 
     
   
