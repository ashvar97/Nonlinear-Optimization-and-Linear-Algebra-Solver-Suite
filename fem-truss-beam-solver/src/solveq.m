% Solver, with Dirichlet boundary conditions.
% Shared by both drivers in this folder (main_balken.m and SP_main_stab_1d.m) -- the
% linear-system solve is identical regardless of which element formulation produced K/F.
% K ----------------------------- global stiffness matrix
% F ----------------------------- global load vector
% bc ----------------------------- Dirichlet boundary conditions: [dof, prescribed_value; ...]
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
% solve for the free DOFs, with the prescribed DOFs moved to the right-hand side
s=K(fdof,fdof)\(F(fdof)-K(fdof,pdof)*dp);
% solution vector
d(pdof)=dp;
d(fdof)=s;
% reaction forces
Q = K*d - F;
