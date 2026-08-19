% Loeser, Beruecksichtigung der Dirichlet Randbedingungen
% solver, consideration of Dirichlet boundary conditions
% K ----------------------------- Systemsteifigkeitsmatrix / gloabal stiffness matrix
% F ----------------------------- Systemlastvektor / global load vector
% bc ---------------------------- Dirichlet Randbedingungen / Dirichlet boundary conditions
function [d,Q]=SP_solveq(K,F,bc)
%------------------------------------------------------------- 
[nd,nd]=size(K);
fdof=[1:nd]';
%
d=zeros(size(fdof));
Q=zeros(size(fdof));
% pdof -------------------------- Freiheitsgrade mit Dirichlet RB / DOF with Dirichlet bc
% dp ---------------------------- vorgegebene Verschiebungen / predefined displacement
% fdof -------------------------- Freiheitsgrade ohen Dirichlet RB / DOF without Dirichlet bc
pdof=bc(:,1);
dp=bc(:,2);
fdof(pdof)=[];
% Loesung des Gleichungssystem, 
% solution of the system of equations
% Beruecksichtigung der Dirichlet RB "auf der rechten Seite"
% consideration of the Dirichlet bc "on the right hand side"
s=K(fdof,fdof)\(F(fdof)-K(fdof,pdof)*dp);
% Loesungsvektor / solution vector
d(pdof)=dp;
d(fdof)=s;
% Reaktionskraefte
Q = K*d - F;
 
     
   
