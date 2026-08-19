%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uebung zur Vorlesung Finite Elemente %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FEM 1D - Beispiel: beidseitig eingespannter Stab %%%%%%%%%%%%%%%%%%%%%%%%
% unter linear veraenderlicher, verteilter Last n=k*x %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise to the lecture Finite Elements %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FEM 1D - example: truss clamped at both sides %%%%%%%%%%%%%%%%%%%%%%%%%%%
% with linearly distributed load n=k*x %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% JM - 14.05.07 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SP - 12.05.16 Neue Elementnummerierung
%
%  local 1d quadratic element
%
%       1------3-------2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nel ---------------------------------- Anzahl Elemente /number of elements
% pol ---------------------------------- Polynomgrad / polynomial dof
% k_load ------------------------------- Lastfaktor / load vector
% e_mod -------------------------------- E-Modul
clear all
close all
% Eingabe / Input
nel = input('number of elements = ');
pol = input('polynomial dof for the elements (1-linear, 2-quadratic) = ');
k_load = input('load - factor (in N/mm^2) k = ');

% Young's modulus
% emod = input('E-Modul = ');
 emod = 210000; %N/mm^2, steel
 disp(['E = ', num2str(emod), ' N/mm^2']);
% konstante Querschnittsfläche A
% --------------------
% constant cross section A
 A=2500; %mm^2, cross section of 50mmx50mm
 disp(['A = ', num2str(A), ' mm^2']);
% Laenge des Stabs l
% --------------------
% length of the truss l
 l=1000; %mm
 disp(['l = ', num2str(l), ' mm']);

%emod = input('E-Modul = ');
% diskretes System (Einlesen der Geometrie, Randbedingungen, des Netzes)
% discrete system (reading geometry, bc, mesh)
% X ------------------------------------ Vektor der Knotenkoordinaten / vector of node coordinates
% edof  -------------------------------- Konnektivitaetsmatrix / connectivity matrix
% bc ----------------------------------- Dirichlet Randbedingungen / Dirichlet bc
% f ------------------------------------ Volumenlast / body force
% ndof --------------------------------- Anzahl Freiheitsgrade / number of dof
[X,edof,bc,f,ndof] = SP_stab_diskret(nel,pol,k_load,l)
% Systemsteifigkeitsmatrix, Systemlastvektor
% global stiffness, global load vector
K = zeros(ndof,ndof);
F = zeros(ndof,1);
% Schleife ueber alle Elemente / loop over all elements
for ie=1:nel
    % KE ------------------------------- Elementsteifigkeitsmatrix / element stiffness matrix
    % FE ------------------------------- Elementlastvektor / element load vector
    [KE,FE] = SP_elem_1d(X(edof(ie,2:end)),pol,emod,A,f);
    % Zusammenbau / assembly
    K(edof(ie,2:end),edof(ie,2:end))=K(edof(ie,2:end),edof(ie,2:end))+KE;
    F(edof(ie,2:end))=F(edof(ie,2:end))+FE;
end
% Loesung des Gleichungssystems, 
% solution of the system of equations
% Einbau der Dirichlet Randbedingungen
% applying Dirichlet bcs
[u,Q] = SP_solveq(K,F,bc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nachlaufrechnung, Spannungen
% postprocessing, stresses
for ie=1:nel
    XE=X(edof(ie,2:end)); he=XE(2)-XE(1);
    if pol == 1
        eps(ie) = 2/he*[-0.5 0.5]*u(edof(ie,[2 3]));
        sig(ie) = emod*eps(ie);
    elseif pol == 2
        eps0(ie) = 2/he*[-1.5 2 -0.5]*u(edof(ie,[2 4 3]));
        epse(ie) = 2/he*[0.5 -2  1.5]*u(edof(ie,[2 4 3]));
        sig0(ie) = emod*eps0(ie);
        sige(ie) = emod*epse(ie);
    end
end

% Plot u in mm against x in m (-> plotx_fac=0.001):
plotx_fac = 0.001;
% Analytische Vergleichsloesung
% fuer linear veraenderliche Streckenlast mit f(x)=k_load*x;
% analytical solution for linearly variable line load f(x)=k_load*x;
figure('Name', 'solution')
xa = 0:l/100:l;
ya  = 1/(emod*A)*k_load*xa.*(-1/6*xa.^2+1/6*l^2);
siga = k_load/(6*A)*(l^2-3*xa.^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graphische Ausgabe der numerischen und der analytischen Loesung
% Analytische Vergleichsloesung
% graphical output of the numerical and the analytical solution
figure(1)
plot(0.001*xa, ya,'r');
% numerische Loesung
% numerical solution
hold on
for ie=1:nel
    if pol == 1
        p = polyfit(X(edof(ie,[2 3]))',u(edof(ie,2:end)),pol);
    elseif pol == 2
        p = polyfit(X(edof(ie,[2 4 3]))',u(edof(ie,[2 4 3])),pol);
    end
    %p = polyfit(X(edof(ie,3:end))',u(edof(ie,3:end)),pol);
    xn=[X(edof(ie,2)):X(edof(ie,3))/100:X(edof(ie,3))];
    f = polyval(p,xn');
    plot(plotx_fac*X(edof(ie,2:3)),u(edof(ie,2:3)),'ob',plotx_fac*xn,f,'-b');
end
legend('analytical','numerical');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graphische Ausgabe der Spannungen
% graphical output of the stress
figure('Name','stress_analytical')
% Analytische Vergleichsloesung / analytical solution
plot(plotx_fac*xa,siga,'r');
hold on
% numerische Loesung / numerical solution
for ie=1:nel
    hold on
    if pol == 1;
        plot(plotx_fac*X(edof(ie,2:end)),[sig(ie) sig(ie)]);
        %if ie~=nel
        %    plot([plotx_fac*X(edof(ie,end)) plotx_fac*X(edof(ie,end))],[sig(ie) sig(ie+1)]);
        %end
    elseif pol == 2;
        plot([plotx_fac*X(edof(ie,2)) plotx_fac*X(edof(ie,3))] ,[sig0(ie) sige(ie)]);
%         if ie~=nel
%             plot([plotx_fac*X(edof(ie,3)) plotx_fac*X(edof(ie,3))],[sige(ie) sig0(ie+1)]);
%         end
    end
end
legend('analytical','numerical');