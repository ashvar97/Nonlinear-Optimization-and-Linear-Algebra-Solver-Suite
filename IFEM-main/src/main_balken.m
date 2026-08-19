%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uebung zur Vorlesung Finite Element %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Strukturelemente - Beispiel: eingespannter Balken %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise to the lecture Finite Elements %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% JM - 27.05.07 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
% Input
nel = input('number of elements = ');
load = input('line load q = ');
Q_load = input('point load Q = ');
M_load = input('concentrated momentum M (beam theory orientation) = ');
EI = input('bending stiffness EI = ');
% discrete system
% X ------------------- node coordinates
% edof ---------------- connectivity
% bc ------------------ Dirichlet bcs
% f_ex ---------------- vector of point loads
% ndof ---------------- number of DOFs
% he ------------------ element length
[X,edof,bc,f_ex,ndof,he] = balken_diskret(nel,Q_load,M_load);
% stiffness matrix, load vector
% K ------------------- global stiffness matrix
% F ------------------- gloabal load vector
K = zeros(ndof,ndof);
F = zeros(ndof,1);
% loop over all elements
% KE, FE -------------- element stiffness matrix, -load vector
for ie=1:nel
    % determination of element size
    [KE,FE] = elem_balken(X(edof(ie,3:2:end)/2),he,EI,load);
    % assembly of global stiffness matrix and load vector
    K(edof(ie,2:end),edof(ie,2:end))=K(edof(ie,2:end),edof(ie,2:end))+KE;
    F(edof(ie,2:end))=F(edof(ie,2:end))+FE;
end
% adding point forces to the load vector
F = F + f_ex;
% solving system of equations, incoorporation of Dirichlet bcs
% w ------------------- solution vector
% Re ------------------ reaction forces
[w,Re] = solveq(K,F,bc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% graphical output of the numerical and the analytical solution
% analytical solution
x_ex = [0:0.01:1];
for i=1:length(x_ex)
    w_ex(i) = -1/EI*(1/24*load*x_ex(i)^4-1/6*(Q_load+load*1)*x_ex(i)^3-...
        1/2*(M_load-Q_load*1-1/2*load*1^2)*x_ex(i)^2);
end
plot(x_ex,w_ex,'r');
% numerical solution
% loop over all elements
for ie = 1:nel
    x_num = [X(edof(ie,3)/2):0.005:X(edof(ie,5)/2)];
    w_el = w(edof(ie,2:end));
    for j=1:length(x_num)
        xsi = (2*x_num(j)-X(edof(ie,3)/2)-X(edof(ie,5)/2))/he;
        mx_N = [1/4*(2-3*xsi+xsi^3); 1/4*(1-xsi-xsi^2+xsi^3)*he/2; ...
            1/4*(2+3*xsi-xsi^3); 1/4*(-1-xsi+xsi^2+xsi^3)*he/2];
        w_num(j) = -mx_N'*w_el;
    end
    hold on; plot(x_num,w_num,'b'); hold on
end
legend('analytic','numeric');