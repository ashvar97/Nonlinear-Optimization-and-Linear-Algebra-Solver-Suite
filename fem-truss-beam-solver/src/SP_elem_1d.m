% Elementunterroutine
% element sub routine
% berechnet Elementsteifigkeitsmatrix und -lastvektor
% computes the element stiffness matrix and the load vector
function[KE,FE] = SP_elem_1d(X,pol,emod,A,f)
% Elementlaenge
% length of the element
he = X(2)-X(1);
% lineares Stabelement
% linear truss element
if pol == 1;
% Elementsteifigkeitsmatrix
% element stiffness matrix
    KE = emod*A/he* [1 -1; -1 1];
% Elementlastvektor
%element load vector
% fuer konstante Streckenlast f(x)=k_load
% for a constant line load f(x)=k_load
%    FE = 1/2 * he * f *[1; 1];
% fuer linear veraenderliche Streckenlast f(x)=k_load*x
% for a linearly variable line load f(x)=k_load*x
    FE = 1/3 * he* f * [X(1)+0.5*X(2); 0.5*X(1)+X(2)];
% quadratisches Stabelement
% quadratic truss element
elseif pol == 2;    
    KE = emod*A/(3*he)* [7 1 -8; 1 7 -8; -8 -8 16];
% fuer linear veraenderliche Streckenlast f(x)=k_load*x
% for a linearly variable line load f(x)=k_load*x
    FE = 1/6 * he* f * [X(1); X(2); 2*(X(1)+X(2))];
end