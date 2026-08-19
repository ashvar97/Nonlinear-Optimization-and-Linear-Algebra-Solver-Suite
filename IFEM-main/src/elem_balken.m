% Element Unterroutine / element subroutine
% Bernoulli Balken-Element
% INPUT:    X -------- Knotenkoordinaten / node coordinates
%           he ------- Elementlaenge / element length
%           EI ------- Biegesteifigkeit / bending stiffness
%           q -------- Streckenlast / line load
function[KE,FE] = elem_1d(X,he,EI,q)
% element stiffness matrix
KE = 2*EI/he^3* [   6       3*he        -6      3*he;...
                    3*he    2*he^2      -3*he   he^2;...
                    -6      -3*he       6       -3*he;...
                    3*he    he^2        -3*he   2*he^2];
% element load vector
FE = he*q/2 * [1 ; he/6; 1; -he/6];
