function[X,edof,bc,f,ndof] = SP_stab_diskret(nel,pol,k_load,l)
% Anzahl Freiheitsgrade
% number of degrees of freedom
ndof=nel*pol+1;
% Elementlaenge
% length of an element
he = l/nel;
% Knotenkoordinaten
% node coordinates
for i=1:nel*pol+1
    X(i)=(i-1)*l/(nel*pol);
end
% Konnektivitaetsmatrix / connectivity matrix
edof=zeros(nel,pol+2);
for i=1:nel
    edof(i,1)=i;
	for j=1:(pol+1)
        if pol==1
            edof_1 = [1 2];
        elseif pol==2
            edof_1 = [1 3 2];
        end
        edof(i,:) = [i, (i-1)*pol+edof_1];
	end
end;
% Dirichlet Randbedingungen 
% Dirichlet bc
udl=0.0;
udr=0.0;
bc = [1 udl; ndof udr];
%bc = [1 udl];
% Volumenlast / volume force
f = k_load;
