function[X,edof,bc,f,ndof,he] = balken_diskret(nel,Q_load, M_load);
% length of the beam
l=1;
% number of DOFs (4 element DOFs)
ndof=(nel+1)*2;
% element length
he = l/nel;
% coordinates
for i=1:nel+1
    X(i)=(i-1)*l/nel;
end
% connectivity
for i=1:nel
    edof(i,1)=i;
	for j=1:4
		edof(i,j+1)=2*(i-1)+j;
	end
end;
% Dirichlet bcs
wdl=0.0;
dwdl=0.0;
bc = [1 wdl; 2 dwdl];
% load
f = zeros(ndof,1);
f(ndof-1) = Q_load;
f(ndof) = -M_load;
