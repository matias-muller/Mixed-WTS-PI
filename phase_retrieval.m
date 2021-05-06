V = [0 1 0 0 0]';%rand(5,1);
V = [0; V; flip(V)];
r = ifft(V);
N = (length(r)+1)/2;
r = [r(N+1:end); r(1:N)];
R=0;for i = 1:(2*N-1)
R = R + r(i)*z^(-i);
end
R = minreal(z^N*R,1e-5);
zpk(R)
R2 = minreal(R*z/((z-1)*(1-z)));
[G,S] = spectralfact(R2)
G = G*sqrt(S);
G = G*(z-1)/z;
[b,~] = tfdata(G);
b = b{1}';
b = [b; zeros(N-1,1)];