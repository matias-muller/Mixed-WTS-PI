clear;
clc;
home;

z = tf('z');
e = @(k,n) [zeros(k-1,1);1;zeros(n-k,1)];

V = e(34,70); %rand(10,1);
V = [0; V; flip(V)];
m = (length(V)+1)/2;
rV = ifft(V);
N = (length(rV)+1)/2;
r = rV(1:m+1);
W = diag([1 2*ones(1,m)]);
C = chol(W);
Cinv = inv(C);
J = cellmat(m,1,m+1,m+1);
for k = 1:m+1
    J{k} = [zeros(m+1-(k-1),k-1), eye(m+1-(k-1)); zeros(k-1,m+1)];
end

clear nu
cvx_begin sdp
    variable nu(m+1)
    variable Q(m+1,m+1) symmetric
    minimize(norm(nu))
    Q >= 0;
    for k = 1:m+1
        trace(1/2*(J{k}+J{k}')*Q) + Cinv(k,:)*nu == r(k);
    end
cvx_end

rr = zeros(m+1,1);
for k = 1:m+1
    rr(k) = trace(1/2*(J{k}+J{k}')*Q);
end
r_full = [flip(r(2:end)); r];
rr_full = [flip(rr(2:end)); rr];

figure
stem(abs(fft(r_full)));
hold on
stem(abs(fft(rr_full)));
legend('|fft(r target)|','|fft(r app)|');

% R = 0;
% for i = 1:length(rr_full)
%     R = R + rr_full(i)*z^(-i);
% end
% R = minreal(z^(m+1)*R,1e-5);
% [G,S] = spectralfact(R);
% G = G*sqrt(S);
% [b,~] = tfdata(G);
% b = b{1}';
% b = [b; zeros(m,1)];

roots_r = roots(rr_full); 
I = find( abs(roots_r) < 1 );
nmp_roots = roots_r(I);
b = poly(nmp_roots)';
b = b*sqrt(max(rr_full)/max(conv(b,flip(b))));
b = [b; zeros(2*N-length(b),1)];

x = sin(2*pi*33/length(V)*(1:(m+1)))';
x = norm(b)/norm(x)*x;
x = [x; zeros(2*N-length(x),1)];

figure
stem(V);
hold on;
stem(abs(fft(b)).^2);
stem(abs(fft(x)).^2);
legend('V','|fft(b)|^2','|fft(x)|^2');
