clear;
close all;
home;

% Metadata
K = 10;    % Number of arms
N = 100;%1000;   % Number of iterations 
Nexp = 1000; % Number of experiments to compute a variance estimator
sigma = .1;  % Standard deviation of noise
lambda = 1; % Prior standard deviation of mean rewards
L = 100;    % length of the input for power iterations


beta_pi = zeros(N,Nexp);
mse_pi = beta_pi;

% Filter 2: Several maxima.
load('Num.mat');        % Filter coefficients (via fdatool)
a = [1 zeros(1,length(Num)-1)];
b = Num;
% Filter gains
G = freqz(b,a,K);
Nplot = 1000;
Gplot = abs(freqz(b,a,Nplot));

% True mean reward distribution
mu_re = real(G);
mu_im = imag(G);
[maxmu, index_maxmu] = max(abs(G));
beta = maxmu;

for n = 1:Nexp
u = sigma*randn(L,1);
u = u*sqrt(L)/norm(u);

    for t = 1:N
      % Power iterations
       y = filter(b,a,u) + sigma*randn(L,1);
       ytilde = flip(y);
       mu = norm(ytilde,2)/sqrt(L);
       u = ytilde/mu;
       beta_pi(t,n) = u'*ytilde/L;
       mse_pi(t,n) = (beta - beta_pi(t,n))^2;
    end

end

plot(sum(mse_pi,2)/Nexp);
% ylim([0 1.2e-3]);
 xlim([0 1000]);
% set(gca, 'YScale', 'log')
% SAVE
 save('results_code.mat');