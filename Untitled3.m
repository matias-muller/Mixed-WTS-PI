tic
clear;
% close all;
home;

% Metadata
K = 200;             % Number of arms {3, 10, 17, 24,... 7*L+3}
T = 1e2;%            % Number iterative experiments
M = 1000;           % Number of MC samples in WTS
T_lim = 25;
Nexp = 1e0;         % Number of repetitions to average out
sigma = 0.2;          % Standard deviation of noise
lambda = 1;         % Prior standard deviation of mean rewards
N = 2*K+1;          % length of the input for power iterations
eps = 1e-15;        % regulations term for p
tolerance = 1e-7;   % treshold to consider a frequency in the Hinf-norm est.
precision = 1e-5;
J = sqrt(-1);

fprintf('Initializing...\n');
fprintf('T = %d, Nexp = %d\n', T, Nexp);



% Filter properties
% Filter 1: One maximum.
% % r = 0.95;
% % w0 = 75/100*pi;
% % a = [1 -2*r*cos(w0) r^2];
% % b = 1-r;
% Filter 2: Several maxima.
% load('Num_fit.mat');        % Filter coefficients (via fdatool)
% a = [1 zeros(1,length(Num)-1)];
% b = Num;
L = N+1; %length of the orginal filter
LL = floor(sqrt(2*N*T/log(2*N*T))); % length of LS FIR filter
a = [1, -0.9854, 0.8187];
ahat = [1 zeros(1,LL-1)];
b = [0, 0.2155, 0.2012];
% Filter gains
G = freqz(b,a,2*K+1,'whole');   % Generates the 2*K+1 freq. resp. in the interval [0,2*pi)
G = G(2:K+1);           % Recovers the freq. resp. at the K freqs of interest
Nplot = 1000;
Gplot = abs(freqz(b,a,Nplot));  
Gz = tf(b,a,1);

%   True mean reward distribution
mu_re = real(G);
mu_im = imag(G);
[maxmu, index_maxmu] = max(abs(G));
beta = norm(Gz,Inf,1e-15);
kstar = index_maxmu;

%   Initialization
mse_wts = zeros(T,Nexp);
mse_pi = zeros(T,Nexp);
mse_mixed = zeros(T,Nexp);
mse_white = zeros(T,Nexp);
mse_rls = zeros(T,Nexp);
mse_rls_pi = zeros(T,Nexp);
mse_pi_true = zeros(T,Nexp);

beta_wts = zeros(T,Nexp);
beta_pi = beta_wts;
beta_mixed = beta_wts;
beta_rls = beta_wts;
beta_rls_pi = beta_wts;
beta_pi_true = beta_wts;

%% Game iterations
% figure
for n = 1:Nexp
    p_wts   = ones(K,1)*N/(2*K);          % Initial weights
    sp_wts  = zeros(K,1);           % Sum of previous weights
    spw_re_wts = zeros(K,1);        % Sum of previous weighted rewards
    spw_im_wts = zeros(K,1);        % Sum of previous weighted rewards

    p_mixed   = ones(K,1)*N/(2*K);        % Initial weights
    sp_mixed  = zeros(K,1);         % Sum of previous weights
    spw_re_mixed = zeros(K,1);      % Sum of previous weighted rewards
    spw_im_mixed = zeros(K,1);      % Sum of previous weighted rewards
    spw_re_cent_mixed = zeros(K,1); % Centered sum of previous weighted rewards
    spw_im_cent_mixed = zeros(K,1); % Centered sum of previous weighted rewards
    
    p_white   = ones(K,1)/K;        % Initial weights
    sp_white  = zeros(K,1);         % Sum of previous weights
    spw_re_white = zeros(K,1);      % Sum of previous weighted rewards
    spw_im_white = zeros(K,1);      % Sum of previous weighted rewards
    spw_re_cent_white = zeros(K,1); % Centered sum of previous weighted rewards
    spw_im_cent_white = zeros(K,1); % Centered sum of previous weighted rewards

    arm = 1;
    
    u_next_exp = rand(N,1);
    u_next_exp = u_next_exp*sqrt(N/2)/norm(u_next_exp,2);
    u_next_exp_white = u_next_exp;
    
    Phi = zeros(2*N,LL);
    ghat = zeros(LL,1);
    Phi_pi = zeros(2*N,LL);
    ghat_pi = zeros(LL,1);
    
    flag = 0;
    flag_white = 0;
    
    % Game
    for t = 1:T
           
            e_wts_CP = sigma*randn(N+L-1,1);
            e_wts = e_wts_CP(end-N+1:end);
            
        %% ====== True PI ========
        if t == 1
            u_pi_true = randn(2*N,1);
            u_pi_true = u_pi_true*sqrt(2*N)/norm(u_pi_true);
        else
            u_pi_true = u_next_true;
        end
        y_pi_true = filter(b,a,u_pi_true) + e_wts_CP;
        ytilde_true = flip(y_pi_true);
        mu_true = norm(ytilde_true)/sqrt(2*N);
        u_next_true = ytilde_true/mu_true;
        beta_pi_true(t,n) = norm(y_pi_true)/sqrt(2*N);
            
        %% ====== RLS with PI ========
        YY_new_pi = y_pi_true(1:2*N);
        Phi_new_pi = toeplitz(u_pi_true, [u_pi_true(1), zeros(1,LL-1)]);
        Phi_pi = Phi_pi + Phi_new_pi;
        ghat_pi = ghat_pi + (Phi_pi'*Phi_pi)\(Phi_new_pi'*YY_new_pi - Phi_new_pi'*Phi_new_pi*ghat_pi);
        beta_rls_pi(t,n) = norm(minreal(tf(ghat_pi',ahat,1)),'inf',precision);
        
        %% ======= RLS with white noise ======
        u_rls = randn(2*N,1);
        u_rls = sqrt(2*N)*u_rls/norm(u_rls);
        y_rls = filter(b,a,u_rls) + e_wts_CP;
        YY_new = y_rls(1:2*N);
        Phi_new = toeplitz(u_rls, [u_rls(1), zeros(1,LL-1)]);
        Phi = Phi + Phi_new;
        ghat = ghat + (Phi'*Phi)\(Phi_new'*YY_new - Phi_new'*Phi_new*ghat);
        beta_rls(t,n) = norm(minreal(tf(ghat',ahat,1),1e-3),'inf',precision);

        %% Mean squared error
        mse_rls_pi(t,n) = (beta - beta_rls_pi(t,n))^2;
        mse_rls(t,n) = (beta - beta_rls(t,n))^2;
        mse_pi_true(t,n) = (beta - beta_pi_true(t,n))^2;

    end % T iterations
    

print_perc(n,Nexp);
end     % Nexp experiments

toc

%%
% Post processing
mse_white_avg = sum(mse_white,2)/Nexp;
mse_rls_avg = sum(mse_rls,2)/Nexp;
mse_rls_pi_avg = sum(mse_rls_pi,2)/Nexp;
mse_pi_true_avg = sum(mse_pi_true,2)/Nexp;

% SAVE
save('results_code.mat');
linewidthval = 1; %1.5
figure
hold on
plot(mse_rls_pi_avg,'linewidth',linewidthval);
plot(mse_rls_avg,'linewidth',linewidthval);
plot(mse_pi_true_avg,'linewidth',linewidthval);
legend('rls pi','rls','pi true');
%  set(gca, 'YScale', 'log')