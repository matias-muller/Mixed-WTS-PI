clear;
close all;
home;

% Metadata
K = 143;             % Number of arms {3, 10, 17, 24,... 7*L+3}
T = 500;             % Number of iterations % 20 000 per night, under 1e5 MC points. N=1000,M=1000 ~ 20sec.
T_lim = 25;
Nexp = 1e0;         % Number of experiments to compute a variance estimator
sigma = 1;          % Standard deviation of noise
lambda = 1;         % Prior standard deviation of mean rewards
N = 2*K+1;          % length of the input for power iterations
eps = 1e-15;        % regulations term for p
tolerance = 1e-7;   % treshold to consider a frequency in the Hinf-norm est.
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
load('Num_fit.mat');        % Filter coefficients (via fdatool)
a = [1 zeros(1,length(Num)-1)];
b = Num;
L = length(Num);
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

regret_wts = zeros(T,Nexp);
regret_pi = regret_wts;
regret_mixed = regret_wts;

beta_wts = regret_wts;
beta_pi = regret_wts;
beta_mixed = regret_wts;
beta_white = regret_wts;


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
    u_next_exp = u_next_exp*sqrt(N)/norm(u_next_exp,2);
    u_white = rand(N,1);
    u_white = u_white/norm(u_white,2);%*sqrt(L);
    
    flag = 0;
    flag_white = 0;
    
    % Game
    for t = 1:T
            %% ================= PI ========================
            % Update input signal
            u_pi = u_next_exp;
            
            % Add cyclic prefix and run experiment
            u_pi_CP = [u_pi(N-L+2:N);u_pi];
%             u_pi_CP = [zeros(L-1,1); u_pi];
            e_pi_CP = sigma*randn(N+L-1,1);
            y_pi_CP = filter(b,a,u_pi_CP) + e_pi_CP;
            
            % Remove prefix
            y_pi = y_pi_CP(L:end);
            
            % Design input signal for next experiment
            ytilde = flip(y_pi);
            mu = norm(ytilde,2)/sqrt(N);
            u_next_exp = ytilde/mu;
            
            % Estimation
            beta_pi(t,n) = u_next_exp'*ytilde/N;
            
            %% ================= MIXED ========================
            
                % Play PI (already played), map PI data to WTS rewards
            U_mixed = nfft(u_pi);
            Y_mixed = nfft(y_pi);
            X_mixed = Y_mixed./U_mixed;
            X_mixed = X_mixed(2:K+1);
            X_re = real(X_mixed);
            X_im = imag(X_mixed);
            p_mixed = abs(U_mixed).^2;
            p_mixed = p_mixed(2:K+1);
            
            X_re_1(t) = X_re(1);
            X_im_1(t) = X_im(1);
            
            % Update Mixed statistics
            % if T < T_lim, then continuously update statistics. When T =
            % T_lim, reset the statistics and update them starting form a
            % prior G ~ N( m_post(T_lim-1), v_post(T_lim-1))
                sp_mixed  = sp_mixed + p_mixed;
                spw_re_mixed = spw_re_mixed + p_mixed.*X_re;
                spw_im_mixed = spw_im_mixed + p_mixed.*X_im;

                m_re_post_mixed = 2*lambda^2*spw_re_mixed ./ (sigma^2 + 2*lambda^2*sp_mixed); % Posterior mean of all arms
                m_im_post_mixed = 2*lambda^2*spw_im_mixed ./ (sigma^2 + 2*lambda^2*sp_mixed); 
                v_post_mixed = lambda^2 ./ (1 + 2*lambda^2*sp_mixed/sigma^2);   % Posterior variance of all arms

    
      %% Mean squared error
      mse_wts(t,n) = (beta - beta_wts(t,n))^2;
      mse_pi(t,n) = (beta - beta_pi(t,n))^2;
      mse_mixed(t,n) = (beta - beta_mixed(t,n))^2;
      mse_white(t,n) = (beta - beta_white(t,n))^2;

        %% Figures
 
% subplot(3,1,1)
% plot(linspace(0,K+1/2,length(Gplot)),Gplot);
% hold on
% stem(abs(m_re_post_mixed + J*m_im_post_mixed));
% ylim([0 2]);
% xlim([0,K+1])
% hold off
% 
% subplot(3,1,2);
% plot(mu_re);
% hold on;
% stem(real(X_mixed));
% % plot(m_re_post_white);
% ylim([-1 1]);
% xlim([0,K+1])
% hold off
% 
% subplot(3,1,3)
% plot(mu_im);
% hold on
% stem(imag(X_mixed));
% % plot(m_im_post_white);
% ylim([-1 1]);
% xlim([0,K+1])
% hold off
% 
% pause(0.001);


% % %     subplot(3,1,1)
% % %     stem(p_mixed);
% % %     hold on;
% % %     stem(kstar,p_mixed(kstar),'r');
% % %     U = abs(fft(uu,4000));
% % %     U = U/max(U);
% % %     plot(linspace(0,K+1/2,2000),U(1:2000));
% % %     plot(linspace(0,K+1/2-1/Nplot,Nplot),Gplot);
% % %     ylim([0 1]);
% % %     xlim([0,K+1]);
% % %     title('p mixed');
% % %     hold off;  
% % %     
% % %     subplot(3,1,2)
% % %     stem(sqrt(m_re_post_wts.^2 + m_im_post_wts.^2));
% % %     hold on;
% % %     stem(kstar,sqrt(m_re_post_wts(kstar).^2 + m_im_post_wts(kstar).^2));
% % %     plot(linspace(0,K+1/2-1/Nplot,Nplot),Gplot);
% % %     ylim([0 1]);
% % %     xlim([0,K+1]);
% % %     plot(v_post_wts);
% % %     hold off;  
% % %     title('|m post re| wts');
% % %     
% % %     subplot(3,1,3)
% % %     stem(sqrt(m_re_post_mixed.^2 + m_im_post_mixed.^2));
% % %     hold on;
% % %     stem(kstar,sqrt(m_re_post_mixed(kstar).^2 + m_im_post_mixed(kstar).^2));
% % %     plot(linspace(0,K+1/2-1/Nplot,Nplot),Gplot);
% % %     ylim([0 1]);
% % %     xlim([0,K+1]);
% % %     plot(v_post_mixed);
% % %     hold off;  
% % %     title('|m post re| mixed');
% % % 
% % %     pause(0.5);
       
    
    end % T iterations
    
%     disp(n); 
%     
%     figure
%     plot(mse_wts(:,n),'linewidth',1.5);
%     hold on;
%     plot(mse_pi(:,n),'linewidth',1.5);
%     plot(mse_mixed(:,n),'linewidth',1.5);
%     plot(mse_white(:,n),'linewidth',1.5);
%     legend('wts','pi','mixed','white');
%     set(gca, 'YScale', 'log');
%     hold off;
%     pause(0.1);


print_perc(n,Nexp);
end     % Nexp experiments

%%
% Post processing
mse_wts_avg = sum(mse_wts,2)/Nexp;
mse_pi_avg = sum(mse_pi,2)/Nexp;
mse_mixed_avg = sum(mse_mixed,2)/Nexp;
mse_white_avg = sum(mse_white,2)/Nexp;

% SAVE
 save('results_code.mat');
 figure
 hold on
 plot(mse_wts_avg,'linewidth',1.5);
 plot(mse_pi_avg,'linewidth',1.5);
 plot(mse_mixed_avg,'linewidth',1.5);
 plot(mse_white_avg,'linewidth',1.5);
 legend('wts','pi','mixed','white');
 set(gca, 'YScale', 'log')