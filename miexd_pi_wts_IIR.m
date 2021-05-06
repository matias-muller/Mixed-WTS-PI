clear;
% close all;
home;

% Metadata
K = 200;             % Number of arms {3, 10, 17, 24,... 7*L+3}
T = 1e2;%            % Number iterative experiments
M = 1000;           % Number of MC samples in WTS
T_lim = 50;
Nexp = 1e0;         % Number of repetitions to average out
sigma = 0.8;          % Standard deviation of noise
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
% load('Num_fit.mat');        % Filter coefficients (via fdatool)
% a = [1 zeros(1,length(Num)-1)];
% b = Num;
L = 61; %length of the orginal filter
a = [1, -0.9854, 0.8187];
ahat = [1 zeros(1,L-1)];
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
    u_next_exp = u_next_exp*sqrt(N)/norm(u_next_exp,2);
    u_next_exp_white = u_next_exp;
    
    Phi = zeros(N,L);
    ghat = zeros(L,1);
    Phi_pi = zeros(N,L);
    ghat_pi = zeros(L,1);
    
    flag = 0;
    flag_white = 0;
    
    % Game
    for t = 1:T
            %% ================= WTS ========================
            
            % Design input signal
            U_wts = [0; sqrt(p_wts); flip(sqrt(p_wts))];
            u_wts = real(nifft(U_wts));
            
            % Add cyclic prefix and perform experiment
            u_wts_CP = [u_wts(N-L+2:N);u_wts];
            e_wts_CP = sigma*randn(N+L-1,1);
            e_wts = e_wts_CP(end-N+1:end);
            y_wts_CP = filter(b,a,u_wts_CP) + e_wts_CP;
            
            % Remove prefixes and collect rewards
            y_wts = y_wts_CP(L:end);
            U_wts = nfft(u_wts);
            Y_wts = nfft(y_wts);
            X_wts = Y_wts./U_wts;
            X_wts = X_wts(2:K+1);
            X_re = real(X_wts);
            X_im = imag(X_wts);
%             X_re = mu_re + randn(K,1)*sigma./sqrt(2*p_wts);
%             X_im = mu_im + randn(K,1)*sigma./sqrt(2*p_wts);

            % Update statistics
            sp_wts  = sp_wts + p_wts;
            spw_re_wts = spw_re_wts + p_wts.*X_re;
            spw_im_wts = spw_im_wts + p_wts.*X_im;
            
            m_re_post_wts   = lambda^2*spw_re_wts ./ (sigma^2 + lambda^2*sp_wts); % Posterior mean of all arms
            m_im_post_wts   = lambda^2*spw_im_wts ./ (sigma^2 + lambda^2*sp_wts);
            v_post_wts      = lambda^2 ./ (1 + lambda^2*sp_wts/sigma^2);   % Posterior variance of all arms

            % Obtain MC samples from the posterior mean of each arm
            samples_re  = diag(sqrt(v_post_wts))*randn(K,M) + m_re_post_wts*ones(1,M);
            samples_im  = diag(sqrt(v_post_wts))*randn(K,M) + m_im_post_wts*ones(1,M);
            samples     = samples_re + sqrt(-1)*samples_im;

            % Find the empirical distribution of the best arm (given data)
            max_arm = zeros(K,1);
            for j = 1:M
            [~, indmax] = max(abs(samples(:,j)));
            max_arm(indmax) = max_arm(indmax) + 1;
            end
            p_wts = max_arm/M + eps;
            p_wts = p_wts/sum(p_wts);
            p_wts = p_wts*N/2;
            
            % Regret
            
            % Estimation
            beta_wts(t,n) = 2*abs(sum(spw_re_wts + J*spw_im_wts))/(t*N);
            
            %% ================= PI ========================
            % Update input signal
            u_pi = u_next_exp;
            
            % Add cyclic prefix and run experiment
            u_pi_CP = [u_pi(N-L+2:N);u_pi];
%              u_pi_CP = [zeros(L-1,1); u_pi];
%             e_pi_CP = sigma*randn(N+L-1,1);
            y_pi_CP = filter(b,a,u_pi_CP) + e_wts_CP;
            
            % Remove prefix
            y_pi = y_pi_CP(L:end);
            
            % Design input signal for next experiment
            ytilde = flip(y_pi);
            mu = norm(ytilde,2)/sqrt(N);
            u_next_exp = ytilde/mu;
            
            % Estimation
            beta_pi(t,n) = u_next_exp'*ytilde/N;
            
            %% ================= MIXED ========================
            
            % Play either PI or WTS
            if( t < T_lim )
                % Play PI (already played), map PI data to WTS rewards
                U_mixed = nfft(u_pi);
                Y_mixed = nfft(y_pi);
                X_mixed = Y_mixed./U_mixed;
                X_mixed = X_mixed(2:K+1);
                X_re = real(X_mixed);
                X_im = imag(X_mixed);
                p_mixed = abs(U_mixed).^2;
                p_mixed = p_mixed(2:K+1);
     
            else % Play WTS
                % Design input signal
                U_mixed = [0; sqrt(p_mixed); flip(sqrt(p_mixed))];
                u_mixed = real(nifft(U_mixed));

                % Add cyclic prefix and perform experiment
                u_mixed_CP = [u_mixed(N-L+2:N);u_mixed];
                y_mixed_CP = filter(b,a,u_mixed_CP) + e_wts_CP;

                % Remove prefixes and collect rewards
                y_mixed = y_mixed_CP(L:end);
                U_mixed = nfft(u_mixed);
                Y_mixed = nfft(y_mixed);
                X_mixed = Y_mixed./U_mixed;
                X_mixed = X_mixed(2:K+1);
                X_re = real(X_mixed);
                X_im = imag(X_mixed);
            end
            
            % Update Mixed statistics
            % if T < T_lim, then continuously update statistics. When T =
            % T_lim, reset the statistics and update them starting form a
            % prior G ~ N( m_post(T_lim-1), v_post(T_lim-1))
            if( t < T_lim )
                sp_mixed  = sp_mixed + p_mixed;
                spw_re_mixed = spw_re_mixed + p_mixed.*X_re;
                spw_im_mixed = spw_im_mixed + p_mixed.*X_im;

                m_re_post_mixed = 2*lambda^2*spw_re_mixed ./ (sigma^2 + 2*lambda^2*sp_mixed); % Posterior mean of all arms
                m_im_post_mixed = 2*lambda^2*spw_im_mixed ./ (sigma^2 + 2*lambda^2*sp_mixed); 
                v_post_mixed = lambda^2 ./ (1 + 2*lambda^2*sp_mixed/sigma^2);   % Posterior variance of all arms
            else
                if (flag == 0)
                    flag = 1;
                    mu_o_re = m_re_post_mixed;
                    mu_o_im = m_im_post_mixed;
                    lambda_o2 = v_post_mixed;
                    
                    % Reset weights
                    sp_mixed  = zeros(K,1);   
                    spw_re_mixed = zeros(K,1);
                    spw_im_mixed = zeros(K,1);
                    spw_re_cent_mixed = zeros(K,1);
                    spw_im_cent_mixed = zeros(K,1);
                end
                
                sp_mixed = sp_mixed + p_mixed;
                spw_re_mixed = spw_re_mixed + p_mixed.*X_re;
                spw_im_mixed = spw_im_mixed + p_mixed.*X_im;
                spw_re_cent_mixed = spw_re_cent_mixed + p_mixed.*(X_re - mu_o_re);
                spw_im_cent_mixed = spw_im_cent_mixed + p_mixed.*(X_im - mu_o_im);

                m_re_post_mixed = mu_o_re + 2*lambda_o2.*spw_re_cent_mixed ./ (sigma^2 + 2*lambda_o2.*sp_mixed); % Posterior mean of all arms
                m_im_post_mixed = mu_o_im + 2*lambda_o2.*spw_im_cent_mixed ./ (sigma^2 + 2*lambda_o2.*sp_mixed); 
                v_post_mixed = lambda_o2 ./ (1 + 2*lambda_o2.*sp_mixed/sigma^2);   % Posterior variance of all arms
            end
                
            
            % Design when WTS is being played
            if( t >= T_lim - 1 )
                % Obtain MC samples from the posterior mean of each arm
                samples_re  = diag(sqrt(v_post_mixed))*randn(K,M) + m_re_post_mixed*ones(1,M);
                samples_im  = diag(sqrt(v_post_mixed))*randn(K,M) + m_im_post_mixed*ones(1,M);
                samples     = samples_re + sqrt(-1)*samples_im;

                % Find the empirical distribution of the best arm given the data
                max_arm = zeros(K,1);
                for j = 1:M
                [~, indmax] = max(abs(samples(:,j)));
                max_arm(indmax) = max_arm(indmax) + 1;
                end
                p_mixed = max_arm/M + eps;
                p_mixed = p_mixed/sum(p_mixed);
                p_mixed = p_mixed*N/2;
            end
            
            % Estimation
            if( t < T_lim )
                % Estimate using PI
                beta_mixed(t,n) = u_next_exp'*ytilde/N;
            else
                % Estimate using WTS
                beta_mixed(t,n) = abs(sum(spw_re_mixed + J*spw_im_mixed))/(sum(sp_mixed));
            end

            
%             %% ================= White Noise MIXED ==================
%             % Play either white noise or WTS
%             if( t < T_lim )  
%                 % Update input signal
%                 u_white = u_next_exp_white;
% 
%                 % Add cyclic prefix and run experiment
%                 u_white_CP = [u_white(N-L+2:N);u_white];
%                 %             u_pi_CP = [zeros(L-1,1); u_pi];
%                 y_white_CP = filter(b,a,u_white_CP) + e_wts_CP;
% 
%                 % Remove prefix
%                 y_white = y_white_CP(L:end);
% 
%                 % Design input signal for next experiment
%                 ytilde_white = flip(y_white);
%                 mu_white = norm(ytilde_white,2)/sqrt(N);
%                 u_next_exp_white = ytilde_white/mu_white;
%                 
%                 % Map data to WTS rewards
%                 U_white = nfft(u_white);
%                 Y_white = nfft(y_white);
%                 X_white = Y_white./U_white;
%                 X_white = X_white(2:K+1);
%                 X_re = real(X_white);
%                 X_im = imag(X_white);
%                 p_white = abs(U_white).^2;
%                 p_white = p_white(2:K+1);
%             else
%                 % Design input signal
%                 U_white = [0; sqrt(p_white); flip(sqrt(p_white))];
%                 u_white = real(nifft(U_white));
% 
%                 % Add cyclic prefix and perform experiment
%                 u_white_CP = [u_white(N-L+2:N);u_white];
%                 y_white_CP = filter(b,a,u_white_CP) + e_wts_CP;
% 
%                 % Remove prefixes and collect rewards
%                 y_white = y_white_CP(L:end);
%                 U_white = nfft(u_white);
%                 Y_white = nfft(y_white);
%                 X_white = Y_white./U_white;
%                 X_white = X_white(2:K+1);
%                 X_re = real(X_white);
%                 X_im = imag(X_white);
%             end
% 
%             % Update Mixed statistics
%             if( t < T_lim )
%                 sp_white  = sp_white + p_white;
%                 spw_re_white = spw_re_white + p_white.*X_re;
%                 spw_im_white = spw_im_white + p_white.*X_im;
% 
%                 m_re_post_white = 2*lambda^2*spw_re_white ./ (sigma^2 + 2*lambda^2*sp_white); % Posterior mean of all arms
%                 m_im_post_white = 2*lambda^2*spw_im_white ./ (sigma^2 + 2*lambda^2*sp_white); 
%                 v_post_white = lambda^2 ./ (1 + 2*lambda^2*sp_white/sigma^2);   % Posterior variance of all arms
%             else
%                 if (flag_white == 0)
%                     flag_white = 1;
%                     mu_o_re_white = m_re_post_white;
%                     mu_o_im_white = m_im_post_white;
%                     lambda_o2_white = v_post_white;
%                     % Reset weights
%                     sp_white  = zeros(K,1);     % Sum of previous weights
%                     spw_re_white = zeros(K,1);  % Sum of previous weighted rewards
%                     spw_im_white = zeros(K,1);  % Sum of previous weighted rewards
%                     spw_re_cent_white = zeros(K,1);
%                     spw_im_cent_white = zeros(K,1);
%                 end
%                 
%                 sp_white = sp_white + p_white;
%                 spw_re_white = spw_re_white + p_white.*X_re;
%                 spw_im_white = spw_im_white + p_white.*X_im;
%                 spw_re_cent_white = spw_re_cent_white + p_white.*(X_re - mu_o_re_white);
%                 spw_im_cent_white = spw_im_cent_white + p_white.*(X_im - mu_o_im_white);
% 
%                 m_re_post_white = mu_o_re_white + 2*lambda_o2_white.*spw_re_cent_white ./ (sigma^2 + 2*lambda_o2_white.*sp_white); % Posterior mean of all arms
%                 m_im_post_white = mu_o_im_white + 2*lambda_o2_white.*spw_im_cent_white ./ (sigma^2 + 2*lambda_o2_white.*sp_white); 
%                 v_post_white = lambda_o2_white ./ (1 + 2*lambda_o2_white.*sp_white/sigma^2);   % Posterior variance of all arms
%             end
%                 
%             
%             % Update weights only if WTS is played
%             if( t >= T_lim - 1 )
%                 % Obtain MC samples from the posterior mean of each arm
%                 samples_re  = diag(sqrt(v_post_white))*randn(K,M) + m_re_post_white*ones(1,M);
%                 samples_im  = diag(sqrt(v_post_white))*randn(K,M) + m_im_post_white*ones(1,M);
%                 samples     = samples_re + sqrt(-1)*samples_im;
% 
%                 % Find the empirical distribution of the best arm given the data
%                 max_arm = zeros(K,1);
%                 for j = 1:M
%                     [~, indmax] = max(abs(samples(:,j)));
%                     max_arm(indmax) = max_arm(indmax) + 1;
%                 end
%                 p_white = max_arm/M + eps;
%                 p_white = p_white/sum(p_white);
%                 p_white = p_white*N/2;
%             end
%             
%             % Estimation
%             if t < T_lim
%                 beta_white(t,n) = 0; %max(abs(m_re_post_white + J*m_im_post_white));
%             else
%                 beta_white(t,n) = abs(sum(spw_re_white + J*spw_im_white))/(sum(sp_white));
%             end
% 
%     
        %% ====== True PI ========
        if t == 1
            u_pi_true = randn(N,1);
            u_pi_true = u_pi_true*sqrt(N)/norm(u_pi_true);
        else
            u_pi_true = u_next_true;
        end
        y_pi_true = filter(b,a,u_pi_true) + e_wts;
        ytilde_true = flip(y_pi_true);
        mu_true = norm(ytilde_true)/sqrt(N);
        u_next_true = ytilde_true/mu_true;
        beta_pi_true(t,n) = norm(y_pi_true)/sqrt(N);
            
        %% ====== RLS with PI ========
        YY_new_pi = y_pi_true(1:N);
        Phi_new_pi = toeplitz(u_pi_true, [u_pi_true(1), zeros(1,L-1)]);
        Phi_pi = Phi_pi + Phi_new_pi;
        ghat_pi = ghat_pi + (Phi_pi'*Phi_pi)\(Phi_new_pi'*YY_new_pi - Phi_new_pi'*Phi_new_pi*ghat_pi);
        beta_rls_pi(t,n) = norm(tf(ghat_pi',ahat,1),'inf',1e-10);
        
        %% ======= RLS with white noise ======
        u_rls = randn(N,1);
        u_rls = sqrt(N)*u_rls/norm(u_rls);
        y_rls = filter(b,a,u_rls) + e_wts;
        YY_new = y_rls(1:N);
        Phi_new = toeplitz(u_rls, [u_rls(1), zeros(1,L-1)]);
        Phi = Phi + Phi_new;
        ghat = ghat + (Phi'*Phi)\(Phi_new'*YY_new - Phi_new'*Phi_new*ghat);
        beta_rls(t,n) = norm(tf(ghat',ahat,1),'inf',1e-10);

        %% Mean squared error
        mse_wts(t,n) = (beta - beta_wts(t,n))^2;
        mse_pi(t,n) = (beta - beta_pi(t,n))^2;
        mse_mixed(t,n) = (beta - beta_mixed(t,n))^2;
        %       mse_white(t,n) = (beta - beta_white(t,n))^2;
        mse_rls_pi(t,n) = (beta - beta_rls_pi(t,n))^2;
        mse_rls(t,n) = (beta - beta_rls(t,n))^2;
        mse_pi_true(t,n) = (beta - beta_pi_true(t,n))^2;

        %% Figures
    % %   Comment this to get the result faster.
    % %   Plot: Both algorithms live
    %   subplot(2,1,1)
    %   freq = linspace(0, K-1/length(Gplot), length(Gplot));
    %   plot(freq,abs(Gplot));
    %   hold on;
    %   stem(p);
    %   stem(index_maxmu,p(index_maxmu),'k','linewidth',4);
    %   hold off;
    %   legend('G(e^{2\pi j k/N})','p','best','Location',[0.2 0.8 0.1 0.1]);
    %   ylim([0 1]);
    %   title('New algorithm');
    %   subplot(2,1,2)
    %   plot(p_ts);
    %   hold on;
    %   stem(arm, p_ts(arm),'r','linewidth',4);
    %   hold off;
    %   legend('p(A*|F_t)','A_t');
    %   title('Thompson Sampling');
    %   pause(0.01);
    %   
        
    
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
% pause(0.1);


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
mse_rls_avg = sum(mse_rls,2)/Nexp;
mse_rls_pi_avg = sum(mse_rls_pi,2)/Nexp;
mse_pi_true_avg = sum(mse_pi_true,2)/Nexp;

% SAVE
save('results_code.mat');
linewidthval = 1; %1.5
figure
hold on
plot(mse_wts_avg,'linewidth',linewidthval);
plot(mse_pi_avg,'linewidth',linewidthval);
plot(mse_mixed_avg,'linewidth',linewidthval);
plot(mse_rls_pi_avg,'linewidth',linewidthval);
plot(mse_rls_avg,'linewidth',linewidthval);
plot(mse_pi_true_avg,'linewidth',linewidthval);
legend('wts','pi','mixed','rls pi','rls','pi true');
%  set(gca, 'YScale', 'log')