clear;
% close all;
home;

% Metadata
K = 200;             % Number of arms {3, 10, 17, 24,... 7*L+3}
T = 1e2;            % Number iterative experiments
M = 1000;           % Number of MC samples in WTS
T_lim = 25;
Nw = 100;
Nexp = 1e0;         % Number of repetitions to average out
sigma = 0.8;        % Standard deviation of noise
lambda = 1;         % Prior standard deviation of mean rewards
N = 2*K+1;          % length of the input for power iterations
eps = 1e-15;        % regulations term for p
tolerance = 1e-7;   % treshold to consider a frequency in the Hinf-norm est.
precision = 1e-5;
J = sqrt(-1);

fprintf('Initializing...\n');
fprintf('T = %d, Nexp = %d\n', T, Nexp);

mse_100_wts = zeros(Nw,1);
mse_100_pi = mse_100_wts;
mse_100_mixed = mse_100_wts;
mse_100_rls = mse_100_wts;
mse_100_rls_pi = mse_100_wts;
mse_100_pi_true = mse_100_wts;
www =  0.25*pi + (0.95*pi-0.25*pi)*rand(Nw,1); %linspace(0.01,0.99*pi,Nw);%

for nw = 1:Nw
    
%     ww = 2*pi*nw/Nw;
ww = www(nw);
r = 0.9;
L = N+1; %length of the orginal filter
LL = floor(sqrt(2*N*T/log(2*N*T))); % length of LS FIR filter
a =  [1, -2*r*cos(ww), r^2]; %[1, -0.9854, 0.8187]; %;[1, -0.21969, 0.910569, -0.207156, 0.692948];
ahat = [1 zeros(1,LL-1)];
b = (1-r)*sqrt((1-r)^2*cos(ww)^2 + (1+r)^2*sin(ww)^2); %[0, 0, 0, 0.2155, 0.2012]; %

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
            beta_wts(t,n) = abs(sum(spw_re_wts + J*spw_im_wts))/(sum(sp_wts));
            
            %% ================= PI FOR MIXED ALG. ========================
            % Update input signal
            u_pi = u_next_exp;
            
            % Add cyclic prefix and run experiment
            u_pi_CP = [u_pi(N-L+2:N);u_pi];
%              u_pi_CP = [zeros(L-1,1); u_pi];
%             e_pi_CP = sigma*randn(N+L-1,1);
            y_pi_CP = filter(b,a,u_pi_CP) + e_wts_CP;
            
            % Retreive prefix to design next experiment and sufix to update
            % wts posteriors
            y_pi_prefix = y_pi_CP(1:N);
            y_pi_sufix = y_pi_CP(L:end);
            
            % Design input signal for next experiment
            ytilde = flip(y_pi_prefix);
            mu = norm(ytilde,2)/sqrt(N);
            u_next_exp = ytilde/mu;
            
            % Estimation based on complete measurements
            beta_pi(t,n) = norm(y_pi_CP)/sqrt(2*N); %u_next_exp'*ytilde/N;
            
            %% ================= MIXED ========================
            
            % Play either PI or WTS
            if( t < T_lim )
                % Play PI (already played), map PI data to WTS rewards
                U_mixed = nfft(u_pi);
                Y_mixed = nfft(y_pi_sufix);
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
                beta_mixed(t,n) = beta_pi(t,n); %u_next_exp'*ytilde/N;
            else
                % Estimate using WTS
                beta_mixed(t,n) = abs(sum(spw_re_mixed + J*spw_im_mixed))/(sum(sp_mixed));
            end

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
        beta_rls_pi(t,n) = norm(tf(ghat_pi',ahat,1),'inf',precision);
        
        %% ======= RLS with white noise ======
        u_rls = randn(2*N,1);
        u_rls = sqrt(2*N)*u_rls/norm(u_rls);
        y_rls = filter(b,a,u_rls) + e_wts_CP;
        YY_new = y_rls(1:2*N);
        Phi_new = toeplitz(u_rls, [u_rls(1), zeros(1,LL-1)]);
        Phi = Phi + Phi_new;
        ghat = ghat + (Phi'*Phi)\(Phi_new'*YY_new - Phi_new'*Phi_new*ghat);
        beta_rls(t,n) = norm(tf(ghat',ahat,1),'inf',precision);

        %% Mean squared error
        mse_wts(t,n) = (beta - beta_wts(t,n))^2;
        mse_pi(t,n) = (beta - beta_pi(t,n))^2;
        mse_mixed(t,n) = (beta - beta_mixed(t,n))^2;
        %       mse_white(t,n) = (beta - beta_white(t,n))^2;
        mse_rls_pi(t,n) = (beta - beta_rls_pi(t,n))^2;
        mse_rls(t,n) = (beta - beta_rls(t,n))^2;
        mse_pi_true(t,n) = (beta - beta_pi_true(t,n))^2;
       
    
    end % T iterations
    
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

mse_100_wts(nw) = mse_wts_avg(end);
mse_100_pi(nw) = mse_pi_avg(end);
mse_100_mixed(nw) = mse_mixed_avg (end);
mse_100_rls(nw) = mse_rls_avg(end);
mse_100_rls_pi(nw) = mse_rls_pi_avg(end);
mse_100_pi_true(nw) = mse_pi_true_avg(end);

end     % Nw

% %% Figs
% save('results_code.mat');
% linewidthval = 1; %1.5
% txtsz = 14;
% figure
% hold on
% plot(linspace(0,pi,Nw),mse_100_wts,'linewidth',linewidthval);
% plot(linspace(0,pi,Nw),mse_100_pi,'linewidth',linewidthval);
% plot(linspace(0,pi,Nw),mse_100_mixed,'linewidth',linewidthval);
% plot(linspace(0,pi,Nw),mse_100_rls,'linewidth',linewidthval);
% plot(linspace(0,pi,Nw),mse_100_rls_pi,'linewidth',linewidthval);
% plot(linspace(0,pi,Nw),mse_100_pi_true,'linewidth',linewidthval);
% legend('WTS','PI-CP','proposed','RLS-PI','RLS-white','PI-2T');
% box on
% set(gca, 'YScale', 'log')
% title('MSE after 100 iterations');
% xlabel('\omega^\star');
% ylabel('Mean squared error');

%% Boxplot Figs
save('results_code.mat');
linewidthval = 1; %1.5
txtsz = 14;
figure
hold on
data = [mse_100_wts,mse_100_pi,mse_100_mixed,mse_100_rls,mse_100_rls_pi,mse_100_pi_true];
boxplot(data);
%legend('WTS','PI-CP','proposed','RLS-PI','RLS-white','PI-2T');
xticklabels({'WTS','PI-CP','proposed','RLS-PI','RLS-white','PI-2T'});
box on
% set(gca, 'YScale', 'log')
title('MSE after 100 iterations');
ylabel('Mean squared error');
set(gca,'fontsize',14);