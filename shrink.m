function u = shrink(v)
    J = sqrt(-1);
    T = length(v);
    K = (T-1)/2;
    V = nfft(v);
    absV = abs(V(2:end)); % V = [V(jw1), V(jw2), ..., V(jwK), V'(jwK), V0(jw(K-1)), ..., V(jw1)]
    absV = sqrt(absV.^2*T/sum(absV.^2));
    F = dftmtx(T);
    Fh = F';
    A = Fh(end-K+1:end,2:end);
    B = diag(absV);
    
    clear z;
    myfun = @(z) (A*B*[exp(J*(z+conj(z))); exp(-J*flip(z+conj(z)))]);
%     options = optimoptions('fsolve','Display','none','PlotFcn',@optimplotfirstorderopt);
    options = optimoptions('fsolve','MaxIterations',1000,'MaxFunctionEvaluations',1e5);
    x = fsolve(myfun, pi*rand(K,1), options);
    angU = [0; [exp(J*(x+conj(x))); exp(-J*flip(x+conj(x)))]];
    U = [0; absV].*angU;
    u = real(nifft(U));
%     u = [u(1:K+1); zeros(K,1)];
end