function x = nifft(X)
    N = length(X);
    x = ifft(X)*sqrt(N);
end