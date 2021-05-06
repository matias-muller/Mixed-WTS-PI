function X = nfft(x)
    N = length(x);
    X = fft(x)/sqrt(N);
end