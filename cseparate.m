function [xhat, xhat_all] = cseparate(x, M, N, H)
% complex-valued frequency domain separation by independent components
% using relative phase representation
%
% inputs:
%    x - the audio signal to separate (1 row)
%    M - the number of sources to extract
%
% output:
%    xhat - the separated signals (M rows)
%    xhat_all - the M separated signals mixed (1 row)
%
% Copyright (C) 2014 Michael A. Casey, Bregman Media Labs, 
% Dartmouth College All Rights Reserved

if nargin<2,
    M = 20;
end

if nargin<3,
    N = 4096;
end

if nargin<4,
    hop = 1024;
end

Asig = stft(x, N, hop, 0, 'hamming');
[rows,cols] = size(Asig);

% Relative phase calculations (using the phase vocoder trick)

% 1. Expected phase advance (procession) for each center frequency
dphi = zeros(1,N/2+1);
dphi(2:(1 + N/2)) = (2*pi*hop)./(N./(1:(N/2)));

% 2. phase deviations from expected procession of center frequencies
A = diff(angle(Asig),1,2); % Complete Phase Map
[rows,cols]=size(A);
U = [angle(Asig(:,1)), A - repmat(dphi',1,cols)];
U = U - round(U./(2*pi))*2*pi; % phase unwrapping

if(0) % Sanity Check: test identity reconstruction
    [rows,cols]=size(U);
    UU = cumsum(U + repmat(dphi',1,cols),2);
    Ahat = abs(Asig).*exp(1i*UU);
    xhat = stft(Ahat, nfft, hop, 0, 'hamming');
end

% OK, we have verified that U encodes relative phase correctly
% Make relative phase complex signal
[rows,cols]=size(U);
Ahat = abs(Asig).*exp(1i*U);

% separate in the relative phase complex spectrum representation
[Ae,Se]=cjade(Ahat.',M);

AS = (Ae*Se).'; % Non hermitian transpose avoids complex conjugation
UU = cumsum(angle(AS) + repmat(dphi',1,cols),2);
Shat = abs(AS).*exp(1i*UU);
xhat_all = stft(Shat, N, hop, 0, 'hamming');
xhat = zeros(M, size(xhat_all,2)); % pre allocate

% Reconstruct magnitude and also 'Fourier' phase from relative phase
for k = 1:M
    AS = (Ae(:,k)*Se(k,:)).';
    UU = cumsum(angle(AS) + repmat(dphi',1,cols),2);
    Shat = abs(AS).*exp(1i*UU);
    xhat(k,:) = stft(Shat, N, hop, 0, 'hamming');
end


