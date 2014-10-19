% demo of complex-domain relative phase independent component analysis
%  for audio source separation from monophonic mixtures
%
% Michael A. Casey, Bregman Media Labs, 2014
%
sndfile = 'gmin.wav'
if exist('audioread'),
    [x,sr] = audioread(sndfile);
else
    [x,sr] = wavread(sndfile);
end
M = 7; % Number of components to separate
fprintf(1,['separating ', num2str(M), ' components from mixture...']);
[xhat, xhat_all] = cseparate(x, M);
fprintf(1,'done.\n');
fprintf(1,'hit any key to hear reconstructed mixture...\n');
pause()
soundsc(xhat_all,sr)
for k = 1:M
  fprintf(1,['hit any key to hear component',num2str(k),'...\n']);
  pause()
  soundsc(xhat(k,:),sr);
end
