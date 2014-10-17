[x,sr] = audioread('gmin.wav');
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



