import numpy as np
from bregman.suite import *
from cjade import cjade
from scipy.optimize import curve_fit

def cseparate(x, M=None, N=4096, H =1024, W=4096, max_iter=200, pre_emphasis=True):
    """
    complex-valued frequency domain separation by independent components
    using relative phase representation
    
    inputs:
      x - the audio signal to separate (1 row)
      M - the number of sources to extract
    options:
      N - fft length in samples [4096]
      H - hop size in samples   [1024]
      W - window length in samples (fft padded with N-W zeros) [4096]
      max_iter - maximum JADE ICA iterations [200]
      pre_emphasis - apply an exponential spectral pre-emphasis filter [False]
    output:
      xhat - the separated signals (M rows)
      xhat_all - the M separated signals mixed (1 row)
    
    Copyright (C) 2014 Michael A. Casey, Bregman Media Labs, 
    Dartmouth College All Rights Reserved
    """
    def pre_func(x, a, b, c):
    	return a * np.exp(-b * x) + c

    M = 20 if M is None else M

    phs_rec = lambda rp,dp: (np.angle(rp)+np.tile(np.atleast_2d(dp).T,rp.shape[1])).cumsum(1)

    F = LinearFrequencySpectrum(x, nfft=N, wfft=W, nhop=H)
    U = F._phase_map()    
    XX = np.absolute(F.STFT)
    if pre_emphasis:
    	xx = np.arange(F.X.shape[0])
    	yy = XX.mean(1)
    	popt, pcov = curve_fit(pre_func, xx, yy)
    	XX = (XX.T * (1/pre_func(xx,*popt))).T
    X = XX * np.exp(1j * np.array(F.dPhi)) # Relative phase STFT

    A,S = cjade(X.T, M, max_iter) # complex-domain JADE by J. F. Cardoso

    AS = np.array(A*S).T # Non hermitian transpose avoids complex conjugation
    X_hat = np.absolute(AS)
    if pre_emphasis:
		X_hat = (XX.T * (pre_func(xx,*popt))).T
    Phi_hat = phs_rec(AS, F.dphi)
    x_hat_all = F.inverse(X_hat=X_hat, Phi_hat=Phi_hat, usewin=True)
    
    x_hat = []
    for k in np.arange(M):
        AS = np.array(A[:,k]*S[k,:]).T
        X_hat = np.absolute(AS)
        if pre_emphasis:
			X_hat = (XX.T * (pre_func(xx,*popt))).T
        Phi_hat = phs_rec(AS, F.dphi)
        x_hat.append(F.inverse(X_hat=X_hat, Phi_hat=Phi_hat, usewin=True))

    return x_hat, x_hat_all

        
