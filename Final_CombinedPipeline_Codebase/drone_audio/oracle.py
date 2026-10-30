"""Evaluation-only known-noise Wiener comparator adapted from filter design.

Never imported by inference. This uses a supplied isolated noise reference;
replacing it with an estimate would be a different experiment.
"""
import numpy as np
from scipy import signal


def known_noise_wiener(microphone, isolated_noise, alpha=.95, gain_floor=.02):
    y=np.asarray(microphone,float);noise=np.asarray(isolated_noise,float)
    if y.ndim!=1 or y.shape!=noise.shape or not np.isfinite(y).all() or not np.isfinite(noise).all():raise ValueError("Aligned finite mono mixture/noise required")
    if not 0<=alpha<1 or not 0<=gain_floor<=1:raise ValueError("Invalid pole or floor")
    if not len(y):return y.copy()
    n=len(y);y=np.pad(y,(0,max(0,512-n)));noise=np.pad(noise,(0,max(0,512-n)))
    kw=dict(fs=16000,window='hann',nperseg=512,noverlap=384,nfft=512,boundary='zeros',padded=True)
    _,_,Y=signal.stft(y,**kw);_,_,V=signal.stft(noise,**kw)
    N=abs(V)**2
    for t in range(1,N.shape[1]):N[:,t]=alpha*N[:,t-1]+(1-alpha)*N[:,t]
    S=np.maximum(abs(Y)**2-N,0);G=np.clip(S/(S+N+1e-12),gain_floor,1)
    _,out=signal.istft(Y*G,fs=16000,window='hann',nperseg=512,noverlap=384,nfft=512,boundary=True)
    return out[:n]
