"""Independent mathematical reimplementations, with declared adaptations.

These are offline literature-method adaptations, not author-code executions
and not reproductions of the papers' published corpus results. See METHODS.md.
No modified analysis waveform is used as delivered audio.
"""
import numpy as np
from scipy import signal, ndimage, special


def regions(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def frames(x,n=320,hop=160):
    count=(len(x)+hop-1)//hop
    a=np.pad(x,(n-hop,max(0,count*hop-len(x))))
    return np.lib.stride_tricks.sliding_window_view(a,n)[::hop][:count]


def minimum_psd(p,size=150):
    sm=signal.lfilter([.2],[1,-.8],p,axis=0)
    return 1.5*ndimage.minimum_filter1d(sm,size=size,axis=0,mode='nearest')


def weighted_difference(e,noise):
    return np.sqrt(np.abs(np.diff(e,prepend=e[0]))*
        np.maximum(10*np.log10((e+1e-24)/(noise+1e-24)),0))


def rvad_fast_2020(x,sr=16000):
    """rVAD-fast core, Tan et al. (2020), eqs. 2-10 and segment rules.

    Adaptations: 20 ms trailing frames; FFT512; fixed-bias minimum-PSD
    spectral subtraction instead of the authors' MSNE implementation.
    """
    hp=signal.sosfilt(signal.butter(1,80,fs=sr,btype='highpass',output='sos'),x)
    a=frames(hp);mag=np.abs(np.fft.rfft(a*np.hamming(320),512))+1e-20
    flat=np.exp(np.mean(np.log(mag),axis=1))/np.mean(mag,axis=1)
    anchors=flat<=.5;e=np.sum(a*a,axis=1);noise=np.zeros(len(e));old=None
    for i in range(0,len(e),200):
        q=np.percentile(e[i:i+200],10);old=q if old is None else .9*old+.1*q
        noise[i:i+200]=old
    d=ndimage.uniform_filter1d(weighted_difference(e,noise),37,mode='nearest')
    rejected=np.zeros(len(e),bool)
    for i in range(0,len(e),200):
        h=d[i:i+200]>.25*np.max(e[i:i+200])
        for aa,bb in regions(h):
            aa+=i;bb+=i
            if np.sum(anchors[aa:bb])<=2:rejected[aa:bb]=True
    clean_a=a.copy();clean_a[rejected]=0
    spec=np.fft.rfft(clean_a*np.hamming(320),512);p=abs(spec)**2
    noise_p=minimum_psd(p)
    gain=np.sqrt(np.maximum(1-noise_p/(p+1e-24),.01))
    denoised_e=np.sum(abs(spec*gain)**2,axis=1)
    extended=ndimage.maximum_filter1d(anchors.astype(int),121,mode='constant')>0
    pred=np.zeros(len(e),bool)
    for aa,bb in regions(extended):
        en=denoised_e[aa:bb];ne=np.percentile(en,10)
        dd=ndimage.uniform_filter1d(weighted_difference(en,ne),37,mode='nearest')
        pitch=anchors[aa:bb]
        threshold=.4*np.mean(dd[pitch]) if np.any(pitch) else np.inf
        pred[aa:bb]=dd>threshold
    # Asymmetric published vicinity rules around the flatness anchors.
    permitted=np.zeros(len(e),bool);forced=np.zeros(len(e),bool)
    for aa,bb in regions(anchors):
        permitted[max(0,aa-33):min(len(e),bb+47)]=True
        forced[max(0,aa-5):min(len(e),bb+12)]=True
    pred=(pred & permitted)|forced
    for aa,bb in regions(pred):
        if np.mean(e[aa:bb])<.05*np.mean(e):pred[aa:bb]=False
    return pred


def fit_gmm(z):
    z=np.asarray(z,float)
    if len(z)<4:z=np.r_[z,np.repeat(np.mean(z) if len(z) else 0.,4)]
    means=np.percentile(z,[30,70]);variances=np.full(2,max(np.var(z),.04));weights=np.full(2,.5)
    for _ in range(40):
        lp=np.log(weights+1e-12)-.5*(np.log(2*np.pi*variances)+(z[:,None]-means)**2/variances)
        resp=np.exp(lp-special.logsumexp(lp,axis=1)[:,None]);nk=resp.sum(axis=0)+1e-10
        means=(resp*z[:,None]).sum(axis=0)/nk
        variances=np.maximum((resp*(z[:,None]-means)**2).sum(axis=0)/nk,.04)
        weights=nk/nk.sum()
    return means,variances,weights


def gmm_logpdf(z,model):
    m,v,w=model
    return special.logsumexp(np.log(w)-.5*(np.log(2*np.pi*v)+(z[:,None]-m)**2/v),axis=1)


def statistical_sad_2020(x,sr=16000):
    """Heitkaemper et al. (2020), statistical branch, fig.1 and eq.1.

    Adaptations declared: 3 Wiener passes, fixed-bias centered minima,
    100 Hz Butterworth HP, LPC predictor coefficient clipped .95, two
    Gaussians/class, quantile fallback if class-seeding degenerates.
    Offline recording statistics and Viterbi traceback are intentional.
    """
    hp=signal.sosfilt(signal.butter(2,100,fs=sr,btype='highpass',output='sos'),x)
    a=frames(hp);spec=np.fft.rfft(a*np.hanning(320),512)
    for _ in range(3):
        p=abs(spec)**2;npow=minimum_psd(p)
        gain=np.maximum(1-25*npow/(p+1e-24),.05);spec*=gain
    # First-order LPC synthesis emphasizes predictable portions in the analysis branch.
    z=np.fft.irfft(spec,512,axis=1)[:,:320]
    r0=np.sum(z*z,axis=1)+1e-20;r1=np.sum(z[:,1:]*z[:,:-1],axis=1)
    coeff=np.clip(r1/r0,-.95,.95)
    enhanced=z.copy()
    for j in range(1,320):enhanced[:,j]+=coeff*enhanced[:,j-1]
    p=abs(np.fft.rfft(enhanced*np.hanning(320),512))**2
    f=np.fft.rfftfreq(512,1/sr)
    bands=np.column_stack([p[:,(f>=k*1000)&(f<(k+1)*1000)].sum(axis=1) for k in range(8)])
    csbe=ndimage.uniform_filter1d((bands/np.arange(1,9)).sum(axis=1),48,mode='nearest')
    floor=ndimage.minimum_filter1d(csbe,150,mode='nearest');average=np.mean(floor)
    values=np.log(np.maximum(csbe,1e-24))
    low=values[csbe<1.5*(floor+average)];high=values[csbe>3*(floor+average)]
    if len(low)<10:low=values[values<=np.percentile(values,30)]
    if len(high)<10:high=values[values>=np.percentile(values,70)]
    emissions=np.column_stack([gmm_logpdf(values,fit_gmm(low)),gmm_logpdf(values,fit_gmm(high))])
    # Five consecutive noise and speech substates, cyclic transitions .9/.1.
    n=len(values);scores=np.full(10,-np.inf);scores[0]=scores[5]=np.log(.5)
    back=np.zeros((n,10),np.int8)
    for t in range(n):
        new=np.empty(10)
        for j in range(10):
            previous=(j-1)%10;a=scores[j]+np.log(.9);b=scores[previous]+np.log(.1)
            use_previous=b>a;back[t,j]=previous if use_previous else j
            new[j]=max(a,b)+emissions[t,int(j>=5)]
        scores=new
    state=int(np.argmax(scores));path=np.zeros(n,int)
    for t in range(n-1,-1,-1):path[t]=state;state=int(back[t,state])
    return path>=5


def fixed_rms(x,threshold=1e-5):
    # The old API fallback, including its one-second hangover.
    a=frames(x);raw=np.sqrt(np.mean(a*a,axis=1))>threshold
    out=np.zeros(len(raw),bool);quiet=100
    for i,v in enumerate(raw):quiet=0 if v else quiet+1;out[i]=v or quiet<100
    return out


def adaptive_energy(features):
    history=[];pred=[];active=False;quiet=0
    for f in features:
        history.append(f['energy_dbfs']);history=history[-200:]
        floor=np.percentile(history,15)
        raw=f['energy_dbfs']>max(-90,floor+8)
        if raw:active=True;quiet=0
        elif active:
            quiet+=1
            if quiet>=15:active=False
        pred.append(active)
    return np.asarray(pred)
