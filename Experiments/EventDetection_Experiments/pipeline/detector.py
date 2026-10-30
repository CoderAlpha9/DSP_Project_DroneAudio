"""Causal general activity + independent speech score. No inference clean/noise oracle."""
from pathlib import Path
import json
from dataclasses import dataclass,asdict
import numpy as np
from scipy import signal,special

NAMES=['excess_db','band_excess','flux','js_change','complex_error','phase_error','gd_change','flatness','periodicity','centroid']
SETS={'energy':[0,1], 'spectral':[0,1,2,3,7,8,9], 'phase':[0,1,2,3,4,5,7,8,9], 'group_delay':[0,1,2,3,6,7,8,9], 'full':list(range(10))}

class Features:
 def __init__(self):
  self.sos=signal.butter(2,[100,6500],btype='bandpass',fs=16000,output='sos');self.zi=np.zeros((2,2));self.last=np.zeros(240)
  self.win=np.hanning(400);self.floor=None;self.prev=None;self.older=None;self.gd=None;self.i=0
 def step(self,x):
  x=np.asarray(x,float)
  if x.shape!=(160,) or not np.isfinite(x).all():raise ValueError('160 finite samples at 16 kHz required')
  z,self.zi=signal.sosfilt(self.sos,x,zi=self.zi);a=np.r_[self.last,z];self.last=a[-240:]
  X=np.fft.rfft(a*self.win,512)[4:209];P=abs(X)**2+1e-18
  # Asymmetric minimum follower; .3% upward relaxation per 10 ms (~1.3 dB/s).
  if self.floor is None:self.floor=P.copy()
  ratio=P/(self.floor+1e-18)
  excess=float(10*np.log10(np.mean(np.maximum(ratio,1))))
  band_excess=float(np.mean(np.sort(10*np.log10(np.maximum(ratio,1)))[-40:]))
  q=P/P.sum();mag=abs(X)
  flux=js=cd=pe=gd=0.
  # Regularized signal group-delay proxy, NOT filter/propagation delay.
  Y=np.fft.rfft(np.arange(400)*a*self.win,512)[4:209]
  tau=np.clip((Y*X.conj()).real/(P+.01*P.max()),0,400)/400
  if self.prev is not None:
   old=self.prev;oldq=abs(old)**2+1e-18;oldq/=oldq.sum();mid=.5*(q+oldq)
   flux=float(np.maximum(np.sqrt(q)-np.sqrt(oldq),0).sum())
   js=float(.5*(np.sum(q*np.log((q+1e-18)/(mid+1e-18)))+np.sum(oldq*np.log((oldq+1e-18)/(mid+1e-18)))))
   gd=float(np.sum(np.minimum(q,oldq)*abs(tau-self.gd)))
   if self.older is not None:
    predicted=abs(old)*np.exp(1j*(2*np.angle(old)-np.angle(self.older)))
    cd=float(np.sum(abs(X-predicted))/(np.sum(mag)+np.sum(abs(old))+1e-18))
    deviation=np.angle(X*np.exp(-1j*(2*np.angle(old)-np.angle(self.older))))
    mask=(ratio>2)&(P>P.max()*1e-4)
    pe=float(np.sum(P[mask]*abs(deviation[mask]))/(np.sum(P[mask])+1e-18)/np.pi)
  flat=float(np.exp(np.mean(np.log(P)))/np.mean(P))
  ac=signal.correlate(a,a,mode='full',method='fft')[399:];period=float(np.clip(np.max(ac[40:201])/(ac[0]+1e-18),0,1))
  centroid=float(np.sum(np.arange(4,209)*q)/256)
  self.floor=np.minimum(P,.997*self.floor+.003*P)
  self.older=self.prev;self.prev=X;self.gd=tau;self.i+=1
  return np.array([excess,band_excess,flux,js,cd,pe,gd,flat,period,centroid])

def extract(x):
 f=Features();out=[]
 for i in range(0,len(x),160):
  a=x[i:i+160];out.append(f.step(np.pad(a,(0,160-len(a)))))
 return np.asarray(out)

class State:
 def __init__(self,on=.6,off=.35,onset=2,release=15,gap=40):
  self.on=on;self.off=off;self.onset=onset;self.release=release;self.gap=gap;self.active=False;self.event=False;self.c=0;self.quiet=0;self.eq=0;self.eid=0
 def step(self,p):
  if not self.active:
   self.c=self.c+1 if p>=self.on else 0
   if self.c>=self.onset:self.active=True;self.quiet=0;self.c=0
  else:
   self.quiet=0 if p>=self.off else self.quiet+1
   if self.quiet>=self.release:self.active=False
  opened=closed=False
  if self.active:
   self.eq=0
   if not self.event:self.event=True;self.eid+=1;opened=True
  elif self.event:
   self.eq+=1
   if self.eq>=self.gap:self.event=False;closed=True
  return dict(activity_active=int(self.active),event_active=int(self.event),event_opened=int(opened),event_closed=int(closed),event_id=self.eid if self.event else 0)

class Detector:
 def __init__(self,model=None):
  if model is None:model=json.loads((Path(__file__).resolve().parents[1]/'configs/selected.json').read_text())
  self.model=model;self.features=Features();self.activity=State(**model['state']);self.speech=State(on=.6,off=.35,onset=3,release=12,gap=30);self.index=0
 def step_features(self,f):
  m=self.model;z=np.clip((f-np.array(m['center']))/np.array(m['scale']),-12,12)
  pa=float(special.expit(np.dot(z,m['activity_coef'])+m['activity_intercept']))
  ps=float(special.expit(np.dot(z,m['speech_coef'])+m['speech_intercept']))
  a=self.activity.step(pa);s=self.speech.step(ps)
  r=dict(time_start_s=self.index*.01,available_time_s=(self.index+1)*.01,activity_score=pa,speech_score=ps,**a,
         speech_active=int(s['activity_active'] and a['event_active']),reset_reference=0,
         **dict(zip(NAMES,map(float,f))))
  self.index+=1;return r
 def process_hop(self,x):return self.step_features(self.features.step(x))

def detect(x,model=None):
 d=Detector(model);out=[]
 for i in range(0,len(x),160):out.append(d.process_hop(np.pad(x[i:i+160],(0,max(0,160-len(x[i:i+160]))))))
 return out

def from_features(f,model):
 d=Detector(model);return [d.step_features(a) for a in f]
