"""Continuous mixed-audio amplitude demonstration; this is NOT a denoiser."""
from pathlib import Path
import json
import numpy as np
from .detector import Detector
from .amplitude import Config,AmplitudePipeline
class MixedPipeline:
 def __init__(self,model=None,amplitude_config=None):
  self.detector=Detector(model)
  if amplitude_config is None:amplitude_config=Config(**json.loads((Path(__file__).resolve().parents[1]/'configs/amplitude.json').read_text()))
  self.amplitude=AmplitudePipeline(amplitude_config)
 def new_session(self):
  self.amplitude.reset_event()
 def process_block(self,x,*,pose=None,pose_valid=False,source_position=(0.,0.,0.)):
  x=np.asarray(x,float)
  if x.shape!=(320,) or not np.isfinite(x).all():raise ValueError('320 finite mono samples required')
  logs=[self.detector.process_hop(x[:160]),self.detector.process_hop(x[160:])]
  event=any(v['event_active'] for v in logs)
  y,g,a=self.amplitude.process_block(x,pose=pose,pose_valid=pose_valid,source_position=source_position,event_active=True,reference_update=event)
  return y,g,logs,a
