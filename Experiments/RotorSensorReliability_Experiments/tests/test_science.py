"""Scientific invariants and independent mathematical oracles."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'original/code'));import experiment as o
sys.path.insert(0,str(ROOT/'fresh'));import fresh_reliability_study_v2 as f
class ScientificTests(unittest.TestCase):
 def test_stft_identity_both_studies(self):
  x=np.random.default_rng(4).normal(size=12347)
  for m in [o,f]:np.testing.assert_allclose(m.istft(m.stft(x),len(x)),x,atol=1e-12)
 def test_per_motor_constant_scale(self):
  s,v,y,r,_=f.make_case(16)
  for m,rf in [(f,f.rpm_frames(r,f.stft(y).shape[1])),(o,f.rpm_frames(r,o.stft(y).shape[1]))]:
   n0=np.abs(m.stft(y)[:,5:40]).mean(1)**2
   np.testing.assert_allclose(m.transport(n0,rf),m.transport(n0,rf*np.array([.7,1.2,1.5,.9])),rtol=1e-11,atol=1e-13)
 def test_conditional_blend_risk_oracle(self):
  rng=np.random.default_rng(39);a=rng.normal(size=2000);b=.3*a+rng.normal(size=2000)*.5
  ra=np.mean(a*a);rr=np.mean(b*b);c=np.mean(a*b);weight=np.clip((ra-c)/(ra+rr-2*c),0,1)
  risk=[np.mean(((1-l)*a+l*b)**2) for l in np.linspace(0,1,1001)]
  self.assertLessEqual(np.mean(((1-weight)*a+weight*b)**2),min(risk)+1e-12)
 def test_wiener_risk_identity(self):
  rng=np.random.default_rng(11);s=10**rng.uniform(-3,3,10000);n=10**rng.uniform(-3,3,10000);g=rng.random(10000);opt=s/(s+n)
  np.testing.assert_allclose((1-g)**2*s+g*g*n-s*n/(s+n),(s+n)*(g-opt)**2,atol=5e-13)
 def test_generator_contract_and_seed(self):
  s,v,y,r,db=f.make_case(17);s2,v2,y2,r2,db2=f.make_case(17)
  np.testing.assert_array_equal(y,y2);np.testing.assert_array_equal(r,r2)
  np.testing.assert_allclose(y,s+v,atol=1e-15);self.assertEqual(len(y),96000)
  self.assertEqual(np.count_nonzero(s[:16000]),0);self.assertTrue(np.all((r>=2200)&(r<=6500)))
  self.assertAlmostEqual(10*np.log10(np.mean(s[16000:]**2)/np.mean(v[16000:]**2)),db,places=7)
 def test_conditional_component_reconstruction(self):
  s,v,y,r,_=f.make_case(16);Y,S,V,P,NA,NR,NRc,c=f.state(s,v,y,r);g=f.gain(P,.25*NA+.75*NRc)
  np.testing.assert_allclose(f.istft(g*Y,len(y)),f.istft(g*S,len(y))+f.istft(g*V,len(y)),atol=1e-12)
 def test_operational_gain_uses_no_clean_stem(self):
  s,v,y,r,_=f.make_case(16);a=f.state(s,v,y,r);b=f.state(np.zeros_like(s),np.zeros_like(v),y,r)
  for i in [0,3,4,5,6,7]:np.testing.assert_array_equal(a[i],b[i])
 def test_drift_is_not_invariant(self):
  s,v,y,r,_=f.make_case(16);a=f.state(s,v,y,r)[5];b=f.state(s,v,y,f.corrupt(r,'drift_m20'))[5]
  self.assertGreater(np.max(np.abs(a-b)),1e-6)
 def test_original_all_frozen_algorithms_smoke(self):
  import json
  s,v,y,r,_=f.make_case(16);rpm=f.rpm_frames(r,o.stft(y).shape[1])
  for cfg in json.loads((ROOT/'configs/frozen_configs.json').read_text()):
   out,seconds=o.enhance(y,rpm,cfg)
   self.assertEqual(out.shape,y.shape);self.assertTrue(np.isfinite(out).all());self.assertGreaterEqual(seconds,0)
if __name__=='__main__':unittest.main()
