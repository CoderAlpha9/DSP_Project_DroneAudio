import unittest
import numpy as np
from pipeline.detector import Features,extract,Detector,detect,from_features,State
class Checks(unittest.TestCase):
 def test_features_finite_silence(self):self.assertTrue(np.isfinite(extract(np.zeros(3200))).all())
 def test_input_rejected(self):
  for x in [np.zeros(159),np.full(160,np.nan),np.zeros((160,1))]:
   with self.assertRaises(ValueError):Features().step(x)
 def test_feature_prefix(self):
  x=np.random.default_rng(2).normal(0,.01,3200);np.testing.assert_array_equal(extract(x)[:10],extract(x[:1600]))
 def test_steady_sine_features_finite(self):self.assertTrue(np.isfinite(extract(.1*np.sin(2*np.pi*1000*np.arange(16000)/16000))).all())
 def test_separate_release_gap(self):
  s=State(on=.6,off=.3,onset=1,release=2,gap=8);s.step(.9);r=[s.step(0) for _ in range(4)];self.assertFalse(r[-1]['activity_active']);self.assertTrue(r[-1]['event_active'])
 def test_no_auto_reference_reset(self):
  s=State();self.assertNotIn('reset_reference',s.step(0))
 def test_selected_stream_cache(self):
  x=np.random.default_rng(4).normal(0,.01,3200);a=detect(x);b=from_features(extract(x),Detector().model);self.assertEqual(a,b)
 def test_selected_prefix(self):
  x=np.random.default_rng(9).normal(0,.01,3200);self.assertEqual(detect(x)[:10],detect(x[:1600]))
if __name__=='__main__':unittest.main()
