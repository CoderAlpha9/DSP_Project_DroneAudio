import sys,unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import Config,AmplitudePipeline,process_audio


class Tests(unittest.TestCase):
    def test_identity(self):
        x=np.random.default_rng(3).normal(0,.01,997)
        y,g,_=process_audio(x,Config(method='identity'),event_active=True)
        np.testing.assert_allclose(x,y,atol=1e-15)

    def test_no_geometry_defaults_to_unity(self):
        x=np.random.default_rng(3).normal(0,.01,3200)
        y,g,_=process_audio(x,Config(fallback='hold',geometry_weight=1),event_active=True)
        np.testing.assert_allclose(x,y)

    def test_clean_distance_doubling(self):
        c=Config(method='geometry',smooth_s=.01,reference_distance_m=2,max_slew_db_s=1000)
        x=np.ones(320*30)*.01;tele=[dict(pose=[4,0,0],pose_valid=True)]*30
        y,g,_=process_audio(x,c,telemetry=tele,event_active=True)
        np.testing.assert_allclose(y,.02,atol=1e-12)

    def test_constant_range_preserves_prosody(self):
        t=np.arange(16000)/16000;x=(.03+.02*np.sin(2*np.pi*3*t))*np.sin(2*np.pi*401*t)
        tele=[dict(pose=[5,0,3],pose_valid=True)]*50
        y,g,_=process_audio(x,Config(method='geometry'),telemetry=tele,event_active=True)
        np.testing.assert_allclose(x,y,atol=1e-12)

    def test_causality(self):
        x=np.random.default_rng(7).normal(0,.02,320*50);z=x.copy();z[320*30:]*=2
        a=process_audio(x,Config(method='energy'),event_active=True)[0]
        b=process_audio(z,Config(method='energy'),event_active=True)[0]
        np.testing.assert_array_equal(a[:320*30],b[:320*30])

    def test_reset_and_peak_guard(self):
        e=AmplitudePipeline(Config(method='geometry',reference_distance_m=2))
        y,g,_=e.process_block(np.ones(320)*.5,pose=[10,0,0],pose_valid=True,event_active=True)
        self.assertLessEqual(max(abs(y)),.95000001)
        e.process_block(np.zeros(320),event_active=False)
        self.assertIsNone(e.closest_range)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):process_audio(np.full(320,np.nan))


if __name__=='__main__':unittest.main()
