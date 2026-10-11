import unittest
import numpy as np
from src.person_appearance import torso_histogram,similarity


class PersonAppearanceTests(unittest.TestCase):
    def test_same_jersey_high_dark_referee_low(self):
        green=np.full((80,60,3),(20,230,30),np.uint8)
        black=np.full((80,60,3),(20,20,20),np.uint8)
        obs={"bbox":[10,10,50,70]}
        a=torso_histogram(green,obs);b=torso_histogram(black,obs)
        # OpenCV histogram/smoothing uses float32: tolerate its rounding unit.
        self.assertAlmostEqual(similarity(a,a),1,delta=1e-6)
        self.assertLess(similarity(a,b),.5)

    def test_missing_signature_not_evidence(self):
        self.assertIsNone(similarity(None,np.ones(3)))
