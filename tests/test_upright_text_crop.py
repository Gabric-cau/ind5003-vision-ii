import unittest
import numpy as np
from src.upright_text_crop import warp_text_upright


class UprightTextCropTests(unittest.TestCase):
    def test_tall_digit_does_not_rotate(self):
        image=np.zeros((100,100,3),np.uint8);image[10:30,20:50]=255
        crop=warp_text_upright(image,[[20,10],[50,10],[50,90],[20,90]])
        self.assertEqual(crop.shape[:2],(80,30));self.assertGreater(crop[:10].mean(),crop[-10:].mean())
    def test_wide_name_shape(self):
        self.assertEqual(warp_text_upright(np.zeros((100,100,3),np.uint8),[[10,10],[90,10],[90,30],[10,30]]).shape[:2],(20,80))
    def test_degenerate_refused(self):self.assertIsNone(warp_text_upright(np.zeros((10,10,3),np.uint8),[[0,0]]*4))
    def test_nan_refused(self):
        with self.assertRaises(ValueError):warp_text_upright(np.zeros((10,10,3),np.uint8),[[float('nan'),0]]*4)


if __name__=='__main__':unittest.main()
