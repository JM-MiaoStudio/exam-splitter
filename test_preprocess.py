import unittest
import numpy as np
from PIL import Image
import preprocess


class PreprocessTests(unittest.TestCase):
    def test_shadow_correction_preserves_shape_and_ink(self):
        self.assertTrue(callable(getattr(preprocess,'flatten_background',None)))
        im=np.tile(np.linspace(100,240,500,dtype=np.uint8),(500,1))
        im[200:205,100:400]=20
        image=Image.fromarray(im).convert('RGB')
        result=np.asarray(preprocess.flatten_background(image))
        self.assertEqual(result.shape,(500,500,3))
        self.assertLess(result[50,:,0].std(),im[50].std()/2)
        self.assertLess(result[202,200,0],100)
        self.assertTrue(np.array_equal(np.asarray(image)[:,:,0],im))

    def test_small_images_remain_valid(self):
        out=preprocess.flatten_background(Image.new('RGB',(2,3),'white'))
        self.assertEqual(out.size,(2,3))

if __name__=='__main__':unittest.main()
