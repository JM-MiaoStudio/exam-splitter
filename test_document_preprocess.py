import unittest
import numpy as np
import cv2
from PIL import Image
import document_preprocess as dp

class DocumentPreprocessTests(unittest.TestCase):
    def page(self):
        im=np.zeros((800,700,3),np.uint8)+35
        cv2.fillConvexPoly(im,np.array([[100,70],[610,110],[580,730],[65,690]]), (230,230,230))
        cv2.line(im,(145,180),(520,210),(20,20,20),3)
        return Image.fromarray(im)

    def test_detect_page_and_round_trip_coordinate(self):
        im=self.page();prepared,inv,meta=dp.prepare_document(im,perspective=True,shadow=False)
        self.assertTrue(meta['perspective'])
        p=np.array([[[100.,100.],[200.,100.],[200.,130.],[100.,130.]]],np.float32)
        original=cv2.perspectiveTransform(p,inv)
        roundtrip=cv2.perspectiveTransform(original,np.linalg.inv(inv))
        self.assertLess(float(np.max(abs(roundtrip-p))),.01)
        rows=dp.restore_rows([{'text':'文字','score':.99,'quad':p[0].tolist()}],inv,700,800)
        self.assertTrue(np.allclose(rows[0]['quad'],original[0]))

    def test_full_bleed_and_internal_table_not_cropped(self):
        im=np.zeros((800,700,3),np.uint8)+240
        cv2.rectangle(im,(150,150),(550,450),(20,20,20),3)
        out,inv,meta=dp.prepare_document(Image.fromarray(im),perspective=True,shadow=False)
        self.assertFalse(meta['perspective']);self.assertEqual(out.size,(700,800))
        self.assertTrue(np.array_equal(inv,np.eye(3)))

    def test_shadow_conditional_preserves_ink_and_original(self):
        im=np.tile(np.linspace(140,240,700,dtype=np.uint8),(800,1))
        im[200:203,200:500]=70;im[250:252,210:220]=90
        original=im.copy()
        out,_,meta=dp.prepare_document(Image.fromarray(im).convert('RGB'),perspective=False,shadow=True)
        self.assertTrue(meta['shadow'])
        gray=np.asarray(out)[:,:,0]
        self.assertLess(gray[201,300],gray[190,300]-30)
        self.assertLess(gray[251,215],gray[240,215]-25)
        self.assertTrue(np.array_equal(im,original))
        _,_,meta=dp.prepare_document(Image.new('RGB',(700,800),'white'),perspective=False,shadow=True)
        self.assertFalse(meta['shadow'])

    def test_invalid_inverse_and_out_of_frame_quads_rejected(self):
        row={'text':'x','score':1,'quad':[[0,0],[10,0],[10,10],[0,10]]}
        with self.assertRaises(ValueError):dp.restore_rows([row],np.zeros((3,3)),700,800)
        inv=np.eye(3);inv[0,2]=1000
        with self.assertRaises(ValueError):dp.restore_rows([row],inv,700,800)

if __name__=='__main__':unittest.main()
