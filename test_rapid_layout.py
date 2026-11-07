import unittest
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image
from rapid_layout_adapter import normalize_boxes, detect_rapid_layout, recover_region_rows

class RapidLayoutTests(unittest.TestCase):
    def test_model_categories_and_pixels_are_normalized_without_clamping_bad_boxes(self):
        r=SimpleNamespace(boxes=[[10,20,90,70],[0,1,20,30],[-1,0,10,10]],
                          class_names=['inline_formula','image','table'],scores=[.9,.8,.9])
        boxes=normalize_boxes(r,100,100)
        self.assertEqual([b['label'] for b in boxes],['formula','image'])
        self.assertEqual(boxes[0]['coordinate'],[10,20,90,70])
    def test_missing_or_relative_model_never_downloads(self):
        for p in ['relative.onnx','E:/definitely-missing/model.onnx']:
            with self.assertRaises(ValueError):detect_rapid_layout(Image.new('RGB',(100,100)),p)
    def test_region_ocr_maps_pixels_and_marks_internal_boundary_text(self):
        boxes=[{'label':'text','score':.9,'coordinate':[20,30,180,70]}]
        def ocr(im):return [{'quad':[[0,5],[80,5],[80,20],[0,20]],'text':'1. Example sentence','score':.99}]
        out=recover_region_rows(Image.new('RGB',(200,100)),boxes,recognize_crop=ocr)
        self.assertEqual(out[0]['quad'][0],[16,31])
        self.assertEqual(out[0]['source'],'strip') # clipped crop cannot overwrite transcription
        self.assertTrue(out[0]['clipped'])
    def test_region_ocr_ignores_figures_and_caps_work(self):
        boxes=[{'label':'image','score':.9,'coordinate':[0,0,90,50]}]
        self.assertEqual(recover_region_rows(Image.new('RGB',(100,100)),boxes,recognize_crop=lambda _:self.fail()),[])

    def test_region_recovery_caps_count_and_pixel_budget(self):
        calls=[]
        ocr=lambda im:calls.append(im.size) or []
        boxes=[{'label':'text','score':.9,'coordinate':[i,10,i+10,20]} for i in range(60)]
        recover_region_rows(Image.new('RGB',(1000,1000)),boxes,recognize_crop=ocr)
        self.assertEqual(len(calls),50)
        calls.clear()
        boxes=[{'label':'text','score':.9,'coordinate':[i,0,100,100]} for i in range(10)]
        recover_region_rows(Image.new('RGB',(100,100)),boxes,recognize_crop=ocr)
        self.assertLessEqual(sum(w*h for w,h in calls),20000)
