import base64
import hashlib
import io
import unittest
from unittest.mock import patch
from PIL import Image

from worker import validate_image, process_page
from layout import LayoutError


def request(width=1000, height=1200):
    output=io.BytesIO()
    Image.new('RGB',(width,height),'white').save(output,format='JPEG')
    data=output.getvalue()
    return {'version':1,'pageId':'test','imageHash':hashlib.sha256(data).hexdigest(),
            'width':width,'height':height,'images':[{'mime':'image/jpeg','base64':base64.b64encode(data).decode()}]}


class WorkerTests(unittest.TestCase):
    def test_jpeg_metadata_must_match_uploaded_pixels(self):
        page=request()
        self.assertEqual(validate_image(page).size,(1000,1200))
        page['width']=900
        with self.assertRaisesRegex(LayoutError,'IMAGE_DIMENSIONS_MISMATCH'):
            validate_image(page)

    def test_hash_checked_before_ocr(self):
        page=request();page['imageHash']='0'*64
        with self.assertRaisesRegex(LayoutError,'IMAGE_HASH_MISMATCH'):
            validate_image(page)

    def test_corrupted_jpeg_with_plausible_magic_rejected(self):
        page=request();data=b'\xff\xd8broken\xff\xd9'
        page['images'][0]['base64']=base64.b64encode(data).decode()
        page['imageHash']=hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(LayoutError,'INVALID_JPEG'):
            validate_image(page)

    def test_dimensions_bounded_before_pixel_decode(self):
        page=request(2049,100)
        with self.assertRaisesRegex(LayoutError,'INVALID_IMAGE_SIZE'):
            validate_image(page)

    def test_rotated_exif_cannot_silently_change_coordinate_space(self):
        page=request();image=Image.new('RGB',(1000,1200),'white')
        exif=Image.Exif();exif[274]=6;out=io.BytesIO();image.save(out,format='JPEG',exif=exif)
        data=out.getvalue();page['imageHash']=hashlib.sha256(data).hexdigest();page['images'][0]['base64']=base64.b64encode(data).decode()
        with self.assertRaisesRegex(LayoutError,'IMAGE_ORIENTATION_NOT_NORMALIZED'):
            validate_image(page)

    def test_full_pixel_page_no_image_bytes_or_student_text(self):
        page=request()
        rows=[{'quad':[[40,100],[600,100],[600,130],[40,130]],'text':'12.求质量','score':.99}]
        with patch('worker.recognize',return_value=rows):
            result=process_page(page)
        self.assertEqual(result['coordinateSpace'],'uploaded-image')
        self.assertEqual(result['imageHash'],page['imageHash'])
        self.assertNotIn('images',result)
        self.assertEqual(result['blocks'][0]['bbox']['width'],1000)

    def test_no_reliable_anchor_enters_existing_manual_box_flow(self):
        with patch('worker.recognize',return_value=[]):
            page=process_page(request())
        self.assertEqual(page['blocks'],[])
        self.assertEqual(page['coordinateSpace'],'uploaded-image')

    def test_ambiguous_layout_does_not_return_confident_wrong_boxes(self):
        with patch('worker.recognize',return_value=[]), patch('worker.split_questions',side_effect=LayoutError('AMBIGUOUS_COLUMNS')):
            page=process_page(request())
        self.assertEqual(page['blocks'],[])

    def test_layout_assistance_is_opt_in_and_receives_original_image(self):
        rows=[{'quad':[[40,100],[600,100],[600,130],[40,130]],'text':'12.求质量','score':.99}]
        refined=[{'id':'b1','ordinal':1,'printedNumber':'12','bbox':{'x':40,'y':98,'width':600,'height':100},
                  'regions':[{'x':40,'y':98,'width':600,'height':100},{'x':100,'y':700,'width':200,'height':100}],
                  'questionText':'','options':[],'studentAnswer':'','subject':'','questionType':'','unreadableParts':[],'hasFigure':True}]
        with patch('worker.recognize',return_value=rows),patch('worker.detect_layout',return_value=[]) as detect,patch('worker.refine_with_layout',return_value=refined) as refine:
            page=process_page(request())
            detect.assert_not_called();refine.assert_not_called()
            page=process_page(request(),layout_model_dir='E:/local-model')
            self.assertEqual(detect.call_args.args[0].size,(1000,1200))
            self.assertEqual(page['coordinateSpace'],'uploaded-image')
            self.assertEqual(page['blocks'][0]['bbox'],refined[0]['bbox'])
            self.assertEqual(page['blocks'][0]['regions'],refined[0]['regions'])
            self.assertIn('求质量',page['blocks'][0]['questionText'])
            self.assertTrue(page['blocks'][0]['unreadableParts'])

    def test_prepared_ocr_rows_are_mapped_to_upload_before_grouping(self):
        import numpy as np
        inv=np.eye(3);inv[0,2]=50;inv[1,2]=80
        rows=[{'quad':[[40,100],[600,100],[600,130],[40,130]],'text':'12.求质量','score':.99}]
        with patch('document_preprocess.prepare_document',return_value=(Image.new('RGB',(700,900),'white'),inv,{'perspective':True,'shadow':False})),patch('worker.recognize',return_value=rows):
            result=process_page(request(),prepare='perspective')
        self.assertEqual(result['blocks'][0]['bbox']['y'],178)
        self.assertEqual(result['width'],1000)
        self.assertTrue(result['blocks'][0]['unreadableParts'])

    def test_layout_model_failure_is_not_silently_reported_as_success(self):
        with patch('worker.recognize',return_value=[]),patch('worker.detect_layout',side_effect=RuntimeError('model unavailable')):
            with self.assertRaises(RuntimeError):process_page(request(),layout_model_dir='E:/missing-model')

    def test_v3_does_not_reocr_already_grouped_page(self):
        rows=[{'quad':[[40,100],[600,100],[600,130],[40,130]],'text':'12.求质量','score':.99}]
        with patch('worker.recognize',return_value=rows),patch('rapid_layout_adapter.detect_rapid_layout',return_value=[]),patch('rapid_layout_adapter.recover_region_rows') as recover:
            page=process_page(request(),rapid_layout_model='candidate.onnx')
            self.assertEqual(len(page['blocks']),1)
            recover.assert_not_called()

    def test_v3_recovers_observed_anchors_only_when_grouping_abstains(self):
        rows=[{'quad':[[40,y],[600,y],[600,y+30],[40,y+30]],'text':f'{i}. 求质量','score':.99,'source':'layout'} for i,y in [(1,100),(2,400)]]
        with patch('worker.recognize',return_value=[]),patch('rapid_layout_adapter.detect_rapid_layout',return_value=[]),patch('rapid_layout_adapter.recover_region_rows',return_value=rows):
            page=process_page(request(),rapid_layout_model='candidate.onnx')
            self.assertEqual([b['printedNumber'] for b in page['blocks']],['1','2'])
            self.assertTrue(all(b['unreadableParts'] for b in page['blocks']))

    def test_dense_english_abstention_recovers_reviewable_complete_envelopes(self):
        rows=[{'quad':[[40,100+i*35],[650,100+i*35],[650,150+i*35],[40,150+i*35]],
               'text':f'{i+1}. You have visited this place, have you?', 'score':.99} for i in range(6)]
        with patch('worker.recognize',return_value=rows),patch('dense_english._recognize_strip',return_value=[]):
            page=process_page(request())
        self.assertEqual([b['printedNumber'] for b in page['blocks']],['1','2','3','4','5','6'])
        self.assertTrue(all(b['regions']==[b['bbox']] and b['unreadableParts'] for b in page['blocks']))
        self.assertGreaterEqual(page['blocks'][0]['bbox']['y']+page['blocks'][0]['bbox']['height'],150)

    def test_layout_fallback_recovers_cropped_candidates_without_fabricating_numbers(self):
        def line(text,y):return {'text':text,'score':.99,'quad':[[1,y],[800,y],[800,y+30],[1,y+30]]}
        rows=[line('已知两个向量求投影向量',100),line('4.已知两个三角函数求值',300),line('如图所示求图像面积',500)]
        boxes=[{'label':'text','score':.9,'coordinate':[0,y,900,y+50]} for y in [95,295,495]]
        boxes.append({'label':'formula','score':.9,'coordinate':[350,520,500,550]})
        with patch('worker.recognize',return_value=rows),patch('worker.detect_layout',return_value=boxes):
            baseline=process_page(request())
            page=process_page(request(),layout_model_dir='E:/local-model')
        self.assertEqual(len(baseline['blocks']),1)
        self.assertEqual(len(page['blocks']),3)
        self.assertTrue(all(b['printedNumber'] is None and b['unreadableParts'] for b in page['blocks']))
        self.assertEqual(page['coordinateSpace'],'uploaded-image')


if __name__=='__main__':unittest.main()
