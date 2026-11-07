import io
import unittest
from unittest.mock import patch
import worker


class PerformanceTests(unittest.TestCase):
    def test_profile_is_opt_in_and_does_not_enter_page_output(self):
        from PIL import Image
        page=dict(version=1,pageId='private-id',imageHash='a'*64,width=100,height=100)
        with patch('worker.validate_image',return_value=Image.new('RGB',(100,100))), \
             patch('worker.recognize',return_value=[]):
            normal=worker.process_page(page)
            metrics={}
            with patch('sys.stdout',io.StringIO()) as stdout:
                measured=worker.process_page(page,profile=metrics)
            self.assertEqual(normal,measured)
            self.assertEqual(stdout.getvalue(),'')
            self.assertIn('total',metrics)
            self.assertIn('image.decode',metrics)
            self.assertNotIn('private-id',str(metrics))
            self.assertNotIn('profile',measured)

    def test_failed_request_records_failure_but_preserves_original_exception(self):
        metrics={}
        with patch('worker.validate_image',side_effect=ValueError('private text')):
            with self.assertRaisesRegex(ValueError,'private text'):
                worker.process_page({},profile=metrics)
        self.assertEqual(metrics['total']['failures'],1)
        self.assertNotIn('private',str(metrics))

    def test_collector_is_request_local_and_resets_after_failure(self):
        from performance import collect,stage
        one={};two={}
        with collect(one):
            with stage('grouping'):pass
            with collect(two):
                with stage('layout'):pass
            with stage('grouping'):pass
        with stage('layout'):pass
        self.assertEqual(one['grouping']['calls'],2)
        self.assertNotIn('layout',one)
        self.assertEqual(two['layout']['calls'],1)

    def test_ocr_stages_count_page_and_strip_calls_without_recording_text(self):
        from performance import collect
        from PIL import Image
        metrics={}
        with collect(metrics),patch('worker.ocr_engine',return_value=lambda image:([],None)):
            self.assertEqual(worker.recognize(Image.new('RGB',(100,100))),[])
        for name in ['ocr.page','ocr.left','ocr.right']:
            self.assertEqual(metrics[name]['calls'],1)
