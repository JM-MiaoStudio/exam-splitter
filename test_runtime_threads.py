import os
import unittest
from unittest.mock import patch
from ocr_runtime import ocr_engine


class RuntimeThreadTests(unittest.TestCase):
    def tearDown(self):ocr_engine.cache_clear()

    def test_cache_directory_changes_engine_identity_and_is_not_ambient_for_onnx(self):
        ocr_engine.cache_clear()
        with patch('openvino_v4.create_engine') as factory:
            for folder in ['cache-a','cache-b']:
                with patch.dict(os.environ,{'EXAM_OCR_BACKEND':'openvino','EXAM_OV_CACHE_DIR':folder}):ocr_engine()
            self.assertEqual(factory.call_count,2)
            self.assertEqual(factory.call_args.kwargs['cache_dir'],'cache-b')
        with patch.dict(os.environ,{'EXAM_OCR_BACKEND':'onnxruntime','EXAM_OV_CACHE_DIR':'ignored'}),patch('rapidocr_onnxruntime.RapidOCR') as ort:
            ocr_engine();ort.assert_called_once()

    def test_backend_is_explicit_and_part_of_cache_identity(self):
        ocr_engine.cache_clear()
        with patch('rapidocr_onnxruntime.RapidOCR') as ort, patch('openvino_v4.create_engine') as ov:
            with patch.dict(os.environ,{'EXAM_OCR_BACKEND':'onnxruntime','EXAM_OCR_THREADS':'2'}):
                first=ocr_engine();self.assertIs(first,ocr_engine())
            with patch.dict(os.environ,{'EXAM_OCR_BACKEND':'openvino','EXAM_OCR_THREADS':'2'}):
                self.assertIs(ocr_engine(),ov.return_value)
                self.assertIsNot(ocr_engine(),first)
            ort.assert_called_once();ov.assert_called_once()

    def test_invalid_backend_never_silently_uses_default(self):
        with patch.dict(os.environ,{'EXAM_OCR_BACKEND':'auto'}):
            with self.assertRaisesRegex(ValueError,'INVALID_OCR_BACKEND'):ocr_engine()

    def test_thread_change_never_reuses_an_engine_with_previous_settings(self):
        ocr_engine.cache_clear()
        with patch('ocr_runtime.os.cpu_count',return_value=8), \
             patch('rapidocr_onnxruntime.RapidOCR') as factory:
            with patch.dict(os.environ,{'EXAM_OCR_THREADS':'4'}):
                first=ocr_engine();self.assertIs(first,ocr_engine())
                factory.assert_called_once_with(intra_op_num_threads=4,inter_op_num_threads=1)
            with patch.dict(os.environ,{'EXAM_OCR_THREADS':'2'}):ocr_engine()
            self.assertEqual(factory.call_count,2)
            factory.assert_called_with(intra_op_num_threads=2,inter_op_num_threads=1)

    def test_invalid_threads_fail_before_engine_creation(self):
        for value in ['0','17','-1','2.5','NaN','']:
            with self.subTest(value=value),patch.dict(os.environ,{'EXAM_OCR_THREADS':value}), \
                 patch('rapidocr_onnxruntime.RapidOCR') as factory:
                with self.assertRaisesRegex(ValueError,'INVALID_OCR_THREADS'):ocr_engine()
                factory.assert_not_called()

    def test_thread_request_is_capped_to_cpu_count_before_library_validation(self):
        # RapidOCR treats a value above cpu_count as unset (unbounded ORT default).
        ocr_engine.cache_clear()
        with patch.dict(os.environ,{'EXAM_OCR_THREADS':'16'}), \
             patch('ocr_runtime.os.cpu_count',return_value=2), \
             patch('rapidocr_onnxruntime.RapidOCR') as factory:
            ocr_engine()
            factory.assert_called_once_with(intra_op_num_threads=2,inter_op_num_threads=1)
