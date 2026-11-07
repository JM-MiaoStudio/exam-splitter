import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np


class OpenVinoV4Tests(unittest.TestCase):
    def test_optional_cache_only_configures_model_directory(self):
        from openvino_v4 import model_sessions
        core=Mock();factory=Mock(return_value=core)
        with tempfile.TemporaryDirectory() as folder:
            sessions=model_sessions(factory,{'Det':'det.onnx','Rec':'rec.onnx'},2,folder)
            directory=Path(core.set_property.call_args.args[0]['CACHE_DIR'])
            self.assertTrue(directory.is_dir())
            self.assertEqual(directory.parent,Path(folder))
            self.assertEqual(set(sessions),{'Det','Rec'})
            self.assertEqual(core.compile_model.call_count,2)

    def test_cache_write_failure_uses_fresh_uncached_core(self):
        from openvino_v4 import model_sessions
        broken=Mock();fresh=Mock();factory=Mock(side_effect=[broken,fresh])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'file';path.write_text('do not overwrite')
            model_sessions(factory,{'Det':'det.onnx'},2,str(path))
            self.assertEqual(path.read_text(),'do not overwrite')
        broken.compile_model.assert_not_called();fresh.compile_model.assert_called_once()
        fresh.set_property.assert_not_called()

    def test_corrupt_cache_retries_same_models_once_without_cache(self):
        from openvino_v4 import model_sessions
        cached=Mock();cached.compile_model.side_effect=RuntimeError('bad blob')
        fresh=Mock();factory=Mock(side_effect=[cached,fresh])
        with tempfile.TemporaryDirectory() as folder:
            model_sessions(factory,{'Det':'same.onnx'},2,folder)
        self.assertEqual(factory.call_count,2)
        self.assertEqual(cached.compile_model.call_args,fresh.compile_model.call_args)
        fresh.set_property.assert_not_called()

    def test_real_compile_error_is_not_hidden_by_cache_fallback(self):
        from openvino_v4 import model_sessions
        core=Mock();core.compile_model.side_effect=RuntimeError('not compilable')
        factory=Mock(return_value=core)
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(RuntimeError,'not compilable'):
            model_sessions(factory,{'Det':'same.onnx'},2,folder)
        self.assertEqual(factory.call_count,2)

    def test_relative_cache_path_is_rejected_without_creating_core(self):
        from openvino_v4 import model_sessions
        factory=Mock()
        for value in ['', 'relative-cache']:
            with self.assertRaisesRegex(ValueError,'INVALID_OCR_CACHE_DIR'):
                model_sessions(factory,{'Det':'det.onnx'},2,value)
        factory.assert_not_called()

    def test_fp32_serial_session_and_detached_output(self):
        from openvino_v4 import InferenceSession
        value=np.ones((1,2,3),dtype=np.float32)
        compiled=Mock(return_value={'out':value});compiled.outputs=['out']
        core=Mock();core.compile_model.return_value=compiled
        session=InferenceSession(core,'local.onnx',2)
        config=core.compile_model.call_args.args[2]
        self.assertEqual(config['INFERENCE_PRECISION_HINT'],'f32')
        self.assertEqual(config['INFERENCE_NUM_THREADS'],2)
        self.assertEqual(config['NUM_STREAMS'],1)
        actual=session(value)
        self.assertEqual(len(actual),1)
        np.testing.assert_array_equal(actual[0],value)
        actual[0][0,0,0]=99
        self.assertEqual(value[0,0,0],1)

    def test_invalid_threads_rejected_before_compile(self):
        from openvino_v4 import InferenceSession
        core=Mock()
        for threads in [0,17,-1,True,1.5]:
            with self.subTest(threads=threads),self.assertRaisesRegex(ValueError,'INVALID_OCR_THREADS'):
                InferenceSession(core,'local.onnx',threads)
        core.compile_model.assert_not_called()

    def test_replacement_is_instance_local_and_preserves_decoders(self):
        from openvino_v4 import replace_sessions
        decoder=object();preprocess=object()
        engine=SimpleNamespace(text_det=SimpleNamespace(infer='old-det',preprocess_op=preprocess),
            text_cls=SimpleNamespace(infer='old-cls'),text_rec=SimpleNamespace(session='old-rec',postprocess_op=decoder))
        other=SimpleNamespace(infer='untouched')
        self.assertIs(replace_sessions(engine,{'Det':'new-det','Cls':'new-cls','Rec':'new-rec'}),engine)
        self.assertEqual(engine.text_det.infer,'new-det');self.assertEqual(engine.text_rec.session,'new-rec')
        self.assertIs(engine.text_det.preprocess_op,preprocess)
        self.assertIs(engine.text_rec.postprocess_op,decoder)
        self.assertEqual(other.infer,'untouched')

    def test_changed_model_rejected_before_inference(self):
        from openvino_v4 import verified_models,MODELS
        with tempfile.TemporaryDirectory() as folder:
            for filename,_ in MODELS.values():(Path(folder)/filename).write_bytes(b'wrong weights')
            with self.assertRaisesRegex(ValueError,'OCR_MODEL_HASH_MISMATCH'):
                verified_models(Path(folder))
