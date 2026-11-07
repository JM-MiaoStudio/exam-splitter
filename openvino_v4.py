"""Optional, serial CPU runtime for the *same* pinned legacy PP-OCRv4 models.

Only inference sessions change on a private RapidOCR instance. Its image
preparation, CTC dictionary and postprocessing stay untouched. No downloads,
global monkeypatches, device fallback or student-result cache.
"""
import hashlib
import sys
from importlib.metadata import version
from pathlib import Path

MODELS={
    'Det':('ch_PP-OCRv4_det_infer.onnx','d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9'),
    'Cls':('ch_ppocr_mobile_v2.0_cls_infer.onnx','e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c'),
    'Rec':('ch_PP-OCRv4_rec_infer.onnx','48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b'),
}


def verified_models(directory):
    paths={}
    for key,(filename,digest) in MODELS.items():
        path=directory/filename
        with path.open('rb') as source:
            if hashlib.file_digest(source,'sha256').hexdigest()!=digest:
                raise ValueError('OCR_MODEL_HASH_MISMATCH')
        paths[key]=path
    return paths


class InferenceSession:
    def __init__(self,core,path,threads):
        if type(threads) is not int or not 1<=threads<=16:
            raise ValueError('INVALID_OCR_THREADS')
        self.compiled=core.compile_model(str(path),'CPU',{
            'INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':threads,
            'NUM_STREAMS':1,'PERFORMANCE_HINT':'LATENCY',
        })

    def __call__(self,array):
        outputs=self.compiled([array])
        return [outputs[port].copy() for port in self.compiled.outputs]


def replace_sessions(engine,sessions):
    engine.text_det.infer=sessions['Det']
    engine.text_cls.infer=sessions['Cls']
    engine.text_rec.session=sessions['Rec']
    return engine


def model_sessions(core_factory,paths,threads,cache_dir=None):
    if cache_dir is not None and not Path(cache_dir).is_absolute():
        raise ValueError('INVALID_OCR_CACHE_DIR')
    core=core_factory()
    cached=False
    if cache_dir is not None:
        try:
            # Models are verified by create_engine before this directory is
            # touched. OpenVINO also keys compiled blobs by model and device.
            directory=Path(cache_dir)/'ppocrv4-ov2026.4.1-f32'
            directory.mkdir(parents=True,exist_ok=True)
            core.set_property({'CACHE_DIR':str(directory)})
            cached=True
        except (OSError,RuntimeError):
            print('OCR_MODEL_CACHE_BYPASSED',file=sys.stderr)
            core=core_factory()
    try:
        return {key:InferenceSession(core,path,threads) for key,path in paths.items()}
    except RuntimeError:
        if not cached:raise
        # An optional broken/unreadable cache must not change the OCR model or
        # send an otherwise readable page to the paid cloud. Retry only once,
        # with the same verified paths/precision on a new uncached core.
        print('OCR_MODEL_CACHE_BYPASSED',file=sys.stderr)
        core=core_factory()
        return {key:InferenceSession(core,path,threads) for key,path in paths.items()}


def create_engine(threads,cache_dir=None):
    if version('rapidocr-onnxruntime')!='1.4.4' or version('openvino')!='2026.4.1':
        raise ValueError('OCR_RUNTIME_VERSION_UNTESTED')
    import rapidocr_onnxruntime
    from openvino import Core
    paths=verified_models(Path(rapidocr_onnxruntime.__file__).parent/'models')
    # Let the pinned library initialize its exact dictionary/decoder/config.
    # Temporary ORT sessions are released when replaced below. Cold-start
    # benchmarking must include this cost; steady-state inference is OpenVINO.
    engine=rapidocr_onnxruntime.RapidOCR(intra_op_num_threads=threads,inter_op_num_threads=1)
    sessions=model_sessions(Core,paths,threads,cache_dir)
    return replace_sessions(engine,sessions)
