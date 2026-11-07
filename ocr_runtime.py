"""One serial OCR engine for page, strip and recovery inference.

Separate module keeps CLI (__main__) and imported worker paths on the same cache.
Only model sessions are retained, never a per-student result cache.
"""
from functools import lru_cache
import os


@lru_cache(maxsize=1)
def _engine(threads,backend,cache_dir):
    if backend=='openvino':
        from openvino_v4 import create_engine
        return create_engine(threads,cache_dir=cache_dir)
    from rapidocr_onnxruntime import RapidOCR
    # Package-bundled models; no implicit download or provider credentials.
    return RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1)


def ocr_engine():
    backend=os.environ.get('EXAM_OCR_BACKEND','onnxruntime')
    if backend not in ('onnxruntime','openvino'):
        raise ValueError('INVALID_OCR_BACKEND')
    raw=os.environ.get('EXAM_OCR_THREADS','2')
    if not raw.isascii() or not raw.isdigit() or not 1<=int(raw)<=16:
        raise ValueError('INVALID_OCR_THREADS')
    # RapidOCR leaves ORT defaults in place when a request exceeds cpu_count.
    # Cap explicitly so a smaller deployment cannot silently become unbounded.
    cache_dir=os.environ.get('EXAM_OV_CACHE_DIR') if backend=='openvino' else None
    return _engine(min(int(raw),os.cpu_count() or 1),backend,cache_dir)


# Keep the explicit lifecycle hook used by offline tests and benchmark tools.
ocr_engine.cache_clear=_engine.cache_clear
