"""Optional local Paddle layout inference. No implicit weight acquisition."""
from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=1)
def _model(directory):
    path=Path(directory)
    if not path.is_absolute() or any(not (path/name).is_file() for name in ('inference.json','inference.pdiparams','inference.yml')):
        raise ValueError('LAYOUT_MODEL_NOT_CONFIGURED')
    # Paddle's Windows native reader cannot reliably open non-ASCII model paths.
    if os.name=='nt' and not str(path).isascii():
        raise ValueError('LAYOUT_MODEL_PATH_REQUIRES_ASCII')
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    os.environ['PADDLE_PDX_CACHE_HOME']=str(path.parent/'runtime-cache')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HOME']=str(path.parent/'hf-cache')
    from paddleocr import LayoutDetection
    return LayoutDetection(model_name='PP-DocLayout-L',model_dir=str(path),device='cpu',enable_mkldnn=False,cpu_threads=2)


def detect_layout(image, model_dir):
    import numpy as np
    model=_model(model_dir)
    result=next(iter(model.predict(np.asarray(image)[:,:,::-1].copy())))
    return result['boxes']
