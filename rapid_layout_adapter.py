"""Explicit-file V3 inference on upload pixels. No runtime model downloads."""
from functools import lru_cache
from pathlib import Path
import math


@lru_cache(maxsize=1)
def _engine(model_path):
    path=Path(model_path)
    if not path.is_absolute() or not path.is_file() or path.suffix.lower()!='.onnx':
        raise ValueError('RAPID_LAYOUT_MODEL_NOT_CONFIGURED')
    from rapid_layout import RapidLayout
    return RapidLayout(model_type='pp_doc_layoutv3',model_dir_or_path=str(path),
                       engine_cfg={'intra_op_num_threads':2,'inter_op_num_threads':1})

def normalize_boxes(result,width,height):
    output=[]
    aliases={'inline_formula':'formula','display_formula':'formula','figure_title':'figure_caption'}
    for box,label,score in zip(result.boxes,result.class_names,result.scores):
        try:
            values=[float(x) for x in box];score=float(score)
            if len(values)!=4 or not all(math.isfinite(v) for v in values+[score]):continue
            x,y,r,b=values
            if not (0<=x<r<=width and 0<=y<b<=height and .5<=score<=1):continue
            output.append({'label':aliases.get(label,label),'score':score,'coordinate':values})
        except (TypeError,ValueError):continue
    return output

def detect_rapid_layout(image,model_path):
    import numpy as np
    result=_engine(model_path)(np.asarray(image)[:,:,::-1].copy())
    return normalize_boxes(result,image.width,image.height)

def recover_region_rows(image,boxes,*,recognize_crop=None):
    """Bounded recovery after grouping abstains. OCR output, never layout guesses."""
    from dense_english import _recognize_strip
    recognize_crop=recognize_crop or _recognize_strip
    rows=[];seen=set();pixels=0;count=0
    for box in boxes:
        if box['label'] not in {'text','figure_caption'} or box['score']<.65:continue
        x,y,r,b=box['coordinate']
        crop=(max(0,math.floor(x)-4),max(0,math.floor(y)-4),min(image.width,math.ceil(r)+4),min(image.height,math.ceil(b)+4))
        if crop in seen:continue
        seen.add(crop)
        area=(crop[2]-crop[0])*(crop[3]-crop[1])
        if count>=50 or pixels+area>image.width*image.height*2:break
        count+=1;pixels+=area
        for row in recognize_crop(image.crop(crop)):
            q=row['quad'];xs=[p[0] for p in q];ys=[p[1] for p in q]
            clipped=(crop[0]>0 and min(xs)<=1 or crop[1]>0 and min(ys)<=1 or
                     crop[2]<image.width and max(xs)>=crop[2]-crop[0]-1 or
                     crop[3]<image.height and max(ys)>=crop[3]-crop[1]-1)
            rows.append(dict(row,quad=[[x+crop[0],y+crop[1]] for x,y in q],
                             source='strip' if clipped else 'layout',clipped=bool(clipped)))
    return rows
