"""Local selected-question table evidence; original image remains authoritative."""
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
import math
import re
import hashlib

# This tested ONNX includes its character dictionary. Reject other revisions
# before RapidOCR can enter its implicit dictionary-download fallback.
_REC_SHA256='6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884'

class _Cells(HTMLParser):
    def __init__(self):super().__init__();self.cells=[];self.current=None
    def handle_starttag(self,tag,attrs):
        if tag in ('td','th'):
            if self.current is not None:raise ValueError('NESTED_CELL')
            self.current={'text':'','attrs':dict(attrs)}
    def handle_endtag(self,tag):
        if tag in ('td','th'):
            if self.current is None:raise ValueError('INVALID_CELL')
            self.cells.append(self.current);self.current=None
    def handle_data(self,data):
        if self.current is not None:self.current['text']+=data

def structured_cells(html,logic):
    if len(html)>100000:raise ValueError('TABLE_TOO_LARGE')
    parser=_Cells();parser.feed(html)
    if parser.current is not None or len(parser.cells)!=len(logic) or not 1<=len(logic)<=500:
        raise ValueError('CELL_COUNT_MISMATCH')
    cells=[];occupied=set()
    for raw,coords in zip(parser.cells,logic):
        if len(coords)!=4 or any(not math.isfinite(float(x)) or int(x)!=x for x in coords):raise ValueError('INVALID_CELL_INDEX')
        row,end_row,col,end_col=map(int,coords)
        if not (0<=row<=end_row<100 and 0<=col<=end_col<100):raise ValueError('INVALID_CELL_INDEX')
        rs=end_row-row+1;cs=end_col-col+1
        if int(raw['attrs'].get('rowspan',1))!=rs or int(raw['attrs'].get('colspan',1))!=cs:raise ValueError('SPAN_MISMATCH')
        area={(r,c) for r in range(row,end_row+1) for c in range(col,end_col+1)}
        if occupied & area:raise ValueError('OVERLAPPING_CELLS')
        occupied|=area
        if len(raw['text'])>12000:raise ValueError('TABLE_TOO_LARGE')
        cells.append({'row':row,'column':col,'rowSpan':rs,'columnSpan':cs,'text':raw['text'].strip(),'confidence':None})
    if len(occupied)!=(max(r for r,c in occupied)+1)*(max(c for r,c in occupied)+1):raise ValueError('MISSING_CELLS')
    return cells

def table_issues(cells,scores):
    notes=[]
    if not scores or any(not math.isfinite(float(s)) or not .98<=float(s)<=1 for s in scores):
        notes.append('表格文字置信度不足，请云端核对')
    if any(not c['text'] for c in cells):notes.append('表格存在空单元格，请核对是否漏字')
    if any(re.search(r"\d\s*['’‘`]\s*[℃°A-Za-z]",c['text']) for c in cells):
        notes.append('表格数值与单位之间存在异常符号，请云端核对')
    return notes

@lru_cache(maxsize=1)
def _table(model_path):
    path=Path(model_path)
    if not path.is_absolute() or not path.is_file() or path.suffix.lower()!='.onnx':raise ValueError('TABLE_MODEL_NOT_CONFIGURED')
    from rapid_table import RapidTable,RapidTableInput,ModelType
    # Construct structure-only, then attach supplied OCR; never instantiate a
    # second OCR engine or permit the package to auto-acquire its OCR weights.
    engine=RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS,model_dir_or_path=str(path),
                      use_ocr=False,engine_cfg={'intra_op_num_threads':2,'inter_op_num_threads':1}))
    engine.cfg.use_ocr=True
    return engine

@lru_cache(maxsize=1)
def _modern_ocr(directory):
    path=Path(directory)
    files={'Det':'PP-OCRv6_det_small.onnx','Cls':'ch_ppocr_mobile_v2.0_cls_mobile.onnx','Rec':'PP-OCRv6_rec_small.onnx'}
    if not path.is_absolute() or any(not (path/name).is_file() for name in files.values()):raise ValueError('TABLE_OCR_NOT_CONFIGURED')
    with (path/files['Rec']).open('rb') as source:
        if hashlib.file_digest(source,'sha256').hexdigest()!=_REC_SHA256:raise ValueError('TABLE_OCR_UNTESTED_DICTIONARY')
    from rapidocr import RapidOCR
    params={f'{key}.model_path':str(path/name) for key,name in files.items()}
    params.update({'EngineConfig.onnxruntime.intra_op_num_threads':2,'EngineConfig.onnxruntime.inter_op_num_threads':1})
    return RapidOCR(params=params)

def recognize_tables(image,layout_model,table_model,modern_ocr_dir=None):
    import numpy as np
    from rapid_layout_adapter import detect_rapid_layout
    detected=[b for b in detect_rapid_layout(image,layout_model) if b['label']=='table']
    boxes=[b for b in detected if b['score']>=.7]
    if not 1<=len(boxes)<=8:return {'tables':[],'issues':['未定位到可靠表格或表格过多，请云端核对']}
    engine=_table(table_model);tables=[]
    issues=['存在低置信表格区域，请云端核对是否漏表'] if len(detected)!=len(boxes) else []
    for box in boxes:
        x,y,r,b=box['coordinate']
        x=max(0,math.floor(x)-4);y=max(0,math.floor(y)-4);r=min(image.width,math.ceil(r)+4);b=min(image.height,math.ceil(b)+4)
        arr=np.asarray(image.crop((x,y,r,b)))[:,:,::-1].copy()
        if modern_ocr_dir:
            result=_modern_ocr(modern_ocr_dir)(arr)
            quads,texts,scores=result.boxes,result.txts,result.scores
        else:
            from ocr_runtime import ocr_engine
            result,_=ocr_engine()(arr);result=result or []
            quads=np.asarray([v[0] for v in result]);texts=tuple(v[1] for v in result);scores=tuple(v[2] for v in result)
        if quads is None or not len(quads):issues.append('表格文字未识别，请云端核对');continue
        result=engine(arr,ocr_results=[(quads,texts,scores)])
        cells=structured_cells(result.pred_htmls[0],result.logic_points[0])
        issues+=table_issues(cells,list(scores))
        # Keep OCR geometry as evidence; model cell polygons are not trusted as
        # exact visual boundaries. Row/column structure is separate from boxes.
        tables.append({'bbox':{'x':x,'y':y,'width':r-x,'height':b-y},'cells':cells,
                       'ocrRows':[{'text':str(t),'quad':[[float(a)+x,float(c)+y] for a,c in q]} for q,t in zip(quads,texts)]})
    return {'tables':tables,'issues':list(dict.fromkeys(issues))}
