"""Single-page offline worker. stdin JSON -> stdout PhotoPage; no files/network.

Exit 64 means invalid input, 65 means geometry needs manual selection, 70 means
runtime/dependency failure. stderr only contains a constant diagnostic code.
"""
import base64
import contextlib
import hashlib
import io
import json
import os
import sys
from ocr_runtime import ocr_engine
from performance import stage,profiled

from PIL import Image, UnidentifiedImageError
from layout import LayoutError, split_questions
from layout_model import detect_layout

MAX_INPUT_BYTES = 3_300_000


def validate_image(page):
    try:
        if page.get('version') != 1 or not isinstance(page.get('pageId'),str) or not 1 <= len(page['pageId']) <= 100:
            raise LayoutError('INVALID_INPUT')
        width,height=page['width'],page['height']
        if type(width) is not int or type(height) is not int or not 1 <= width <= 2048 or not 1 <= height <= 2048:
            raise LayoutError('INVALID_IMAGE_SIZE')
        images=page['images']
        if len(images)!=1 or images[0]['mime']!='image/jpeg':
            raise LayoutError('INVALID_INPUT')
        data=base64.b64decode(images[0]['base64'],validate=True)
        if len(data)>1_572_864:
            raise LayoutError('IMAGE_TOO_LARGE')
        if hashlib.sha256(data).hexdigest()!=page['imageHash']:
            raise LayoutError('IMAGE_HASH_MISMATCH')
        image=Image.open(io.BytesIO(data))
        if image.format!='JPEG':
            raise LayoutError('INVALID_JPEG')
        if image.size!=(width,height):
            raise LayoutError('IMAGE_DIMENSIONS_MISMATCH')
        if image.getexif().get(274,1)!=1:
            raise LayoutError('IMAGE_ORIENTATION_NOT_NORMALIZED')
        image.load()
        return image.convert('RGB')
    except LayoutError:
        raise
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):
        raise LayoutError('INVALID_JPEG') from None
    except (KeyError,ValueError,TypeError,IndexError):
        raise LayoutError('INVALID_INPUT') from None


def recognize(image):
    import numpy as np
    with stage('ocr.engine'):
        engine=ocr_engine()
    bgr=np.asarray(image)[:,:,::-1].copy()
    rows=[]
    width=image.width
    # Whole-page layout + narrow left edges of the two common exam columns.
    # A long perspective-skewed line can be missed by DB detection; the short
    # strip retains the printed number, without altering upload coordinates.
    for index,(left,right) in enumerate([(0,width),(0,round(width*.4)),(width//2,min(width,round(width*.9)))]):
        with stage(('ocr.page','ocr.left','ocr.right')[index]):
            result,_=engine(bgr[:,left:right])
        for quad,text,score in result or []:
            rows.append({'quad':[[x+left,y] for x,y in quad],'text':text,'score':score,'source':'page' if index==0 else 'strip'})
    return rows


def refine_with_layout(blocks, rows, boxes, width, height):
    from layout_regions import refine_with_layout as refine
    return refine(blocks, rows, boxes, width, height)


def group_rows(image,rows,layout_boxes=None):
    try:
        blocks=split_questions(rows,image.width,image.height)
    except LayoutError as error:
        if str(error) not in {'NO_QUESTION_ANCHORS','AMBIGUOUS_COLUMNS','AMBIGUOUS_QUESTION_STARTS','TOO_MANY_QUESTIONS'}:
            raise
        # Dense English may have a complete observed sequence whose tilted
        # quad envelopes overlap. Retain whole ink with a review warning.
        from dense_english import split_dense_english
        blocks=split_dense_english(image,rows) or []
    if layout_boxes is not None:
        if not blocks:
            from missing_starts import english_region_candidates
            blocks=english_region_candidates(image,rows,layout_boxes) or blocks
        if len(blocks)<=1:
            from edge_stems import split_cropped_questions
            blocks=split_cropped_questions(rows,layout_boxes,image.width,image.height) or blocks
        from missing_starts import recover_choice_gaps
        blocks=recover_choice_gaps(blocks,rows,layout_boxes,image.width,image.height)
        blocks=refine_with_layout(blocks,rows,layout_boxes,image.width,image.height)
    return blocks


@profiled
def process_page(page, *, layout_model_dir=None, rapid_layout_model=None, enhance_fallback=False, cloud_ocr=None, prepare='off'):
    with stage('image.decode'):
        image=validate_image(page)
    cloud=None
    if cloud_ocr is not None:
        from cloud_geometry import cloud_rows
        cloud=cloud_rows(cloud_ocr,image_hash=page['imageHash'],width=image.width,height=image.height)
    if prepare not in ('off','perspective','shadow','auto'):raise LayoutError('INVALID_PREPARE_MODE')
    if prepare!='off':
        from document_preprocess import prepare_document, restore_rows
        with stage('image.prepare'):
            prepared,inverse,_=prepare_document(image,perspective=prepare in ('perspective','auto'),shadow=prepare in ('shadow','auto'))
        rows=restore_rows(recognize(prepared),inverse,image.width,image.height)
    else:rows=recognize(image)
    if layout_model_dir and rapid_layout_model:raise LayoutError('CONFLICTING_LAYOUT_MODELS')
    if rapid_layout_model:
        from rapid_layout_adapter import detect_rapid_layout
        with stage('layout'):
            boxes=detect_rapid_layout(image,rapid_layout_model)
    else:
        with stage('layout'):
            boxes=detect_layout(image,layout_model_dir) if layout_model_dir else None
    with stage('grouping'):
        blocks=group_rows(image,rows,boxes)
    from assistance import choose_assisted
    if rapid_layout_model and not blocks:
        from rapid_layout_adapter import recover_region_rows
        with stage('ocr.layout_recovery'):
            recovered=rows+recover_region_rows(image,boxes)
        with stage('grouping.recovery'):
            candidates=group_rows(image,recovered,boxes)
        blocks=choose_assisted(blocks,candidates,'本地版面')
        if blocks:rows=recovered
    if enhance_fallback and not blocks:
        from preprocess import flatten_background
        enhanced=flatten_background(image)
        enhanced_rows=recognize(enhanced)
        blocks=choose_assisted(blocks,group_rows(enhanced,enhanced_rows,boxes),'增强')
        if blocks:rows=enhanced_rows
    if cloud is not None:
        candidates=group_rows(image,cloud,boxes)
        was_empty=not blocks
        blocks=choose_assisted(blocks,candidates,'云端')
        if was_empty and any(r.get('clipped') for r in cloud):
            for block in blocks:block['unreadableParts'].append('原图边缘文字可能截断，请核对条件是否完整')
    from transcription import assemble_questions
    with stage('assembly'):
        blocks=assemble_questions(blocks,rows,image.width,image.height)
    from quality import page_warnings
    with stage('quality'):
        warnings=page_warnings(rows,blocks,image.width,image.height)
    return {key:page[key] for key in ('version','pageId','imageHash','width','height')} | {
        'coordinateSpace':'uploaded-image','blocks':blocks,**({'warnings':warnings} if warnings else {})}


def configured_process(page):
    return process_page(page,layout_model_dir=os.environ.get('EXAM_LAYOUT_MODEL_DIR'),
                        rapid_layout_model=os.environ.get('EXAM_RAPID_LAYOUT_MODEL'),
                        enhance_fallback=os.environ.get('EXAM_ENHANCE_FALLBACK')=='1',cloud_ocr=page.get('_cloudOcr'),
                        prepare=os.environ.get('EXAM_PREPARE','off'))


def serve(source=None, output=None):
    """Serial, bounded JSON-lines IPC. Models persist; request data does not."""
    source = source if source is not None else sys.stdin.buffer
    output = output if output is not None else sys.stdout
    while True:
        line=source.readline(MAX_INPUT_BYTES+2049)
        if not line:return 0
        try:
            if len(line)>MAX_INPUT_BYTES+2048 or not line.endswith(b'\n'):
                return 64  # Do not parse the remainder as another request.
            request=json.loads(line)
            if not isinstance(request,dict) or not isinstance(request.get('id'),str) or not 1<=len(request['id'])<=100 or not isinstance(request.get('input'),dict):
                return 64
        except (ValueError,UnicodeDecodeError):
            return 64
        response={'id':request['id']}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                response['result']=configured_process(request['input'])
        except LayoutError:
            response['error']='SPLITTER_INPUT_OR_LAYOUT_FAILED'
        except Exception:
            response['error']='SPLITTER_RUNTIME_FAILED'
        output.write(json.dumps(response,ensure_ascii=True,separators=(',',':'))+'\n')
        output.flush()
        del request,response,line


def main():
    try:
        payload=sys.stdin.buffer.read(MAX_INPUT_BYTES+1)
        if len(payload)>MAX_INPUT_BYTES:
            raise LayoutError('INPUT_TOO_LARGE')
        page=json.loads(payload)
        if not isinstance(page,dict):
            raise LayoutError('INVALID_INPUT')
        # Libraries must not contaminate the machine-readable stdout protocol.
        with contextlib.redirect_stdout(sys.stderr):
            result=configured_process(page)
        sys.stdout.write(json.dumps(result,ensure_ascii=True,separators=(',',':')))
        return 0
    except LayoutError as error:
        # Codes are controlled by this module, never OCR text / image bytes.
        sys.stderr.write(str(error)+'\n')
        return 65 if str(error).startswith(('NO_QUESTION','AMBIGUOUS','TOO_MANY')) else 64
    except (json.JSONDecodeError,UnicodeDecodeError):
        sys.stderr.write('INVALID_INPUT\n')
        return 64
    except Exception:
        sys.stderr.write('SPLITTER_RUNTIME_FAILED\n')
        return 70


if __name__=='__main__':
    raise SystemExit(serve() if sys.argv[1:]==['--serve'] else main())
