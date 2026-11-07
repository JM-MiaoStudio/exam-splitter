"""Bounded stdin JSON -> local table evidence, no cloud or generative model."""
import contextlib,json,os,sys
from worker import validate_image,MAX_INPUT_BYTES
from local_table import recognize_tables

def main():
    try:
        raw=sys.stdin.buffer.read(MAX_INPUT_BYTES+1)
        if len(raw)>MAX_INPUT_BYTES:raise ValueError('INPUT_TOO_LARGE')
        page=json.loads(raw);image=validate_image(page)
        with contextlib.redirect_stdout(sys.stderr):
            result=recognize_tables(image,os.environ['EXAM_RAPID_LAYOUT_MODEL'],os.environ['EXAM_TABLE_MODEL'],os.environ.get('EXAM_TABLE_OCR_DIR'))
        result.update({'source':'Local RapidTable SLANet-plus','imageHash':page['imageHash'],
                       'width':image.width,'height':image.height,'coordinateSpace':'uploaded-image'})
        sys.stdout.write(json.dumps(result,ensure_ascii=False));return 0
    except Exception:
        sys.stderr.write('LOCAL_TABLE_FAILED\n');return 70

if __name__=='__main__':raise SystemExit(main())
