"""Page-level coverage hints, never missing-question reconstruction or grading."""
import re
from layout import clean_spans, figure_numbers


def page_warnings(rows, blocks, width, height):
    if not blocks:
        return ['未识别到可靠题框，请核对图片方向和清晰度，重新拍照或手动框选']
    warnings=[]
    for a,b in zip(blocks,blocks[1:]):
        first,last=a.get('printedNumber'),b.get('printedNumber')
        if first and last and re.fullmatch(r'\d+',first) and re.fullmatch(r'\d+',last) and int(last)>int(first)+1:
            warnings.append('题号不连续，可能存在漏题，请对照原图核对');break
    regions=[r for b in blocks for r in b.get('regions',[b.get('bbox')]) if r]
    uncovered=[]
    for s in clean_spans(rows,width,height):
        # Exclude ordinary headings, captions and short isolated formula/table
        # tokens. At least two separate long text lines must be left outside.
        if s.score<.65 or len(s.text.strip())<10 or figure_numbers(s.text):continue
        if re.search(r'课后习题|本题共|本大题|满分|考试时间|姓名|学号|第\s*\d+\s*页',s.text):continue
        x,y=(s.x+s.right)/2,(s.y+s.bottom)/2
        if any(r['x']<=x<=r['x']+r['width'] and r['y']<=y<=r['y']+r['height'] for r in regions):continue
        # Deduplicate partial strip OCR of the same physical line as well.
        if not any(abs(y-old)<s.height*.6 for old in uncovered):uncovered.append(y)
    if len(uncovered)>=2:
        warnings.append('题框外仍有较多文字，可能存在漏题或未归属内容，请对照原图核对并补充题框')
    return warnings
