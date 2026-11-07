"""Review-only paragraph candidates; never reconstruct cropped question numbers.

Used only after the numbered splitter fails or merges a mathematical edge crop.
Document text regions supply missing starts; formulae and figures stay with the
surrounding paragraph. Every result requires review because page pixels are lost.
"""
import math
import re
from statistics import median
from layout import clean_spans, question_number, question_starts

_SECTION=re.compile(r'^(?:[一二三四五六七八九十]+[、.]\s*)?(?:选择题|单选题|多选题|填空题|解答题|计算题|证明题)|^第\s*\d+\s*页')
_CONTINUATION=re.compile(r'^(?:当|则|所以|因此|其中|故|的|\([1-9]|（[1-9]|[①②③④⑤⑥⑦⑧⑨])')
_OPENING=re.compile(r'^(?:已知|如图|设|若|某|在.*中)')
_WARNING='照片边缘裁切，题号或题干可能不完整；此框为疑似整题，请核对原卷'


def _regions(boxes,width,height):
    output=[]
    for box in boxes:
        try:
            x,y,r,b=box['coordinate'];score=float(box['score'])
            if not all(math.isfinite(v) for v in (x,y,r,b,score)) or not (0<=x<r<=width and 0<=y<b<=height):
                continue
            if score>=.6 and box['label'] in {'text','image','chart','formula'}:
                output.append((box['label'],x,y,r,b))
        except (KeyError,TypeError,ValueError):
            continue
    return output


def split_cropped_questions(rows, layout_boxes, width, height):
    spans=clean_spans(rows,width,height)
    edge=[s for s in spans if s.score>=.72 and s.x<=width*.025 and len(re.findall(r'[\u4e00-\u9fff]',s.text))>=4]
    # An observed edge number supplies evidence of a worksheet, not its missing
    # leading digits. Unnumbered prose and intact margins remain untouched.
    if len(edge)<2 or not any(s.x<=width*.025 and n is not None for s,n,_ in question_starts(rows,width,height)):
        return None
    shapes=_regions(layout_boxes,width,height)
    inline_math=sum(bool(re.search(r'(?:sin|cos|tan|函数|向量|二面角)|[=√∠≤≥]',s.text)) for s in spans if s.score>=.72)
    if not any(s[0]=='formula' for s in shapes) and not (inline_math>=2 and any(s[0] in {'image','chart'} for s in shapes)):
        return None
    line_height=median(s.height for s in spans if s.score>=.72)
    graphic_padding=max(4,math.ceil(line_height*.15))
    headings=[s.y for s in spans if s.score>=.72 and _SECTION.search(s.text)]
    paragraphs=[]
    for label,x,y,r,b in sorted(shapes,key=lambda s:s[2]):
        if label!='text' or x>width*.025 or r-x<width*.5:
            continue
        lines=[s for s in spans if s.score>=.72 and x<=s.x<width*.3 and y-line_height*.5<=s.y and s.y<b]
        first_y=min((s.y for s in lines),default=0)
        lead=min((s for s in lines if s.y<=first_y+line_height),key=lambda s:s.x,default=None)
        if lead and _SECTION.search(lead.text):
            headings.append(y)
            continue
        current=[y,b,lead]
        if paragraphs:
            old=paragraphs[-1]
            overlap=min(b,old[1])-max(y,old[0])
            if overlap>.25*min(b-y,old[1]-old[0]) or lead and _CONTINUATION.search(lead.text):
                old[0]=min(old[0],y);old[1]=max(old[1],b)
                continue
        paragraphs.append(current)
    if not 3<=len(paragraphs)<=12:
        return None
    # Two independent starts are needed before treating layout paragraphs as
    # suspected questions. No per-sample text, coordinates or expected count.
    if sum(lead is not None and (question_number(lead.text) is not None or bool(_OPENING.search(lead.text)))
           for _,_,lead in paragraphs)<2:
        return None
    starts=[]
    for top,bottom,lead in paragraphs:
        text_tops=[s.y for s in spans if s.score>=.72 and top-line_height*.5<=s.y<bottom and s.bottom>top]
        starts.append(max(0,math.floor(min([top]+text_tops))-3))
    blocks=[]
    for index,(top,body_bottom,lead) in enumerate(paragraphs):
        limit=starts[index+1] if index+1<len(starts) else height
        next_headings=[y for y in headings if y>body_bottom]
        if next_headings:limit=min(limit,math.floor(min(next_headings))-3)
        begin=starts[index]
        end=max(body_bottom,limit)
        # Keep whole known graphics/formulae whose center is in this paragraph's
        # band. A tilted next stem may overlap their axis envelope; retain both
        # crops and flag overlap rather than cut either source rectangle.
        for label,x,y,r,b in shapes:
            if label!='text' and starts[index]<=(y+b)/2<limit:
                begin=max(0,min(begin,math.floor(y)-graphic_padding))
                end=max(end,b+graphic_padding)
        for span in spans:
            if span.score>=.72 and starts[index]<=(span.y+span.bottom)/2<limit and not _SECTION.search(span.text):
                begin=max(0,min(begin,math.floor(span.y)-3))
                end=max(end,span.bottom+3)
        bottom=min(height,math.ceil(end))
        block={'id':f'b{index+1}','ordinal':index+1,'printedNumber':None,
               'bbox':{'x':0,'y':begin,'width':width,'height':bottom-begin},
               'questionText':'','options':[],'studentAnswer':'','subject':'','questionType':'',
               'unreadableParts':[_WARNING],'hasFigure':True}
        # Explicit geometry prevents the client's legacy overlap clip from
        # trimming formula/diagram edges that were deliberately preserved.
        block['regions']=[dict(block['bbox'])]
        blocks.append(block)
    for first,second in zip(blocks,blocks[1:]):
        if first['bbox']['y']+first['bbox']['height']>second['bbox']['y']:
            warning='倾斜题干与配图的外接框重叠，已保留完整区域，请核对相邻题边界'
            first['unreadableParts'].append(warning);second['unreadableParts'].append(warning)
    return blocks
