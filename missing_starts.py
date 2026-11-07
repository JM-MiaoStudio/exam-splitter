"""Review-only candidates backed by explicit exercise structure, never numbering guesses."""
import copy
import math
import re
from layout import clean_spans,question_starts,question_number

def _text_boxes(boxes,width,height):
    found=[]
    for b in boxes:
        x,y,r,t=b['coordinate']
        if b['label']=='text' and b['score']>=.65 and 0<=x<r<=width and 0<=y<t<=height:
            found.append((x,y,r,t))
    return sorted(set(found),key=lambda b:(b[1],b[0]))

def _inside(s,box):
    x,y,r,b=box
    return x<=(s.x+s.right)/2<=r and y<=(s.y+s.bottom)/2<=b

def _block(box,number,note):
    x,y,r,b=box
    return {'id':'pending','ordinal':1,'printedNumber':str(number) if number is not None else None,
            'bbox':{'x':x,'y':y,'width':r-x,'height':b-y},'questionText':'','options':[],
            'studentAnswer':'','subject':'','questionType':'','hasFigure':False,'unreadableParts':[note]}

def _ids(blocks):
    for i,b in enumerate(blocks):b['id']=f'b{i+1}';b['ordinal']=i+1
    return blocks

def english_region_candidates(image,rows,boxes):
    spans=clean_spans(rows,image.width,image.height)
    declarations=[int(m[1]) for s in spans if s.score>=.8 for m in
                  re.finditer(r'完成句子.{0,8}一空一词.{0,12}本题共\s*(\d{1,2})\s*小题',s.text)]
    if not declarations or len(set(declarations))!=1 or not 6<=declarations[0]<=30:return None
    candidates=[]
    for box in _text_boxes(boxes,image.width,image.height):
        english=[s for s in spans if _inside(s,box) and s.score>=.7 and
                 len(re.findall('[A-Za-z]',s.text))>=8 and not re.search('[\u4e00-\u9fff]',s.text)]
        if english:candidates.append((box,english))
    if len(candidates)!=declarations[0]:return None
    if max(b[0] for b,_ in candidates)-min(b[0] for b,_ in candidates)>image.width*.2:return None
    anchors=question_starts(rows,image.width,image.height)
    assigned=[[] for _ in candidates]
    for s,n,ns in anchors:
        distances=sorted((abs(s.y-box[1]),i) for i,(box,_) in enumerate(candidates)
                         if box[0]-20<=s.x<=box[0]+image.width*.12)
        if not distances or distances[0][0]>max(12,s.height*.6):continue
        if len(distances)>1 and distances[1][0]-distances[0][0]<2:continue
        assigned[distances[0][1]].append((s,n,ns))
    result=[];observed=[]
    for index,(box,body) in enumerate(candidates):
        # A number belongs to the first printed line, not a handwritten answer
        # or another paragraph. Baseline y tolerates tilted full-line quads.
        nearby=[(s,n,ns) for s,n,ns in assigned[index] if n is not None]
        nums={n for _,n,_ in nearby}
        if len(nums)>1:return None
        number=next(iter(nums)) if nums else None
        observed.append(number)
        ink=[s for s in spans if _inside(s,box)]
        rect=(max(0,math.floor(min([box[0]]+[s.x for s in ink]))-3),
              max(0,math.floor(min([box[1]]+[s.y for s in ink]))-3),
              min(image.width,math.ceil(max([box[2]]+[s.right for s in ink]))+3),
              min(image.height,math.ceil(max([box[3]]+[s.bottom for s in ink]))+3))
        b=_block(rect,number,'独立英语练习的版面候选框，请核对题号、邻题边缘及手写内容')
        if number is None:b['unreadableParts'].append('题号未辨认，未根据顺序补造，请云端补齐或人工核对')
        b['regions']=[copy.deepcopy(b['bbox'])];result.append(b)
    # Check observed sequence against positions, but never fill missing values.
    if sum(n is not None for n in observed)<6 or any(n is not None and n!=i+1 for i,n in enumerate(observed)):return None
    return _ids(result)

def recover_choice_gaps(blocks,rows,boxes,width,height):
    """Split a merged host only for a bounded gap with independent choice stems."""
    spans=clean_spans(rows,width,height);out=[]
    for index,original in enumerate(blocks):
        current=copy.deepcopy(original);out.append(current)
        if index+1>=len(blocks) or current.get('regions'):continue
        following=blocks[index+1]
        a,c=current.get('printedNumber'),following.get('printedNumber')
        if not a or not c or not str(a).isdigit() or not str(c).isdigit():continue
        missing=int(c)-int(a)-1
        if not 1<=missing<=4:continue
        host=current['bbox'];end=following['bbox']['y']
        if abs(host['x']-following['bbox']['x'])>width*.02:continue
        eligible=[]
        for box in _text_boxes(boxes,width,height):
            if box[1]<=host['y']+20 or box[3]>end+2 or not host['x']<=box[0]<host['x']+host['width']*.2:continue
            content=[s for s in spans if _inside(s,box) and s.score>=.72]
            if not content:continue
            top_line=min(s.y for s in content)
            first=min((s for s in content if s.y<=top_line+s.height*.7),key=lambda s:s.x)
            if not re.match(r'^(?:已知|若|设|将|如图|下列|某)',first.text):continue
            labels={m[1] for s in content for m in re.finditer(r'(?:^|\s)([A-D])(?:[.、:：]|$)',s.text)}
            if len(labels)<2 or any(question_number(s.text) is not None for s in content):continue
            top=max(host['y']+1,math.floor(min(box[1],min(s.y for s in content)))-3)
            eligible.append((top,box))
        if len(eligible)!=missing or any(b[0]-a[0]<20 for a,b in zip(eligible,eligible[1:])):continue
        old_end=host['y']+host['height']
        host['height']=eligible[0][0]-host['y']
        current['unreadableParts'].append('原题框内发现独立选择题，已分出待核对候选，请核对选项边界')
        for j,(top,_) in enumerate(eligible):
            bottom=eligible[j+1][0] if j+1<len(eligible) else old_end
            if bottom<=top:continue
            out.append(_block((host['x'],top,host['x']+host['width'],bottom),None,
                       '题号未辨认，依据独立题干及选项保留候选框；未补造题号，请云端补齐或人工核对'))
    return _ids(out)
