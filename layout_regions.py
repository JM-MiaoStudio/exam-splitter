"""Conservative same-page crops from existing OCR and document-layout evidence.

This module neither transcribes questions nor infers figure contents. A caption
can identify a remote crop only when its unique stem explicitly references a
figure. Removing a foreign figure subtracts its rectangle from the original
crop; all other pixels (including unrecognized formulae and work) survive.
"""
from collections import Counter
from copy import deepcopy
import math
import re

from layout import LayoutError, clean_spans, figure_numbers, question_number


_REMOTE_WARNING = '发现同题号图注位于题框外，请核对并补充配图'
_FOREIGN_WARNING = re.compile(r'^题框内有其他题号的配图（([0-9]+(?:、[0-9]+)*)），请核对配图归属$')
_UNCERTAIN = '版面配图关联不确定，请人工核对图注与题干'
_NO_REFERENCE = '图注对应题干缺少明确图引用，未自动关联配图'
_ORPHAN = '配图图注未找到对应题干，请人工核对'
_REPEATED = '配图图注对应多个同号题干，未自动关联配图'
_LIMIT = '配图分离超过8个区域，保留原题框待核对'
_CROSSING = '分离配图可能截断题干、选项或作答，保留原题框待核对'
_FIGURE_REFERENCE = re.compile(r'如图|图[像象]|图为|图中|图示|下图|上图|见图|图所示')
_OPTION = re.compile(r'(?:^|\s)[A-H]\s*[.、:：)）]|^[A-H]\s+\S|^\(\d+\)')
_FORMULA = re.compile(r'[=≠≤≥<>+×÷√∫∑^∴∵]|[A-Za-z][²³⁰¹⁴⁵⁶⁷⁸⁹]|[A-Za-z0-9]\s*[-−]\s*[A-Za-z]')


def _box(rect):
    x,y,right,bottom=rect
    return {'x':x,'y':y,'width':right-x,'height':bottom-y}


def _rect(box):
    return (box['x'],box['y'],box['x']+box['width'],box['y']+box['height'])


def _span_rect(span):
    return (span.x,span.y,span.right,span.bottom)


def _contains(outer, inner):
    return outer[0]<=inner[0] and outer[1]<=inner[1] and inner[2]<=outer[2] and inner[3]<=outer[3]


def _intersects(a,b):
    return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])


def _center_inside(rect, span):
    return rect[0]<=(span.x+span.right)/2<rect[2] and rect[1]<=(span.y+span.bottom)/2<rect[3]


def _padded(rect, width, height, pad=4):
    return (max(0,math.floor(rect[0])-pad),max(0,math.floor(rect[1])-pad),
            min(width,math.ceil(rect[2])+pad),min(height,math.ceil(rect[3])+pad))


def _union(a,b):
    return (min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))


def _warn(block, message):
    if message not in block['unreadableParts']:
        block['unreadableParts'].append(message)


def _body_text(text):
    # A layout model can incorrectly include an option or worked equation in
    # an image box. Its location is not evidence that such text is an axis
    # label. Even genuine plotted equations abstain here: preserving a stem
    # takes precedence over automatic diagram removal. Short numeric ticks,
    # point labels and unit labels such as F/N and L/cm remain admissible.
    return bool(_OPTION.search(text) or _FORMULA.search(text) or
                len(re.findall(r'[\u4e00-\u9fff]',text))>=4)


def _figures(layout_boxes,width,height):
    figures=[]
    for item in layout_boxes:
        try:
            values=item['coordinate']
            score=float(item['score'])
            if item['label'] not in {'image','chart'} or not math.isfinite(score) or score<.65:
                continue
            if len(values)!=4 or not all(math.isfinite(v) for v in values):
                continue
            x,y,right,bottom=values
            if not (0<=x<right<=width and 0<=y<bottom<=height):
                continue
            rect=_padded(values,width,height)
            # Some layout versions return both image and chart for one object.
            if rect not in figures:
                figures.append(rect)
        except (KeyError,TypeError,ValueError):
            continue
    return figures


def _safe_figure_crop(crop,image,caption,spans,width,height):
    caption_rect=_padded(_span_rect(caption),width,height)
    for span in spans:
        rect=_span_rect(span)
        if not _intersects(crop,rect):
            continue
        if span is caption:
            continue
        # Even a detector covering a complete question line is not permission
        # to erase that line. Long Chinese content may also be student work.
        if question_number(span.text) is not None or figure_numbers(span.text):
            return False
        if _body_text(span.text):
            return False
        if _contains(caption_rect,rect):
            continue
        if not _contains(image,rect):
            return False
    return True


def _include_axis_labels(image,spans,width,height):
    expanded=image
    for span in spans:
        rect=_span_rect(span)
        if question_number(span.text) is not None or figure_numbers(span.text):
            continue
        if _body_text(span.text):
            continue
        overlap=max(0,min(image[2],rect[2])-max(image[0],rect[0]))*max(0,min(image[3],rect[3])-max(image[1],rect[1]))
        area=(rect[2]-rect[0])*(rect[3]-rect[1])
        spill=max(image[0]-rect[0],image[1]-rect[1],rect[2]-image[2],rect[3]-image[3])
        if overlap>=.75*area and spill<=.5*span.height:
            expanded=_union(expanded,_padded(rect,width,height,pad=2))
    return expanded


def _caption_candidates(caption,figures):
    candidates=[]
    for index,image in enumerate(figures):
        center=(caption.x+caption.right)/2
        # Captions can be above or below a diagram. Consider both sides before
        # resolving ties: accepting the old lower-caption candidate alone can
        # attach a caption midway between two diagrams to the wrong one.
        gap=max(caption.y-image[3],image[1]-caption.bottom)
        if image[0]<=center<=image[2] and -.25*caption.height<=gap<=4*caption.height:
            candidates.append((max(0,gap),index))
    return sorted(candidates)


def _pair_captions(captions,figures,spans,width,height):
    counts=Counter(number for span,numbers in captions for number in numbers)
    proposals=[]
    for caption,numbers in captions:
        if len(numbers)!=1 or counts[numbers[0]]!=1:
            continue
        candidates=_caption_candidates(caption,figures)
        if not candidates:
            continue
        # A shared/combined graphic spanning several captions, or two equally
        # near graphics under one caption, cannot be divided on label alone.
        if len(candidates)>1 and candidates[1][0]-candidates[0][0]<=.5*caption.height:
            continue
        index=candidates[0][1]
        image=_include_axis_labels(figures[index],spans,width,height)
        crop=_union(image,_padded(_span_rect(caption),width,height))
        proposals.append((numbers[0],index,caption,crop,image))
    uses=Counter(index for _,index,_,_,_ in proposals)
    pairs={}
    for number,index,caption,crop,image in proposals:
        if uses[index]!=1 or not _safe_figure_crop(crop,image,caption,spans,width,height):
            continue
        pairs[number]={'crop':crop,'caption':caption,'image':image}
    return pairs


def _subtract(rect,mask):
    if not _intersects(rect,mask):
        return [rect]
    x,y,right,bottom=rect
    mx,my,mright,mbottom=(max(x,mask[0]),max(y,mask[1]),min(right,mask[2]),min(bottom,mask[3]))
    return [piece for piece in [(x,y,right,my),(x,mbottom,right,bottom),
                                (x,my,mx,mbottom),(mright,my,right,mbottom)]
            if piece[0]<piece[2] and piece[1]<piece[3]]


def _merge_adjacent(rectangles):
    # Merge only exact shared full edges: a bounding union across a hole would
    # restore the very foreign pixels that have just been excluded.
    result=list(rectangles)
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(result):
            for j in range(i+1,len(result)):
                b=result[j]
                horizontal=a[1]==b[1] and a[3]==b[3] and (a[2]==b[0] or b[2]==a[0])
                vertical=a[0]==b[0] and a[2]==b[2] and (a[3]==b[1] or b[3]==a[1])
                if horizontal or vertical:
                    result[i]=_union(a,b)
                    result.pop(j)
                    changed=True
                    break
            if changed:
                break
    return sorted(result,key=lambda r:(r[1],r[0],r[3],r[2]))


def _keep_whole(pieces,bounds,masks,protected):
    """Move crop seams off known OCR lines/retained figures without erasing ink.

    Isolate a safe rectangular corridor containing a split item, subtract that
    corridor from the previous pieces, then put it back as one piece. The union
    of retained pixels is unchanged. Uncertain/over-complex partitions abstain.
    """
    for rect in sorted(protected,key=lambda r:(r[1],r[0])):
        if any(_contains(piece,rect) for piece in pieces):
            continue
        top=max(bounds[1],math.floor(rect[1])-2)
        left,right=bounds[0],bounds[2]
        for mask in masks:
            if mask[1]<rect[3] and top<mask[3]:
                if mask[2]<=rect[0]:
                    left=max(left,mask[2])
                elif mask[0]>=rect[2]:
                    right=min(right,mask[0])
        bottom=bounds[3]
        for mask in masks:
            if mask[0]<right and left<mask[2] and mask[1]>=rect[3]:
                bottom=min(bottom,mask[1])
        corridor=(left,top,right,bottom)
        if not _contains(corridor,rect) or any(_intersects(corridor,m) for m in masks):
            return None
        pieces=_merge_adjacent([part for piece in pieces for part in _subtract(piece,corridor)]+[corridor])
    return pieces if all(any(_contains(p,r) for p in pieces) for r in protected) else None


def refine_with_layout(blocks, rows, layout_boxes, width, height):
    if type(width) is not int or type(height) is not int or not 1<=width<=2048 or not 1<=height<=2048:
        raise LayoutError('INVALID_IMAGE_SIZE')
    result=deepcopy(blocks)
    spans=clean_spans(rows,width,height)
    figures=_figures(layout_boxes,width,height)
    captions=[(s,figure_numbers(s.text)) for s in spans if s.score>=.72 and figure_numbers(s.text)]
    pairs=_pair_captions(captions,figures,spans,width,height)
    # Rejected associations still describe potentially meaningful source ink.
    # Keep every plausible raw candidate, including ties/repeated captions and
    # images rejected for containing body text. None authorizes attachment.
    unpaired=[]
    for caption,numbers in captions:
        if any(n not in pairs for n in numbers):
            for _,index in _caption_candidates(caption,figures):
                image=_include_axis_labels(figures[index],spans,width,height)
                unpaired.append((numbers,_union(image,_padded(_span_rect(caption),width,height))))
    by_number={}
    # Ownership checks always use the baseline geometry, not a previously
    # refined neighbour's first region; iteration order must not change it.
    for item in blocks:
        try:
            by_number.setdefault(int(item['printedNumber']),[]).append(item)
        except (ValueError,TypeError):
            pass

    def has_reference(item):
        bounds=_rect(item['bbox'])
        return any(s.score>=.72 and _FIGURE_REFERENCE.search(s.text) and _center_inside(bounds,s)
                   and not figure_numbers(s.text) and not any(_center_inside(f,s) for f in figures)
                   for s in spans)

    def target_safe(number,pair):
        targets=by_number.get(number,[])
        return (not targets or len(targets)==1 and targets[0].get('bbox') and
                (_contains(_rect(targets[0]['bbox']),pair['crop']) or has_reference(targets[0])))

    for item in result:
        # Refinement is for baseline detections, never for teacher-edited crops.
        if item.get('regions') is not None or not item.get('bbox'):
            continue
        try:
            number=int(item['printedNumber'])
        except (ValueError,TypeError):
            continue
        bounds=_rect(item['bbox'])
        own=[s for s,numbers in captions if number in numbers]
        relevant=[(s,numbers) for s,numbers in captions if _center_inside(bounds,s)]
        if len(by_number[number])!=1:
            if own or relevant:
                _warn(item,_REPEATED)
            continue
        foreign=[]
        unresolved=[crop for _,crop in unpaired if _intersects(bounds,crop)]
        uncertain=any(any(n!=number for n in numbers) and _intersects(bounds,crop)
                      for numbers,crop in unpaired)
        for caption,numbers in relevant:
            for other in numbers:
                if other==number:
                    continue
                pair=pairs.get(other)
                if pair is None or not target_safe(other,pair):
                    uncertain=True
                    if pair is not None and pair['crop'] not in unresolved:
                        unresolved.append(pair['crop'])
                elif pair not in foreign:
                    foreign.append(pair)
                    if not by_number.get(other):
                        _warn(item,_ORPHAN)
        # A caption can sit just outside a baseline boundary while its image
        # crosses that boundary; its known foreign pixels still need exclusion.
        for other,pair in pairs.items():
            if other!=number and _intersects(bounds,pair['crop']) and pair not in foreign:
                if not target_safe(other,pair):
                    uncertain=True
                    if pair['crop'] not in unresolved:
                        unresolved.append(pair['crop'])
                else:
                    foreign.append(pair)
                    if not by_number.get(other):
                        _warn(item,_ORPHAN)
        if uncertain:
            _warn(item,_UNCERTAIN)
        # One conflicting caption does not invalidate independent associations.
        # Its complete image-plus-caption crop stays in the host; a proposed
        # removal that overlaps that unresolved evidence must also abstain.
        foreign=[pair for pair in foreign
                 if not any(_intersects(pair['crop'],r) for r in unresolved)]
        remote=None
        if own and not all(_center_inside(bounds,s) for s in own):
            pair=pairs.get(number)
            if pair is None:
                _warn(item,_UNCERTAIN)
            elif not has_reference(item):
                _warn(item,_NO_REFERENCE)
            elif _intersects(bounds,pair['crop']):
                _warn(item,_CROSSING)
            else:
                remote=pair['crop']
        if not foreign and remote is None:
            continue
        anchors=[s for s in spans if s.score>=.72 and question_number(s.text)==number and _center_inside(bounds,s)]
        if not anchors:
            _warn(item,_CROSSING)
            continue
        anchor=min(anchors,key=lambda s:(s.y,s.x))
        pieces=[bounds]
        for pair in foreign:
            pieces=[part for piece in pieces for part in _subtract(piece,pair['crop'])]
        pieces=_merge_adjacent(pieces)
        masks=[p['crop'] for p in foreign]
        protected=[r for r in [_span_rect(s) for s in spans]+figures+unresolved
                   if _contains(bounds,r) and not any(_intersects(r,m) for m in masks)]
        pieces=_keep_whole(pieces,bounds,masks,protected)
        if pieces is None:
            _warn(item,_CROSSING)
            continue
        primary=next((r for r in pieces if _contains(r,_span_rect(anchor))),None)
        if primary is None:
            _warn(item,_CROSSING)
            continue
        pieces=[primary]+[r for r in pieces if r!=primary]
        if remote is not None:
            pieces.append(remote)
        if len(pieces)>8:
            _warn(item,_LIMIT)
            continue
        item['bbox']=_box(pieces[0])
        item['regions']=[_box(r) for r in pieces]
        # Remove only known machine-generated warnings whose geometry this
        # operation actually resolved. Unknown/teacher-authored text survives.
        resolved_numbers={n for _,numbers in relevant for n in numbers if n!=number
            and all(not any(_intersects(r,_span_rect(s)) for r in pieces)
                    for s,ns in captions if n in ns)}
        warnings=[]
        for warning in item['unreadableParts']:
            if remote is not None and warning==_REMOTE_WARNING:
                continue
            match=_FOREIGN_WARNING.fullmatch(warning)
            if match:
                remaining=[n for n in match.group(1).split('、') if int(n) not in resolved_numbers]
                if not remaining:
                    continue
                warning=f'题框内有其他题号的配图（{"、".join(remaining)}），请核对配图归属'
            warnings.append(warning)
        item['unreadableParts']=warnings
    return result
