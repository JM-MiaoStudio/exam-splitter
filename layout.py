"""Conservative question grouping; no cloud calls or answer reconstruction.

Only the geometry is a product result. OCR text is evidence for finding question
starts, not an accurate transcription of mathematical notation or handwriting.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
from statistics import median
import unicodedata


class LayoutError(ValueError):
    pass


_NUMBER = re.compile(r'^(?:(例)\s*(\d{1,3})\s*[.、:：]?|第\s*(\d{1,3})\s*题\s*[.、:：]?|(\d{1,3})\s*[.、:：])\s*(.*)$')
_HEADING = re.compile(r'^[一二三四五六七八九十]+[、.]\s*\S')
_FIGURE_CAPTION = re.compile(r'\(?\s*第[\s—_-]*(\d{1,3})[\s—_-]*题\s*图\s*\)?(?:\s*\([^)]*\))?')


def figure_numbers(text):
    text = unicodedata.normalize('NFKC', text).strip()
    matches = list(_FIGURE_CAPTION.finditer(text))
    # OCR sometimes combines adjacent captions on a shared line. Accept only
    # a line entirely composed of captions/punctuation, never "图中求...".
    if _FIGURE_CAPTION.sub('', text).strip(' \t\r\n.。,，;；、—_-'):
        return []
    return [int(m[1]) for m in matches if 1 <= int(m[1]) <= 200]


def number_text(text):
    text = unicodedata.normalize('NFKC', text).strip()
    # A photographed ruled border may become a leading ! or |. Remove only
    # these separators before an explicit observed number+delimiter.
    return re.sub(r'^[!|]+\s*(?=\d{1,3}\s*[.、:：])','',text)


def question_number(text):
    text = number_text(text)
    if figure_numbers(text):
        return None
    match = _NUMBER.match(text)
    if not match:
        return None
    number = int(match[2] or match[3] or match[4])
    rest = match[5].strip()
    # Decimal/table values are not question starts, but Chinese stems can begin
    # with a year, quantity or expression. Chinese punctuation and whitespace
    # after a dot provide additional evidence of an actual question delimiter.
    if not 1 <= number <= 200:
        return None
    chinese = len(re.findall(r'[\u4e00-\u9fff]',rest))
    explicit = bool(match[1] or match[3] or re.match(r'^\d{1,3}\s*(?:[、:：]|\.\s+)',text))
    if rest and rest[0].isdigit() and not explicit and chinese < 4:
        return None
    if rest and not chinese and not re.search(r'[A-Za-z]',rest) and not (explicit and len(rest)>=3):
        return None
    if rest and re.match(r'[a-zA-Z]\s*=', rest) and not chinese:
        return None
    return number


@dataclass(frozen=True)
class Span:
    text: str
    score: float
    x: float
    y: float
    right: float
    bottom: float

    @property
    def height(self):
        return self.bottom - self.y


def clean_spans(rows, width, height):
    if not isinstance(rows, list) or len(rows) > 5000:
        raise LayoutError('INVALID_OCR_GEOMETRY')
    spans = []
    for row in rows:
        quad = row['quad']
        if len(quad) != 4 or any(len(p) != 2 or any(not math.isfinite(v) for v in p) for p in quad):
            raise LayoutError('INVALID_OCR_GEOMETRY')
        x, right = max(0, min(p[0] for p in quad)), min(width, max(p[0] for p in quad))
        y, bottom = max(0, min(p[1] for p in quad)), min(height, max(p[1] for p in quad))
        score = float(row['score'])
        if right <= x or bottom <= y or not math.isfinite(score):
            continue
        span = Span(unicodedata.normalize('NFKC', row['text']).strip(), score, x, y, right, bottom)
        # Tile overlap may duplicate a text line with a different transcription.
        duplicate = next((i for i, s in enumerate(spans)
                          if min(right,s.right)-max(x,s.x) > .75*min(right-x,s.right-s.x)
                          and min(right-x,s.right-s.x) > .7*max(right-x,s.right-s.x)
                          and min(bottom,s.bottom)-max(y,s.y) > .65*min(bottom-y,s.height)), None)
        if duplicate is None:
            spans.append(span)
        elif score > spans[duplicate].score:
            spans[duplicate] = span
    return sorted(spans, key=lambda s: (s.y, s.x))


def question_starts(rows,width,height):
    """Geometry anchors are independent of lossy text deduplication.

    Keep observed alternatives until same-ink detections have been grouped.
    An OCR disagreement can choose ONLY an observed number, supported by two
    unambiguous consecutive neighbours. Otherwise keep geometry, number=None.
    Returns (span, number, conflicting_numbers); never changes raw OCR text.
    """
    clean_spans(rows,width,height)  # validate the complete input and row limit
    raw=[];ink={}
    for r in rows:
        for s in clean_spans([r],width,height):
            raw.append(s);q=r['quad']
            ink[id(s)]=(q[0],q[1],q[2],q[3])
    captions=[s for s in raw if s.score>=.72 and figure_numbers(s.text)]
    candidates=[]
    for s in raw:
        n=question_number(s.text)
        if n is None or s.score<.72:continue
        if any(n in figure_numbers(c.text) and abs(s.x-c.x)<s.height and
               min(s.bottom,c.bottom)-max(s.y,c.y)>.7*min(s.height,c.height) for c in captions):continue
        match=_NUMBER.match(number_text(s.text))
        if not match[5].strip() and not any(0<=other.x-s.right<=3*s.height and
             abs(other.y-s.y)<=.5*s.height and len(other.text)>=3 and other.score>=.72 for other in raw):continue
        candidates.append((s,n))
    groups=[]
    def same_line(a,b):
        overlap=min(a.bottom,b.bottom)-max(a.y,b.y)
        if abs(a.x-b.x)>max(a.height,b.height)*1.5 or overlap<.6*min(a.height,b.height):return False
        # Compare ink baselines at the same x, not the tall axis-aligned
        # envelopes of skewed lines; adjacent lines can overlap as rectangles.
        def center(s,x):
            q=ink[id(s)];lx=(q[0][0]+q[3][0])/2;rx=(q[1][0]+q[2][0])/2
            ly=(q[0][1]+q[3][1])/2;ry=(q[1][1]+q[2][1])/2
            return ly+(ry-ly)*(x-lx)/max(1,rx-lx),max(1,abs(q[3][1]-q[0][1]),abs(q[2][1]-q[1][1]))
        ay,ah=center(a,max(a.x,b.x));by,bh=center(b,max(a.x,b.x))
        if abs(ay-by)>.6*max(ah,bh):return False
        # Matching starts also need shared body text, unless one is a separate
        # number token. This avoids merging two different tilted adjacent lines.
        def body(s):
            m=_NUMBER.match(number_text(s.text))
            return re.sub(r'[^A-Za-z\u4e00-\u9fff]','',m[5]).lower()
        x,y=body(a),body(b)
        length=min(len(x),len(y),12)
        return not x or not y or x[:4]==y[:4] or length>=3 and SequenceMatcher(None,x[:length],y[:length]).ratio()>=.65
    for s,n in sorted(candidates,key=lambda a:(a[0].y,a[0].x)):
        target=next((g for g in groups if all(same_line(s,old) for old,_ in g)),None)
        if target is None:groups.append([(s,n)])
        else:target.append((s,n))
    entries=[]
    for g in groups:
        nums={n for _,n in g}
        widest=max((s for s,_ in g),key=lambda s:s.right-s.x)
        # Full envelope retains printed ink when strip OCR has a tighter quad.
        envelope=Span(widest.text,max(s.score for s,_ in g),min(s.x for s,_ in g),
                      min(s.y for s,_ in g),max(s.right for s,_ in g),max(s.bottom for s,_ in g))
        entries.append((envelope,nums))
    output=[]
    for s,nums in entries:
        chosen=next(iter(nums)) if len(nums)==1 else None
        if len(nums)>1:
            column=sorted([(a,ns) for a,ns in entries if abs(a.x-s.x)<=width*.08],key=lambda v:v[0].y)
            pos=next(i for i,(a,_) in enumerate(column) if a is s)
            supported=[]
            for n in nums:
                support=0
                for direction in [-1,1]:
                    for step in range(1,9):
                        j=pos+direction*step
                        if not 0<=j<len(column) or n+direction*step not in column[j][1]:break
                        if len(column[j][1])==1:support+=1
                if support>=2:supported.append(n)
            if len(supported)==1:chosen=supported[0]
        output.append((s,chosen,sorted(nums) if len(nums)>1 else []))
    return output


def question_top(anchor, spans, right):
    # A fraction numerator/superscript can be a separate OCR span just after
    # the text ending in "y=". It belongs to this starting line despite having
    # a higher top edge. Require horizontal adjacency and vertical overlap;
    # do not absorb an unrelated previous diagram based on proximity alone.
    # OCR can omit the equals sign between a function variable and a fraction.
    # Keep the same strict geometry test; do not expand arbitrary prose starts.
    if not (anchor.text.rstrip().endswith('=') or
            re.search(r'函数\s*[yY]\s*$',anchor.text)):
        return max(0, math.floor(anchor.y)-2)
    companions = [s.y for s in spans if s.score >= .72 and s.right <= right
                  and 0 <= s.x-anchor.right <= 2*anchor.height
                  and anchor.y-anchor.height*.85 <= s.y < anchor.y
                  and anchor.y+anchor.height*.25 <= s.bottom <= anchor.bottom
                  and not re.search(r'[\u4e00-\u9fff]', s.text)
                  and re.search(r'[A-Za-z0-9]', s.text)]
    return max(0, math.floor(min([anchor.y]+companions))-2)


def split_questions(rows, width, height):
    if not isinstance(width, int) or not isinstance(height, int) or not 1 <= width <= 2048 or not 1 <= height <= 2048:
        raise LayoutError('INVALID_IMAGE_SIZE')
    spans = clean_spans(rows, width, height)
    starts_with_evidence=question_starts(rows,width,height)
    anchors=[(s,n) for s,n,_ in starts_with_evidence]
    conflicts={id(s):ns for s,_,ns in starts_with_evidence if ns}
    if not anchors:
        raise LayoutError('NO_QUESTION_ANCHORS')
    typical_height = median(a.height for a, _ in anchors)
    # Cluster question starts, not arbitrary text blocks (tables have many cells).
    clusters = []
    for item in sorted(anchors, key=lambda a: a[0].x):
        if not clusters or item[0].x-median(a.x for a,_ in clusters[-1]) > max(width*.08, typical_height*1.5):
            clusters.append([item])
        else:
            clusters[-1].append(item)
    # Indented numbered sentences within a stem are not extra page columns.
    columns = [clusters[0]]
    for cluster in clusters[1:]:
        if median(a.x for a,_ in cluster)-median(a.x for a,_ in columns[-1]) >= width*.30:
            columns.append(cluster)
        else:
            # A genuine main question may be indented beside a figure. Keep a
            # unique sequential bridge; repeated indented numbers are nested
            # markers. Unknown layouts must abstain instead of silently losing
            # a question in the previous crop.
            for anchor,number in sorted(cluster,key=lambda a:a[0].y):
                existing=columns[-1]
                if any(n==number for _,n in existing):
                    continue
                before=sorted([(a,n) for a,n in existing if a.y<anchor.y],key=lambda a:a[0].y)
                after=sorted([(a,n) for a,n in existing if a.y>anchor.y],key=lambda a:a[0].y)
                if before and after and before[-1][1] is not None and after[0][1] is not None and before[-1][1]+1==number and after[0][1]-1==number:
                    existing.append((anchor,number))
                else:
                    raise LayoutError('AMBIGUOUS_COLUMNS')
    if len(columns) > 3:
        raise LayoutError('AMBIGUOUS_COLUMNS')
    boundaries = [0] + [max(1, round(min(a.x for a,_ in col)-typical_height*.8)) for col in columns[1:]] + [width]
    blocks = []
    for index, column in enumerate(columns):
        left, right = boundaries[index:index+2]
        column = sorted(column, key=lambda a: (a[0].y, a[0].x))
        first_y = column[0][0].y
        # Do not slice a full-width stem/formula across a presumed column gutter.
        if any(s.x < right-typical_height and s.right > right+typical_height*.3
               and s.x >= left and s.y >= first_y for s in spans) and index < len(columns)-1:
            raise LayoutError('AMBIGUOUS_COLUMNS')
        starts = [question_top(anchor, spans, right) for anchor,_ in column]
        for n, (anchor, number) in enumerate(column):
            # OCR quads already surround the ink. A fraction of a tall skewed
            # line's bounding height would pull in the previous question's figure.
            top = starts[n]
            bottom = height if n+1 == len(column) else starts[n+1]
            # A section heading belongs to neither neighbouring question.
            headings = [s.y for s in spans if left <= s.x < right and s.y > anchor.bottom and s.y < bottom
                        and (_HEADING.match(s.text) or re.match(r'^第\s*\d+\s*页',s.text))]
            if headings:
                bottom = max(top+1, math.floor(min(headings)-3))
            if bottom <= anchor.bottom:
                raise LayoutError('AMBIGUOUS_QUESTION_STARTS')
            ordinal = len(blocks)+1
            notes=[]
            if id(anchor) in conflicts:
                notes.append('同一行题号识别冲突（'+ '、'.join(map(str,conflicts[id(anchor)])) +'），'+('已按相邻实测题号暂定，请核对' if number is not None else '未确定题号，请核对'))
            blocks.append({'id': f'b{ordinal}', 'ordinal': ordinal, 'printedNumber': str(number) if number is not None else None,
                           'bbox': {'x':left,'y':top,'width':right-left,'height':bottom-top},
                           'questionText':'','options':[],'studentAnswer':'','subject':'','questionType':'',
                           'unreadableParts':notes, 'hasFigure':True})
    if len(blocks) > 50:
        raise LayoutError('TOO_MANY_QUESTIONS')
    # A single rectangular crop cannot include a remote diagram without also
    # swallowing unrelated questions. Keep the detected stem, flag the evidence
    # for teacher review, and never promote figure labels to invented questions.
    captions = [(s, number) for s in spans if s.score >= .72 for number in figure_numbers(s.text)]
    for block in blocks:
        if block['printedNumber'] is None:continue
        box = block['bbox']; number = int(block['printedNumber'])
        def inside(span):
            return (box['x'] <= (span.x+span.right)/2 < box['x']+box['width']
                    and box['y'] <= (span.y+span.bottom)/2 < box['y']+box['height'])
        foreign = sorted({n for s,n in captions if n != number and inside(s)})
        if foreign:
            block['unreadableParts'].append('题框内有其他题号的配图（'+ '、'.join(map(str,foreign)) +'），请核对配图归属')
        own = [s for s,n in captions if n == number]
        unique_number = sum(b['printedNumber'] == block['printedNumber'] for b in blocks) == 1
        if unique_number and own and not any(inside(s) for s in own):
            block['unreadableParts'].append('发现同题号图注位于题框外，请核对并补充配图')
    return blocks
