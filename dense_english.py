"""Recover dense, skewed English worksheets without transcribing answers.

The fallback is deliberately narrow: one complete, observed 1..N English
sequence, at least six starts, modest skew and dense spacing. OCR on an adaptive
number corridor can recover missed tokens. All returned geometry is in the
original image; overlapping envelopes are explicitly marked for review.
"""
import math
import re
from statistics import median
import unicodedata

from layout import clean_spans, question_number

_WARNING = '原图文字行倾斜，矩形题框可能含相邻题目边缘，请核对题框'


def _english(text):
    return len(re.findall(r'[A-Za-z]', text)) >= 8 and not re.search(r'[\u4e00-\u9fff]', text)


def _anchors(rows):
    return [(r, question_number(r['text'])) for r in rows
            if r['score'] >= .80 and question_number(r['text']) is not None and _english(r['text'])]


def _recognize_strip(image):
    import numpy as np
    from ocr_runtime import ocr_engine
    from performance import stage
    engine = ocr_engine()
    with stage('ocr.recovery_crop'):
        result, _ = engine(np.asarray(image)[:, :, ::-1].copy())
    return [{'quad': [[float(x), float(y)] for x, y in q], 'text': t, 'score': float(s)}
            for q, t, s in result or []]


def split_dense_english(image, rows, *, recognize_strip=None):
    """Return original-coordinate blocks, or None when evidence is insufficient.

    ``recognize_strip`` accepts a PIL crop and returns local-coordinate OCR rows;
    optional injection permits tests and reuse of an already initialized engine.
    """
    width, height = image.size
    clean_spans(rows, width, height)  # same geometry validation as primary path
    anchors = _anchors(rows)
    if len({n for _, n in anchors}) < 6:
        return None
    line_height = median(abs(r['quad'][3][1]-r['quad'][0][1]) for r, _ in anchors)
    if line_height < 5:
        return None
    xs = [min(p[0] for p in r['quad']) for r, _ in anchors]
    if max(xs)-min(xs) > width*.20:
        return None
    # Short lines retain the number DB can miss on a long slanted sentence.
    left = max(0, math.floor(min(xs)-line_height*.8))
    right = min(width, math.ceil(max(xs)+line_height*6))
    recovered = (recognize_strip or _recognize_strip)(image.crop((left, 0, right, height)))
    mapped = [dict(r, quad=[[x+left, y] for x, y in r['quad']]) for r in recovered]
    clean_spans(mapped, width, height)
    evidence = rows + mapped
    anchors = _anchors(evidence)
    by_number = {}
    for row, number in anchors:
        previous = by_number.get(number)
        if previous:
            a, b = previous['quad'][0], row['quad'][0]
            if abs(a[0]-b[0]) > line_height*2 or abs(a[1]-b[1]) > line_height:
                return None
        if previous is None or row['score'] > previous['score']:
            by_number[number] = row
    numbers = sorted(by_number)
    if numbers != list(range(1, len(numbers)+1)) or len(numbers) > 50:
        return None
    declared = [int(n) for row in rows for n in re.findall(
        r'本题共\s*(\d{1,2})\s*小题', unicodedata.normalize('NFKC', row['text']))]
    if declared and any(n != len(numbers) for n in declared):
        return None
    ordered = [by_number[n] for n in numbers]
    slopes = [(r['quad'][1][1]-r['quad'][0][1])/(r['quad'][1][0]-r['quad'][0][0])
              for r in ordered if r['quad'][1][0]-r['quad'][0][0] > line_height*3]
    if not slopes:
        return None
    slope = median(slopes)
    if abs(slope) > .12:
        return None
    def corrected(row):
        return [y-slope*x for x, y in row['quad']]
    starts = [min(corrected(r))-3 for r in ordered]
    gaps = [b-a for a, b in zip(starts, starts[1:])]
    if min(gaps) <= line_height*.65 or median(gaps) > line_height*3:
        return None
    # Define the text column from English evidence, excluding adjacent book
    # pages and Chinese footer/QR captions. Low-confidence handwriting inside
    # this column is retained geometrically, never corrected or transcribed.
    column_left = max(0, math.floor(min(min(p[0] for p in r['quad']) for r in ordered)-line_height*.8))
    english = [r for r in evidence if _english(r['text']) and
               min(p[0] for p in r['quad']) >= column_left and
               starts[0] <= median(corrected(r))]
    column_right = min(width, math.ceil(max(max(p[0] for p in r['quad']) for r in english)+line_height*.8))
    # No reliable final boundary follows the last numbered anchor. Keep the
    # entire remaining column, including unknown handwriting, rather than
    # guessing a fixed number of answer lines.
    end = max(height-slope*x for x in (0, width))+1
    boundaries = starts + [end]
    groups = [[] for _ in ordered]
    def rect(row):
        return (min(p[0] for p in row['quad']), min(p[1] for p in row['quad']),
                max(p[0] for p in row['quad']), max(p[1] for p in row['quad']))
    rectangles = [rect(r) for r in evidence]
    # Whole-page OCR and corridor OCR repeat partial sentences. Prefer the
    # encompassing detection so a short tail cannot be assigned independently
    # to the next question under perspective curvature. Its pixels remain in
    # the complete detection's envelope.
    unique = []
    tolerance = line_height*.15
    for row, a in zip(evidence, rectangles):
        contained = any(b[2]-b[0] > (a[2]-a[0])*1.25 and
                        b[0]-tolerance <= a[0] and b[1]-tolerance <= a[1] and
                        b[2]+tolerance >= a[2] and b[3]+tolerance >= a[3]
                        for b in rectangles)
        if not contained:
            unique.append(row)
    for row in unique:
        cx = sum(p[0] for p in row['quad'])/4
        cy = sum(corrected(row))/4
        if not column_left <= cx <= column_right:
            continue
        for i in range(len(ordered)):
            if boundaries[i] <= cy < boundaries[i+1]:
                groups[i].append(row)
                break
    blocks = []
    for i, anchor in enumerate(ordered):
        group = groups[i] + [anchor]
        # Retain the entire original quad envelope, including low-confidence
        # handwriting. Column-wide padding retains undetected ink between
        # fragments. A global shear is only for grouping: page curvature makes
        # extrapolating that shear to crop edges overly broad.
        padding = max(3, line_height*.15)
        y0 = max(0, math.floor(min(p[1] for r in group for p in r['quad'])-padding))
        y1 = (height if i == len(ordered)-1 else
              min(height, math.ceil(max(p[1] for r in group for p in r['quad'])+padding)))
        x0 = max(0, min(column_left, math.floor(min(p[0] for r in group for p in r['quad'])-2)))
        x1 = min(width, max(column_right, math.ceil(max(p[0] for r in group for p in r['quad'])+2)))
        blocks.append({'id':f'b{i+1}', 'ordinal':i+1, 'printedNumber':str(numbers[i]),
                       'bbox':{'x':x0,'y':y0,'width':x1-x0,'height':y1-y0},
                       'questionText':'','options':[],'studentAnswer':'','subject':'','questionType':'',
                       'unreadableParts':[], 'hasFigure':True})
    for i, block in enumerate(blocks):
        b = block['bbox']
        # Consumers trim legacy bbox overlaps; an explicit region preserves
        # this deliberate original-pixel envelope and its review warning.
        block['regions'] = [dict(b)]
        if i == len(blocks)-1:
            block['unreadableParts'].append('末题保留至页底，可能包含页脚，请核对题框')
        if any(j != i and b['y'] < other['bbox']['y']+other['bbox']['height'] and
               b['y']+b['height'] > other['bbox']['y'] for j, other in enumerate(blocks)):
            block['unreadableParts'].append(_WARNING)
    return blocks

