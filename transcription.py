"""Assemble observed OCR only; narrow plain-choice acceptance, otherwise cloud.

Scores are heuristics, not calibrated correctness probabilities. No math repair,
answer inference, handwriting classification or missing-condition reconstruction.
"""
import copy
import re
from layout import clean_spans, question_number, figure_numbers

CHOICE = re.compile(r'(?:^|\s)([A-H])[.．、:：]\s*')
FIGURE = re.compile(r'如图|图中|下图|图示|如表|下表|表格|坐标|电路|图像|图象')
SYMBOL = re.compile(r'[=+×÷√∫∑^<>≤≥±∠△παβγΩ²³₀-₉]|\d\s*[/／]\s*\d|[A-Za-z]\s*\(|函数|方程|化学|向量|投影|分式|对数|三角|摩天轮|零点')
ANSWER = re.compile(r'^\s*(?:解[：:]|答[：:]|答案[：:]|解析[：:]|证明[：:]|故选|所以选)')

def overlap(s,r):
    return max(0,min(s.right,r['x']+r['width'])-max(s.x,r['x']))*max(0,min(s.bottom,r['y']+r['height'])-max(s.y,r['y']))

def assemble_questions(blocks, rows, width, height):
    # Strip OCR is only for finding anchors. Its truncated prefixes/suffixes
    # must never be appended as extra stem/choices or overrule whole lines.
    text_rows=[r for r in rows if r.get('source')!='strip']
    spans=clean_spans(text_rows,width,height)
    out=copy.deepcopy(blocks)
    for b in out:
        regions=b.get('regions') or [b['bbox']]
        owned=[s for s in spans if any(overlap(s,r)>.5*(s.right-s.x)*s.height for r in regions)]
        notes=list(b.get('unreadableParts',[]))
        # Detect clipping using raw quads, before clean_spans clips to image.
        for row in text_rows:
            q=row['quad'];xs=[p[0] for p in q];ys=[p[1] for p in q]
            cx,cy=sum(xs)/4,sum(ys)/4
            if any(r['x']<=cx<=r['x']+r['width'] and r['y']<=cy<=r['y']+r['height'] for r in regions):
                if min(xs)<=1 or max(xs)>=width-1 or min(ys)<=1 or max(ys)>=height-1:
                    notes.append('文字贴近照片边缘，条件可能裁断')
        stem=[];options=[];labels=[];answer_zone=False;previous_option_span=None
        for s in owned:
            text=s.text
            if not text:continue
            # UI tags in photographed question screens are not question content.
            if re.fullmatch(r'[口□☐]?\s*待检查',text):continue
            if re.fullmatch(r'第\s*\d+\s*页(?:\s*共\s*\d+\s*页)?',text):continue
            if s.score<.96:notes.append('本地文字置信度不足，需云端核对')
            if not any(overlap(s,r)>=.98*(s.right-s.x)*s.height for r in regions):
                notes.append('文字跨越题框，需核对完整边界')
            n=question_number(text)
            if n is not None and str(n)!=b.get('printedNumber'):
                notes.append('题框内存在其他题号，需核对是否串题')
            if ANSWER.match(text):answer_zone=True
            if re.fullmatch(r'[A-Ha-h✓✔√×✗✘]',text):
                notes.append('存在作答或解析文字，本地无法可靠分离');continue
            if answer_zone:
                notes.append('存在作答或解析文字，本地无法可靠分离')
                continue
            if figure_numbers(text):
                notes.append('存在配图图注，需核对图文归属');continue
            matches=list(CHOICE.finditer(text))
            if matches:
                prefix=text[:matches[0].start()].strip()
                if prefix:
                    if options:options[-1]+='\n'+prefix
                    else:stem.append(prefix)
                for i,m in enumerate(matches):
                    labels.append(m[1]);options.append(text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)].strip())
                previous_option_span=s
            elif options:
                if previous_option_span is None or s.y-previous_option_span.bottom>1.2*s.height or abs(s.x-previous_option_span.x)>2*s.height:
                    notes.append('存在作答或解析文字，本地无法可靠分离')
                    continue
                options[-1]+='\n'+text;previous_option_span=s
            else:stem.append(text)
        b['questionText']='\n'.join(stem)[:12000]
        b['options']=options[:8]
        b['studentAnswer']='' # Do not guess that dark marks are student writing.
        b['questionType']='选择题' if options else ''
        full='\n'.join(stem+options)
        # NFKC may flatten x²/H₂O. Do not treat resulting x2/H2O as safe text.
        # Only plain Chinese choices are accepted in this initial local tier.
        plain_stem=re.sub(r'^\s*(?:例\s*\d+\s*[.、:]?|第\s*\d+\s*题|\d+\s*[.、:])','', '\n'.join(stem))
        plain_stem=re.sub(r'\(\s*\d+\s*分\s*\)','',plain_stem)
        plain='\n'.join([plain_stem]+[CHOICE.sub('',o,count=1) for o in options])
        has_figure=bool(FIGURE.search(full)) or len(regions)>1 or any('配图' in n or '图注' in n for n in notes)
        b['hasFigure']=has_figure
        if has_figure:notes.append('图形或表格条件需云端核对')
        if SYMBOL.search(full) or re.search(r'[A-Za-z0-9−\-/%℃°]',plain):notes.append('数学或科学符号需云端核对')
        if re.search(r'上文|下文|上述|以下材料|下列材料|阅读|短文|根据材料|根据以上|据此|接上',full):
            notes.append('共享材料或上下文需云端核对')
        if re.search(r'[（(]\s*[A-H]\s*[)）]|_{2,}|…{2,}',full):
            notes.append('填空或作答与题干混排，需云端核对')
        if not stem or len(''.join(stem))<10:notes.append('本地题干不完整')
        if not stem or question_number(stem[0]) is None or str(question_number(stem[0]))!=b.get('printedNumber'):
            notes.append('本地题干不完整')
        if labels!=list('ABCD') or len(options)!=4 or any(not CHOICE.sub('',o,count=1).strip() for o in options):
            notes.append('本地仅可确认完整四选项文字题，其他题型需云端核对')
        if len(full)>12000:notes.append('本地文字超出长度限制')
        resolved_by_ocr={'本地文字置信度不足，需云端核对','图形或表格条件需云端核对','数学或科学符号需云端核对',
                         '填空或作答与题干混排，需云端核对','本地题干不完整','本地仅可确认完整四选项文字题，其他题型需云端核对',
                         '本地文字超出长度限制','存在作答或解析文字，本地无法可靠分离','共享材料或上下文需云端核对'}
        b['unreadableParts']=list(dict.fromkeys('本地转写待补齐：'+n if n in resolved_by_ocr else n for n in notes))[:100]
    return out
