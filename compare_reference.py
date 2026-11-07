"""Offline comparison with saved Tencent DOM responses, not ground-truth accuracy.

--capture invokes the real worker and records OCR. --replay reuses that evidence
to isolate layout changes. No network. Artifacts include code/input hashes.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageOps
import worker
from layout import LayoutError, split_questions


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def reference_boxes(response, size):
    pages = response['QuestionInfo']
    if len(pages) != 1 or pages[0]['Angle'] != 0:
        raise ValueError('Only single-page unrotated references are supported')
    page = pages[0]
    sx, sy = size[0]/page['Width'], size[1]/page['Height']
    boxes = []
    for index, result in enumerate(page['ResultList'], 1):
        coords = result['Coord']
        if len(coords) != 1:
            raise ValueError('Multi-region reference requires explicit handling')
        points = list(coords[0].values())
        x, y = min(p['X'] for p in points)*sx, min(p['Y'] for p in points)*sy
        right, bottom = max(p['X'] for p in points)*sx, max(p['Y'] for p in points)*sy
        boxes.append({'ordinal': index, 'bbox': {'x': x, 'y': y, 'width': right-x, 'height': bottom-y}})
    return boxes


def overlap(a, b):
    intersection = max(0, min(a['x']+a['width'], b['x']+b['width'])-max(a['x'], b['x'])) * max(0, min(a['y']+a['height'], b['y']+b['height'])-max(a['y'], b['y']))
    aa, ab = a['width']*a['height'], b['width']*b['height']
    return {'iou': intersection/(aa+ab-intersection), 'reference_coverage': intersection/ab, 'extra_area_fraction': 1-intersection/aa}


def match_boxes(local, reference):
    # Explicit greedy one-to-one maximum-IoU pairing, no printed-number guesses.
    candidates = sorted([(overlap(a['bbox'], b['bbox'])['iou'], i, j) for i, a in enumerate(local) for j, b in enumerate(reference)], reverse=True)
    used_local, used_reference, matches = set(), set(), []
    for score, i, j in candidates:
        if score <= 0 or i in used_local or j in used_reference:
            continue
        used_local.add(i); used_reference.add(j)
        matches.append({'local_ordinal': i+1, 'reference_ordinal': j+1, **overlap(local[i]['bbox'], reference[j]['bbox'])})
    return {'pairs': sorted(matches, key=lambda m: m['reference_ordinal']),
            'matched_iou_50': sum(m['iou'] >= .5 for m in matches),
            'unmatched_local': [i+1 for i in range(len(local)) if i not in used_local],
            'unmatched_reference': [j+1 for j in range(len(reference)) if j not in used_reference]}


def overlay(image, blocks, color):
    image = image.copy(); draw = ImageDraw.Draw(image)
    for block in blocks:
        b = block['bbox']; x, y, w, h = (b[k] for k in ('x','y','width','height'))
        draw.rectangle((x, y, x+w-1, y+h-1), outline=color, width=3)
        draw.text((x+4,y+3), str(block['ordinal']), fill=color, stroke_width=1)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture', action='store_true')
    mode.add_argument('--replay', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'results.json').exists():
        raise ValueError('Use a new output folder to preserve prior runs')
    manifest = json.loads((args.references/'manifest.json').read_text(encoding='utf-8-sig'))
    code = Path(__file__).parent
    hashes = {}
    for name in ['worker.py', 'layout.py', 'compare_reference.py']:
        data = (code/name).read_bytes(); hashes[name] = digest(data)
        (args.output/('snapshot-'+name)).write_bytes(data)
    write_json(args.output/'engine-hashes.json', hashes)
    summary = []
    for sample in manifest['samples']:
        sid = sample['id']; source = args.references/f'{sid}.original.png'
        source_hash = digest(source.read_bytes())
        if source_hash != sample['original_sha256']:
            raise ValueError('Original image hash mismatch')
        response_data = (args.references/f'{sid}.response.dom.json').read_bytes()
        response = json.loads(response_data)
        if args.replay:
            data = (args.replay/f'{sid}.input.jpg').read_bytes()
            prior = json.loads((args.replay/f'{sid}.result.json').read_text(encoding='utf-8'))
            if prior['source_sha256'] != source_hash or prior['page']['imageHash'] != digest(data):
                raise ValueError('Replay input mismatch')
            rows = json.loads((args.replay/f'{sid}.ocr.json').read_text(encoding='utf-8'))
        else:
            image = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
            image.thumbnail((2048,2048))
            buf = io.BytesIO(); image.save(buf, format='JPEG', quality=85); data=buf.getvalue()
            while len(data) > 1572864:
                image.thumbnail((round(image.width*.8),round(image.height*.8)))
                buf = io.BytesIO(); image.save(buf, format='JPEG', quality=80); data=buf.getvalue()
        image = Image.open(io.BytesIO(data)).convert('RGB')
        (args.output/f'{sid}.input.jpg').write_bytes(data)
        page_input = {'version':1, 'pageId':'calibration-'+sid, 'imageHash':digest(data), 'width':image.width, 'height':image.height,
                      'images':[{'mime':'image/jpeg','base64':base64.b64encode(data).decode()}]}
        start = time.monotonic()
        if args.replay:
            with patch.object(worker, 'recognize', return_value=rows):
                page = worker.process_page(page_input)
        else:
            recognize = worker.recognize
            def capture(img):
                nonlocal rows
                rows = recognize(img)
                return rows
            with patch.object(worker, 'recognize', capture):
                page = worker.process_page(page_input)
        seconds = round(time.monotonic()-start, 3)
        reason = None
        if not page['blocks']:
            try: split_questions(rows, image.width, image.height)
            except LayoutError as error: reason = str(error)
        refs = reference_boxes(response, image.size)
        result = {'id':sid, 'user_file':sample['userFile'], 'mode':'ocr_replay' if args.replay else 'real_worker',
                  'source_sha256':source_hash, 'response_sha256':digest(response_data), 'seconds':seconds,
                  'reference_count':len(refs), 'local_count':len(page['blocks']), 'abstention_reason':reason,
                  'comparison':match_boxes(page['blocks'], refs), 'reference_boxes':refs, 'page':page}
        write_json(args.output/f'{sid}.result.json', result)
        write_json(args.output/f'{sid}.ocr.json', rows)
        canvas = Image.new('RGB', (image.width*2,image.height+30), 'white')
        canvas.paste(overlay(image, refs, '#0066ee'), (0,30))
        canvas.paste(overlay(image, page['blocks'], '#d00000'), (image.width,30))
        draw = ImageDraw.Draw(canvas)
        draw.text((8,8), 'Tencent reference (not ground truth)', fill='black')
        draw.text((image.width+8,8), f'Local: {len(page["blocks"])}; {reason or "review required"}', fill='black')
        canvas.save(args.output/f'{sid}.comparison.jpg', quality=90)
        summary.append(result); write_json(args.output/'results.json', summary)
        print(json.dumps({k:v for k,v in result.items() if k not in {'page','comparison','reference_boxes'}},ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
