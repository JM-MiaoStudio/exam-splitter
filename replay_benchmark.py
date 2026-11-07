"""Offline same-image OCR ablation; historical cloud inputs are reconstructed.

No cloud client. Inputs/outputs are explicit CLI paths. Do not interpret counts
or OCR agreement as accuracy: inspect overlays and use independent annotations.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from PIL import Image, ImageDraw, ImageOps
from cloud_geometry import cloud_rows
from preprocess import flatten_background
import worker


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda o:o.tolist()),encoding='utf-8')


def prepare_case_folder(folder):
    # Deliberately no inference cache: changed weights/preprocessing require a
    # fresh run directory. Never overwrite a previous run's retained inputs.
    if folder.exists() and any(folder.iterdir()):raise ValueError('OUTPUT_ALREADY_EXISTS')
    folder.mkdir(parents=True,exist_ok=True)


def group_route(page,image,rows,boxes):
    return {k:page[k] for k in ('version','pageId','imageHash','width','height')} | {
        'coordinateSpace':'uploaded-image','blocks':worker.group_rows(image,rows,boxes)}


def overlay(image,rows,blocks,destination):
    im=image.copy();draw=ImageDraw.Draw(im)
    for row in rows:draw.polygon([tuple(p) for p in row['quad']],outline='#4999dd',width=1)
    for block in blocks:
        for box in block.get('regions',[block['bbox']]):
            x,y,w,h=[box[k] for k in ('x','y','width','height')]
            draw.rectangle((x,y,x+w-1,y+h-1),outline='red',width=3)
        draw.text((block['bbox']['x']+3,block['bbox']['y']+3),str(block['printedNumber']),fill='red')
    im.save(destination,quality=90)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deepseek',type=Path,required=True)
    ap.add_argument('--ours',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--layout-model',required=True);ap.add_argument('--only',default='')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # Deny actual network connections, including accidental model download.
    sys.addaudithook(lambda event,values: (_ for _ in ()).throw(RuntimeError('NETWORK_DISABLED')) if event=='socket.connect' else None)
    cases=[(n,args.deepseek/'输入样本'/sub/f,args.deepseek/'验证记录'/f'{n}_cloud.json') for n,sub,f in [
        ('p1','real_exam','p1_physics_water_heater.jpg'),('p2','real_exam','p2_chem_metal.jpg'),
        ('p3','real_exam','p3_chem_alloy.jpg'),('c1','curved_book','c1_math_grade3_curl.jpg'),('c2','curved_book','c2_math_book_spread.jpg')]]
    cases += [(f'ours-{i:02}',args.ours/f'{i:02}.input.jpg',None) for i in range(1,7)]
    summaries=json.loads((args.output/'summary.json').read_text(encoding='utf-8')) if (args.output/'summary.json').exists() else []
    for sid,path,response_path in cases:
        if args.only and sid not in args.only.split(','):continue
        folder=args.output/sid;prepare_case_folder(folder)
        original=ImageOps.exif_transpose(Image.open(path)).convert('RGB');raw_size=original.size
        original.thumbnail((2048,2048))
        if path.suffix.lower() in ('.jpg','.jpeg') and raw_size==original.size and Image.open(path).getexif().get(274,1)==1:
            data=path.read_bytes()
        else:
            buf=io.BytesIO();original.save(buf,format='JPEG',quality=90);data=buf.getvalue()
        original=Image.open(io.BytesIO(data)).convert('RGB');original.save(folder/'input.jpg',quality=95)
        # Exact request bytes are retained separately from display exports.
        (folder/'upload.jpg').write_bytes(data)
        digest=hashlib.sha256(data).hexdigest()
        page={'version':1,'pageId':sid,'imageHash':digest,'width':original.width,'height':original.height,
              'images':[{'mime':'image/jpeg','base64':base64.b64encode(data).decode()}]}
        record={'id':sid,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'upload_sha256':digest,'original_size':raw_size,'upload_size':original.size,'routes':{},
                'layout_model_dir':args.layout_model,
                'code_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ('worker.py','cloud_geometry.py','preprocess.py','assistance.py','layout.py','dense_english.py','edge_stems.py','layout_regions.py')}}
        boxes=worker.detect_layout(original,args.layout_model);save(folder/'layout.json',boxes)
        for route in ('local','enhanced','cloud'):
            if route=='cloud' and response_path is None:continue
            start=time.monotonic();image=flatten_background(original) if route=='enhanced' else original
            if route=='enhanced':image.save(folder/'enhanced.jpg',quality=95)
            if route=='cloud':
                # Legacy CLI documented raw image -> INTER_AREA, max side 2000.
                # No captured request hash is available: label reconstruction.
                scale=min(1,2000/max(raw_size));sent=[int(v*scale) for v in raw_size]
                response=json.loads(response_path.read_text(encoding='utf-8-sig'))
                envelope={'version':1,'polygonSpace':'sent-image','imageHash':digest,'originalSize':list(original.size),
                    'processedSize':list(raw_size),'sentSize':sent,
                    'originalToProcessed':[[raw_size[0]/original.width,0,0],[0,raw_size[1]/original.height,0],[0,0,1]],
                    'response':response}
                save(folder/'cloud-envelope.json',envelope)
                record['cloud_provenance']={'status':'reconstructed-unverified-request','response_sha256':hashlib.sha256(response_path.read_bytes()).hexdigest(),
                    'sent_size':sent,'assumption':'uncropped original, max_side=2000; historical upload bytes unavailable',
                    'request_id':response.get('RequestId'),'angle':response.get('Angle',response.get('Angel'))}
                try:rows=cloud_rows(envelope,image_hash=digest,width=original.width,height=original.height)
                except ValueError as e:
                    record['routes'][route]={'error':str(e)};continue
            else:rows=worker.recognize(image)
            ocr_seconds=time.monotonic()-start
            result=group_route(page,image,rows,boxes)
            blocks=result['blocks'];save(folder/f'{route}.rows.json',rows);save(folder/f'{route}.page.json',result)
            overlay(original,rows,blocks,folder/f'{route}.overlay.jpg')
            record['routes'][route]={'rows':len(rows),'blocks':len(blocks),'numbers':[b['printedNumber'] for b in blocks],
                'review_blocks':sum(bool(b['unreadableParts']) for b in blocks),'ocr_seconds':round(ocr_seconds,3),'ocr_cached':False,
                'seconds':round(time.monotonic()-start,3)}
            print(json.dumps({'id':sid,'route':route,**record['routes'][route]},ensure_ascii=True),flush=True)
        save(folder/'summary.json',record);summaries=[r for r in summaries if r['id']!=sid]+[record]
        save(args.output/'summary.json',summaries)


if __name__=='__main__':main()
