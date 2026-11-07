"""Explicit local UVDoc/PP-DocLayout ablation. Corrected pixels are NOT originals.

Use the separate Paddle environment. --models must contain explicitly fetched
weights; inference never invokes the setup downloader. Outputs are private.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['layout','unwarp'],required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    name='UVDoc' if args.mode=='unwarp' else 'PP-DocLayout-L'
    directory=args.models/(name+'_infer')
    if not (directory/'inference.pdiparams').is_file(): raise ValueError('Run explicit model setup first')
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    os.environ['PADDLE_PDX_CACHE_HOME']=str(args.models.resolve()/'runtime-cache')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HOME']=str(args.models.resolve()/'hf-cache')
    from paddleocr import LayoutDetection,TextImageUnwarping
    from PIL import Image,ImageDraw
    model_class=TextImageUnwarping if args.mode=='unwarp' else LayoutDetection
    started=time.monotonic()
    model=model_class(model_name=name,model_dir=str(directory.resolve()),device='cpu',enable_mkldnn=False,cpu_threads=2)
    init_seconds=time.monotonic()-started
    summary=[]
    for sid in ['01','02','03','04','05','06']:
        source=args.images/(sid+'.input.jpg')
        started=time.monotonic();result=next(iter(model.predict(str(source))))
        seconds=round(time.monotonic()-started,3)
        record={'id':sid,'model':name,'seconds':seconds,'initialization_seconds':round(init_seconds,3),
                'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        if args.mode=='unwarp':
            # Paddle's public UVDoc interface returns a corrected BGR image,
            # not a reverse displacement grid. Store it as a new image space.
            image=Image.fromarray(result['doctr_img'][:,:,::-1]);image.thumbnail((2048,2048))
            image.save(args.output/(sid+'.input.jpg'),quality=85)
            record.update({'coordinateSpace':'uvdoc-corrected-image','original_mapping_available':False,'size':image.size})
        else:
            record.update({'coordinateSpace':'uploaded-image','boxes':result['boxes']})
            image=Image.open(source).convert('RGB');draw=ImageDraw.Draw(image)
            for b in result['boxes']:
                x,y,r,bt=b['coordinate'];draw.rectangle((x,y,r,bt),outline='red',width=3)
                draw.text((x+3,y+3),b['label']+' '+str(round(b['score'],2)),fill='red')
            image.save(args.output/(sid+'.layout.jpg'),quality=90)
        encoded=json.dumps(record,ensure_ascii=False,indent=2,default=lambda o:o.tolist())
        (args.output/(sid+'.json')).write_text(encoded,encoding='utf-8');summary.append(record)
        (args.output/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda o:o.tolist()),encoding='utf-8')
        print(json.dumps({'id':sid,'mode':args.mode,'seconds':seconds,'boxes':len(record.get('boxes',[]))}),flush=True)


if __name__=='__main__':main()
