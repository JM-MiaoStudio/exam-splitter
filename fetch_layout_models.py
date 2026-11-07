"""Explicit setup-only official weight download; never called by inference."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

MODELS = {
    'UVDoc': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar',
    'PP-DocLayout-L': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar',
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,url in MODELS.items():
        archive=args.directory/(name+'.tar')
        if not archive.exists():
            with urllib.request.urlopen(url,timeout=120) as response,archive.with_suffix('.download').open('wb') as out:
                while chunk:=response.read(1024*1024): out.write(chunk)
            archive.with_suffix('.download').replace(archive)
        with tarfile.open(archive) as tar: tar.extractall(args.directory,filter='data')
        model_dir=args.directory/(name+'_infer')
        if not (model_dir/'inference.pdiparams').is_file(): raise ValueError('Model archive missing expected weights')
        records.append({'name':name,'source':url,'license_page':f'https://huggingface.co/PaddlePaddle/{name}',
                        'directory':str(model_dir.resolve()),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.iterdir() if p.is_file()}})
        print(name+' downloaded and recorded',flush=True)
    (args.directory/'model-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
