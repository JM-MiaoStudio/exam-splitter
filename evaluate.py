"""Local acceptance tool; sample images and artifacts stay in the chosen folder.

python evaluate.py --images DIR --output DIR [--synthetic-font FONT]
Writes input JPEGs resized exactly once, full pixel results and labelled boxes.
Not a benchmark accuracy claim: callers supply and inspect representative pages.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps
from worker import process_page
from layout import LayoutError


def synthetic(font_path, output):
    font=ImageFont.truetype(str(font_path),22)
    image=Image.new('RGB',(1040,1440),'white');draw=ImageDraw.Draw(image)
    draw.text((40,25),'中国试卷切题验证样张（合成）',font=font,fill='black')
    for number,x,y in [(1,40,100),(2,40,720),(3,560,100),(4,560,720)]:
        draw.text((x,y),f'{number}. 如图所示，求下列问题。',font=font,fill='black')
        draw.text((x,y+42),'（1）计算物体运动的速度。',font=font,fill='black')
        draw.text((x,y+84),'（2）说明判断的依据。',font=font,fill='black')
        draw.text((x,y+126),'A. ①②    B. ②③    C. ①③',font=font,fill='black')
        # Diagrams have little recognisable text; they must still stay in crop.
        draw.line([(x+20,y+260),(x+140,y+190),(x+230,y+270),(x+20,y+260)],fill='black',width=3)
        draw.rectangle((x+15,y+340,x+400,y+450),outline='black',width=2)
        draw.line((x+15,y+395,x+400,y+395),fill='black',width=2)
        draw.line((x+190,y+340,x+190,y+450),fill='black',width=2)
    path=output/'synthetic_two_columns.jpg';image.save(path,quality=94)
    fragment=Image.new('RGB',(800,900),'white');draw=ImageDraw.Draw(fragment)
    draw.text((40,100),'（1）求电热水器的电阻。',font=font,fill='black')
    draw.text((40,180),'（2）请说明判断的依据。',font=font,fill='black')
    draw.rectangle((80,300,600,500),outline='black',width=3)
    fragment_path=output/'synthetic_continuation.jpg';fragment.save(fragment_path,quality=94)
    return [path,fragment_path]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--synthetic-font',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.images.glob('*.jpg'))
    if args.synthetic_font:paths.extend(synthetic(args.synthetic_font,args.output))
    summary=[]
    for path in paths:
        image=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        image.thumbnail((2048,2048))
        prepared=args.output/(path.stem+'.input.jpg');image.save(prepared,quality=90)
        data=prepared.read_bytes()
        request={'version':1,'pageId':path.stem,'imageHash':hashlib.sha256(data).hexdigest(),
                 'width':image.width,'height':image.height,
                 'images':[{'mime':'image/jpeg','base64':base64.b64encode(data).decode()}]}
        start=time.monotonic()
        try:
            page=process_page(request)
            result={'file':path.name,'status':'ok' if page['blocks'] else 'manual_required','seconds':round(time.monotonic()-start,2),'page':page}
            draw=ImageDraw.Draw(image)
            for block in page['blocks']:
                b=block['bbox'];x,y,w,h=[b[k] for k in ('x','y','width','height')]
                draw.rectangle((x,y,x+w-1,y+h-1),outline='red',width=4)
                draw.text((x+5,y+6),'Q'+block['printedNumber'],fill='red',stroke_width=1)
            image.save(args.output/(path.stem+'.boxes.jpg'),quality=92)
        except LayoutError as error:
            result={'file':path.name,'status':'manual_required','seconds':round(time.monotonic()-start,2),'code':str(error)}
        (args.output/(path.stem+'.result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        summary.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='page'},ensure_ascii=True),flush=True)
    (args.output/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
