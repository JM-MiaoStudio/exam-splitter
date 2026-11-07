"""Optional geometric/illumination preparation, with inverse pixel mapping.

Uses generic OpenCV operations, no CamScanner binary or recovered source.
Large unambiguous page contours only; internal tables and edge-cut pages abstain.
"""
import cv2
import numpy as np
from PIL import Image

def page_quad(rgb):
    h,w=rgb.shape[:2]
    scale=min(1,800/max(h,w));small=cv2.resize(rgb,(round(w*scale),round(h*scale)))
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    _,mask=cv2.threshold(cv2.GaussianBlur(gray,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not .55*gray.size<area<.94*gray.size:continue
        poly=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):continue
        q=poly[:,0,:].astype(np.float32)
        if q[:,0].min()<3 or q[:,1].min()<3 or q[:,0].max()>gray.shape[1]-4 or q[:,1].max()>gray.shape[0]-4:continue
        # Image must show a materially darker exterior on all four corners.
        corners=[gray[:12,:12],gray[:12,-12:],gray[-12:,:12],gray[-12:,-12:]]
        inside=np.zeros_like(gray);cv2.fillConvexPoly(inside,q.astype(np.int32),255)
        paper=float(np.percentile(gray[inside>0],75))
        if any(float(np.median(c))>paper-25 for c in corners):continue
        center=q.mean(axis=0);q=q[np.argsort(np.arctan2(q[:,1]-center[1],q[:,0]-center[0]))]
        q=np.roll(q,-int(np.argmin(q.sum(axis=1))),axis=0)/scale
        # Expand into background so strokes on page edges are never trimmed.
        q=q.mean(axis=0)+(q-q.mean(axis=0))*1.025
        if q[:,0].min()<0 or q[:,1].min()<0 or q[:,0].max()>=w or q[:,1].max()>=h:continue
        candidates.append(q)
    return candidates[0] if len(candidates)==1 else None

def prepare_document(image, *, perspective=False, shadow=False):
    rgb=np.asarray(image.convert('RGB')).copy();inv=np.eye(3)
    meta={'perspective':False,'shadow':False}
    q=page_quad(rgb) if perspective else None
    if q is not None:
        top,right,bottom,left=[float(np.linalg.norm(q[i]-q[(i+1)%4])) for i in range(4)]
        w,h=round(max(top,bottom)),round(max(left,right))
        if min(w,h)>100 and max(w,h)/min(w,h)<3:
            target=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
            matrix=cv2.getPerspectiveTransform(q.astype(np.float32),target)
            inv=np.linalg.inv(matrix)
            rgb=cv2.warpPerspective(rgb,matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
            meta['perspective']=True
    if shadow:
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        # Local paper quantiles tolerate printed ink; avoid treating broad black
        # background as paper shadow. This is a candidate, not quality proof.
        tiles=[float(np.percentile(tile,85)) for strip in np.array_split(gray,6) for tile in np.array_split(strip,6,axis=1) if tile.size]
        if len(tiles)==36 and min(tiles)>100 and np.percentile(tiles,90)-np.percentile(tiles,10)>35:
            kernel=max(3,round(min(gray.shape)*.08)|1)
            bg=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
            bg=cv2.GaussianBlur(bg,(0,0),max(1,kernel/3))
            gain=np.clip(235/np.maximum(bg.astype(np.float32),1),1,1.35)
            rgb=np.clip(rgb.astype(np.float32)*gain[:,:,None],0,255).astype(np.uint8)
            meta['shadow']=True
    return Image.fromarray(rgb),inv,meta

def restore_rows(rows,inverse,width,height):
    inv=np.asarray(inverse,dtype=np.float64)
    if inv.shape!=(3,3) or not np.isfinite(inv).all() or abs(np.linalg.det(inv))<1e-10:
        raise ValueError('INVALID_PREPROCESS_MAPPING')
    result=[]
    for row in rows:
        quad=np.asarray(row['quad'],dtype=np.float64)
        if quad.shape!=(4,2) or not np.isfinite(quad).all():raise ValueError('INVALID_OCR_QUAD')
        points=np.column_stack((quad,np.ones(4)))@inv.T
        if np.any(abs(points[:,2])<1e-8):raise ValueError('INVALID_PREPROCESS_MAPPING')
        q=points[:,:2]/points[:,2,None]
        if not np.isfinite(q).all() or q[:,0].min()<-2 or q[:,1].min()<-2 or q[:,0].max()>width+2 or q[:,1].max()>height+2:
            raise ValueError('PREPROCESS_MAPPING_OUTSIDE_UPLOAD')
        q[:,0]=np.clip(q[:,0],0,width);q[:,1]=np.clip(q[:,1],0,height)
        result.append({**row,'quad':q.tolist()})
    return result
