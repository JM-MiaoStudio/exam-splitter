"""Map Tencent text polygons to uploaded pixels using explicit provenance.

No API client, secrets, implicit EXIF rotation, or inferred dewarping. A sender
must preserve actual integer resize dimensions and the preprocessing homography.
Nonlinear dewarping is deliberately unsupported without its inverse mapping.
"""
import math
import re
import numpy as np


def cloud_rows(envelope, *, image_hash, width, height):
    """Return RapidOCR-compatible rows; reject unknown/mismatched image space."""
    try:
        if envelope['version'] != 1 or envelope['polygonSpace']!='sent-image' or not re.fullmatch('[0-9a-f]{64}', image_hash):
            raise ValueError()
        if envelope['imageHash'] != image_hash or envelope['originalSize'] != [width,height]:
            raise ValueError()
        for name in ('originalSize','processedSize','sentSize'):
            size=envelope[name]
            if not isinstance(size,list) or len(size)!=2 or any(type(v) is not int or not 1<=v<=10000 for v in size):
                raise ValueError()
        matrix=np.asarray(envelope['originalToProcessed'],dtype=float)
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all():raise ValueError()
        inverse=np.linalg.inv(matrix)
        if not np.isfinite(inverse).all() or np.linalg.cond(matrix)>1e12:raise ValueError()
        response=envelope['response']
        if not isinstance(response,dict) or 'Error' in response:
            raise ValueError()
        # Tencent Polygon is in the sent image; ItemPolygon is corrected space.
        # Angle/legacy Angel describe detected tilt, not a transform to apply.
        for key in ('Angle','Angel'):
            if key in response and (type(response[key]) not in (int,float) or not math.isfinite(response[key])):
                raise ValueError()
        detections=response['TextDetections']
        if not isinstance(detections,list) or len(detections)>5000:raise ValueError()
        pw,ph=envelope['processedSize'];sw,sh=envelope['sentSize']
        rows=[]
        for detection in detections:
            text=detection['DetectedText'];score=detection['Confidence']
            if not isinstance(text,str) or len(text)>20000 or type(score) not in (float,int) or not math.isfinite(score) or not 0<=score<=100:
                raise ValueError()
            polygon=detection['Polygon']
            if not isinstance(polygon,list) or len(polygon)!=4:raise ValueError()
            quad=[];denominators=[];clipped=False
            for point in polygon:
                x,y=point['X'],point['Y']
                if any(type(v) not in (float,int) or not math.isfinite(v) for v in (x,y)):
                    raise ValueError()
                # OCR quad extrapolation may extend slightly beyond page edges.
                # Bound the tolerance, clip explicitly and retain review evidence.
                tx,ty=min(32,max(2,sw*.03)),min(32,max(2,sh*.03))
                if not -tx<=x<=sw+tx or not -ty<=y<=sh+ty:raise ValueError()
                clipped=clipped or not (0<=x<=sw and 0<=y<=sh)
                x,y=min(sw,max(0,x)),min(sh,max(0,y))
                p=inverse@np.array([x*pw/sw,y*ph/sh,1.])
                if abs(p[2])<1e-10:raise ValueError()
                denominators.append(p[2]);x,y=p[:2]/p[2]
                if not np.isfinite([x,y]).all() or not -2<=x<=width+2 or not -2<=y<=height+2:
                    raise ValueError()
                clipped=clipped or not (0<=x<=width and 0<=y<=height)
                quad.append([float(min(width,max(0,x))),float(min(height,max(0,y)))])
            if min(denominators)*max(denominators)<=0:raise ValueError()
            area=abs(sum(quad[i][0]*quad[(i+1)%4][1]-quad[(i+1)%4][0]*quad[i][1] for i in range(4)))/2
            if area<.1:raise ValueError()
            if text.strip():rows.append({'quad':quad,'text':text.strip(),'score':float(score)/100,'source':'cloud','clipped':clipped})
        return rows
    except (KeyError,TypeError,ValueError,OverflowError,np.linalg.LinAlgError):
        raise ValueError('INVALID_CLOUD_GEOMETRY') from None
