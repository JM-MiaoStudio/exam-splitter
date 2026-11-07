"""Geometry-preserving document preprocessing, independently implemented."""
def flatten_background(image):
    """Estimate smooth illumination with morphological closing; no thresholding.

    Keeps pixel dimensions and does not crop/warp or remove handwriting. This
    is a candidate OCR input, never a replacement for the retained original.
    """
    import cv2
    import numpy as np
    from PIL import Image
    gray=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
    kernel=max(3,round(min(gray.shape)*.08)|1)
    background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
    background=cv2.GaussianBlur(background,(0,0),max(1,kernel/3))
    flat=np.clip(gray.astype(np.float32)*255/np.maximum(background,1),0,255).astype(np.uint8)
    return Image.fromarray(flat).convert('RGB')
