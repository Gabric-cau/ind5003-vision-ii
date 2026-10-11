"""Perspective text rectification without the generic vertical-line 90deg turn.

PP-OCR's general text path rotates crops with height / width >=1.5, suitable
for some vertical text but harmful to tall jersey digits. Here orientation is
preserved. This is not a trained recognizer or a guarantee that text is upright.
"""
import cv2
import numpy as np


def warp_text_upright(image,points):
    points=np.asarray(points,np.float32)
    if points.shape!=(4,2) or not np.isfinite(points).all():raise ValueError('Invalid text quadrilateral')
    width=round(max(np.linalg.norm(points[0]-points[1]),np.linalg.norm(points[2]-points[3])))
    height=round(max(np.linalg.norm(points[0]-points[3]),np.linalg.norm(points[1]-points[2])))
    if width<2 or height<2:return None
    target=np.array([[0,0],[width,0],[width,height],[0,height]],np.float32)
    transform=cv2.getPerspectiveTransform(points,target)
    return cv2.warpPerspective(image,transform,(width,height),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
