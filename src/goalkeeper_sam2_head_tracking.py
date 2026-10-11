"""Reusable SAM2 semantic-head fallback with automatic prompts and strict QC."""
from pathlib import Path
import cv2,numpy as np
from src.goalkeeper_sam2 import head_part_direction_evidence


def automatic_head_prompt(observation, offset):
    kp=np.asarray(observation['keypoints']);cf=np.asarray(observation['keypoint_confidences'])
    if cf[0]<.5:return None
    visible=kp[:5][cf[:5]>=.5]-offset
    box=np.r_[visible.min(0)-5,visible.max(0)+5].astype(np.float32)
    points=np.asarray([kp[0],kp[[5,6]].mean(0),kp[[11,12]].mean(0)],np.float32)-offset
    return box,points,np.array([1,0,0])


def track_head_part(data,body_masks,body_qc,predictor,output_dir):
    """Input predictor must use the declared resolution; no human part prompts."""
    import torch
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    prompt=automatic_head_prompt(data['initial'],data['roi'][:2])
    if prompt is None:return dict(supported=False,direction='unknown',reasons=['no_reliable_automatic_nose_prompt'])
    jpg=output_dir/'input_jpg';jpg.mkdir(exist_ok=True)
    for idx,img in enumerate(data['frames'].values()):
        if not cv2.imwrite(str(jpg/f'{idx:05}.jpg'),img):raise ValueError('Head JPEG write failed')
    box,points,labels=prompt;heads={}
    with torch.inference_mode():
        state=predictor.init_state(str(jpg),offload_video_to_cpu=True,offload_state_to_cpu=True)
        predictor.add_new_points_or_box(state,frame_idx=0,obj_id=2,box=box,points=points,labels=labels)
        for idx,ids,logits in predictor.propagate_in_video(state):
            f=data['first']+idx;heads[f]=(logits[0,0].cpu().numpy()>0).astype(np.uint8)
            dest=output_dir/'head_masks'/f'{f:05}.png';dest.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(dest),heads[f]*255):raise ValueError('Head mask write failed')
    return head_part_direction_evidence(heads,body_masks,body_qc,data['kick'],data['pose']['fps'],data['scale'])
