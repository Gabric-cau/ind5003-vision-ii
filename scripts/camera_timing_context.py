"""Camera-specific wide-goal and near-net geometry from original sample pixels."""
import copy,json,hashlib
from pathlib import Path
import cv2,numpy as np
from src.goal_geometry import detect_goal_frame
from src.camera_timing_v2 import goal_camera_hypotheses
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def near_net(frame):
    h,w=frame.shape[:2];hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,0,165],np.uint8),np.array([179,90,255],np.uint8))
    edges=cv2.Canny(mask,60,180);lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=55,minLineLength=int(.27*h),maxLineGap=int(.04*h))
    left=[];right=[]
    for x1,y1,x2,y2 in lines[:,0,:] if lines is not None else []:
        if abs(y2-y1)<.27*h or abs(x2-x1)>1.3*abs(y2-y1):continue
        x=(x1+x2)/2
        if x<.22*w:left.append([int(x1),int(y1),int(x2),int(y2)])
        elif x>.78*w:right.append([int(x1),int(y1),int(x2),int(y2)])
    return {'supported':bool(left and right),'left_lines':left[:6],'right_lines':right[:6],
            'method':'long_white_side_lines_both_image_borders; not a net or depth proof'}


def extract(scan,out,resume):
    path=out/'context_v2'/f"{scan['clip_id']}.json";signature={'source_sha256':scan['source_sha256'],'script_sha256':sha(__file__),
        'goal_code_sha256':sha(ROOT/'src/goal_geometry.py'),'scan_sha256':sha(out/'scan'/f"{scan['clip_id']}.json")}
    if path.exists():
        if not resume:raise FileExistsError(path)
        result=json.loads(path.read_text(encoding='utf-8'))
        if result['signature']!=signature:raise ValueError('Context cache changed')
        return result
    cap=cv2.VideoCapture(scan['video_path']);samples={}
    for s in scan['segments']:
        for fid in s['sample_frame_ids']:
            cap.set(1,fid);ok,frame=cap.read()
            if not ok:raise ValueError('Sample decode failed')
            proposal=detect_goal_frame(frame,max_goal_width_fraction=.98,max_post_height_fraction=.85);goal=proposal.get('selected')
            samples[str(fid)]={'goal':goal,'goal_proposal_status':proposal['status'],
                 'goal_camera':goal_camera_hypotheses(goal['corners'],scan['width'],scan['height']) if goal else {'status':'no_supported_goal'},
                 'foreground_net_cue':near_net(frame)}
    cap.release();return {'signature':signature,'samples':samples,'human_verified':False}


def merge(features,context):
    result=copy.deepcopy(features)
    for s in result['segments']:
        for sample in s['samples']:
            sample.update(context['samples'][str(sample['frame'])])
    return result
