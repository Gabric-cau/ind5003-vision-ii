"""Read-only visual diagnostics for source-observation event tracks."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from src.goal_plane_rectification import fit_goal_plane,project_image_point
cv2.setNumThreads(2)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def text(im,t,p,color=(255,255,255),scale=.5):cv2.putText(im,t,p,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--directory",default="tracking_v1")
    parser.add_argument('--end-offset',type=int,default=25);args=parser.parse_args()
    base=ROOT/"outputs/goal_event_evidence_2026-10-10";dest=base/args.directory/"qa";dest.mkdir(parents=True,exist_ok=True)
    for p in (base/args.directory).glob("*.json"):
        if p.name=="summary.json":continue
        r=load(p);cache=load(Path(r["cache_path"]));cap=cv2.VideoCapture(cache["video_path"]);tiles=[]
        maps={k:{d["frame"]:d for d in v["trajectory"]} for k,v in r["variants"].items()}
        first=cache["kick_frame"]-8
        for fid in range(first,min(cache["end_frame_exclusive"],cache["kick_frame"]+args.end_offset),2):
            cap.set(1,fid);ok,frame=cap.read()
            if not ok:break
            # The full-width crop retains ball flight and goal without inventing hidden pixels.
            img=frame[230:560].copy()
            for k,color in [("detector_straight",(0,200,255)),("detector_turn",(255,170,0)),("detector_motion_turn",(255,0,255))]:
                d=maps[k].get(fid)
                if d:
                    cv2.circle(img,(round(d["x"]),round(d["y"]-230)),9,color,2)
            tile=cv2.resize(img,(1152,297));tile=cv2.copyMakeBorder(tile,28,0,0,0,cv2.BORDER_CONSTANT)
            text(tile,f'{r["clip_id"]} frame {fid} | gold=straight cyan=turn purple=motion+turn',(8,20),scale=.49)
            tiles.append(tile)
        cap.release()
        for n in range(0,len(tiles),6):cv2.imwrite(str(dest/f'{r["clip_id"]}_frames_{n//6+1}.jpg'),np.vstack(tiles[n:n+6]))
    print(dest)


if __name__=="__main__":main()
