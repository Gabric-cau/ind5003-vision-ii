"""Search existing local crop evidence by clip, jersey number, and/or surname."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.jersey_evidence import normalize_name


def search_records(data,index_root,clip=None,number=None,name=None,include_conflicts=False):
    """Exact requested fields are ANDed; clip metadata is not the query identity."""
    rows=[]
    for r in data['tracklets']:
        if clip and clip not in r['clip_id']:continue
        e=r['upright_evidence'];status=r['upright_query']['status']
        number_match=number is not None and e['number']==number
        name_match=bool(name and any(n['text']==normalize_name(name) for n in e['name_tokens']))
        default_match=status in ('number_match_candidate','number_and_name_candidate','name_match_candidate')
        if number is not None and not number_match:continue
        if name and not name_match:continue
        if number is None and not name and not default_match and not (include_conflicts and status=='number_conflict'):continue
        rows.append({'clip_id':r['clip_id'],'segment_id':r['segment_id'],'tracklet_id':r['tracklet_id'],
             'start_sec':r['start_time_sec'],'end_sec':r['end_time_sec'],'recognized_number':e['number'],
             'containing_clip_query_status':status,'containing_clip_expected_query':r.get('expected_query'),
             'requested_number_match':number_match if number is not None else None,
             'requested_name_match':name_match if name else None,
             'name_tokens':e['name_tokens'],'images':[str(index_root/k['image']) for k in r['keyframes']],
             'human_verified':False,'identity_verified':False})
    return sorted(rows,key=lambda r:(r['clip_id'],r['start_sec'],r['tracklet_id']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,default=ROOT/'outputs/jersey_retrieval_2026-10-11_v1/evaluation_v1/retrieval_index.json')
    p.add_argument('--clip');p.add_argument('--number',type=int);p.add_argument('--name');p.add_argument('--limit',type=int,default=10)
    p.add_argument('--include-conflicts',action='store_true');a=p.parse_args()
    if a.clip is None and a.number is None and a.name is None:p.error('Supply --clip, --number or --name')
    data=json.loads(a.index.read_text(encoding='utf-8'))
    rows=search_records(data,a.index.parent.parent,a.clip,a.number,a.name,a.include_conflicts)
    print(json.dumps({'total_matches':len(rows),'results':rows[:max(0,a.limit)],
           'query_semantics':'AND of provided fields; exact normalized names; even single-frame name hits are retrieval leads, not verified identities.',
           'warning':'Text/metadata retrieval only. Not validated player ReID, action matching or slow-motion detection.'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
