"""Conservative temporal runs of camera cues within existing shot boundaries."""
TARGET_VIEWS={'back_of_kicker','angle_45_right','kicker_closeup','back_of_goalkeeper'}


def supported_intervals(frames,votes,shot_start,shot_end,fps,min_samples=4,min_duration=.6,max_gap_sec=.35):
    if len(frames)!=len(votes) or fps<=0:raise ValueError('Invalid samples/fps')
    if any(b<=a for a,b in zip(frames,frames[1:])):raise ValueError('Frames must be strictly increasing')
    if any(not shot_start<=f<shot_end for f in frames):raise ValueError('Sample outside shot')
    runs=[];current=[];view=None
    for fid,vote in zip(frames,votes):
        if vote not in TARGET_VIEWS or vote!=view or (current and (fid-current[-1])/fps>max_gap_sec):
            if current:runs.append((view,current))
            current=[]
        if vote in TARGET_VIEWS:current.append(fid)
        view=vote
    if current:runs.append((view,current))
    result=[]
    for view,run in runs:
        # Do not extend unsupported endpoints to the whole shot.
        duration=(run[-1]+1-run[0])/fps
        if len(run)<min_samples or duration<min_duration:continue
        result.append({'view_candidate':view,'start_frame':run[0],'end_frame_exclusive':run[-1]+1,
            'start_time_sec':run[0]/fps,'end_time_sec':(run[-1]+1)/fps,'support_frames':run,
            'support_sample_count':len(run),'sample_gap_sec':max((b-a)/fps for a,b in zip(run,run[1:])),
            'playback_status':'unknown','human_verified':False,'accuracy_validated':False,'review_required':True,
            'semantic_boundary_verified':False,'scope':'consecutive sampled spatial cues; unobserved in-between frames not individually classified'})
    return result
