/* Local review UI. Automated candidates never become human labels on load. */
'use strict';
const data=window.CAMERA_DATA,$=id=>document.getElementById(id);
const names={back_of_kicker:'罚球者背后',angle_45_right:'罚球者右侧斜机位（约 45°）',slowmo_kicker_closeup:'慢动作 · 罚球者特写',slowmo_gk_back:'慢动作 · 守门员背后',kicker_closeup:'罚球队员球衣匹配特写（身份待核）',back_of_goalkeeper:'守门员背后候选',other_oblique_left:'罚球者左侧斜机位候选',oblique_side_unresolved:'斜机位（左右未确认）',person_closeup_identity_unresolved:'人物特写（身份不明）',unknown:'不确定'};
const speedNames={unknown:'速度不确定',slow_motion_candidate:'慢动作候选（相对动作对齐）',no_slowdown_evidence:'未发现可靠减速证据'};
let current,selected,stopAt=null;const video=$('video');
const storageKey='visionii-camera-review-'+data.classifier_sha256;let drafts={};
try{drafts=JSON.parse(localStorage.getItem(storageKey)||'{}');}catch(e){drafts={};}
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function fmt(t){return Number(t).toFixed(3)+'s';}
for(const c of data.clips){const o=node('option',c.clip_id);o.value=c.clip_id;$('clip').append(o);}
$('summary').textContent=`${data.summary.clips_processed} 段视频 · ${data.summary.segments} 个候选镜头 · 人工验证 0（模型结果）`;
function key(){return current.clip_id+':'+selected.segment_id;}
function showShot(s){selected=s;stopAt=null;video.pause();video.currentTime=s.start_time_sec;
  for(const b of $('segments').children)b.classList.toggle('active',Number(b.dataset.id)===s.segment_id);
  const d=$('detail');d.replaceChildren(node('h2',`镜头 ${s.segment_id+1} · ${fmt(s.start_time_sec)} — ${fmt(s.end_time_sec)}`),
    node('p','画面：'+(names[s.view_candidate]||s.view_candidate)),node('p','速度：'+(speedNames[s.playback_evidence.status]||s.playback_evidence.status)),
    node('p','作业类别候选：'+(names[s.task_class_candidate]||s.task_class_candidate)),
    node('p',`原比赛文件：${current.source_match_video}；时间 ${fmt(s.source_match_start_sec)} — ${fmt(s.source_match_end_sec)}（清单偏移）`),
    node('p',`抽样意见：${s.sample_votes.map(x=>names[x]||x).join(' / ')}；渐变边界：${s.boundary_start_kind==='gradual_candidate'?'是，需定位真实转场':'否 / 未确认'}`));
  const evidence=node('details');evidence.append(node('summary','查看算法证据（非人工复核）'),node('p',s.sample_reasons.join(' / ')),node('p',s.playback_evidence.reason));d.append(evidence);
  $('thumb').src=s.thumbnail;const draft=drafts[key()];$('reviewView').value=draft?.view||'unknown';$('reviewSpeed').value=draft?.playback||'unknown';
  $('reviewStart').value=draft?.start_time_sec??s.start_time_sec.toFixed(6);$('reviewEnd').value=draft?.end_time_sec??s.end_time_sec.toFixed(6);$('reviewNote').value=draft?.note||'';
  $('reviewStatus').textContent=draft?'此段已有本地复核草稿；未写回模型结果。':'本段尚无人工复核草稿。';}
function showClip(){current=data.clips.find(c=>c.clip_id===$('clip').value);video.src=current.video;$('videoError').textContent='';$('segments').replaceChildren();$('categories').replaceChildren();
  for(const c of data.task_classes){const p=current.task_category_presence[c];const box=node('div',names[c],'category');box.append(node('span',p.segment_ids.length?`${p.segment_ids.length} 段自动候选 · 待核实`:'未检测到 · 不能证明不存在'));$('categories').append(box);}
  for(const s of current.segments){const b=node('button',undefined,'shot');b.type='button';b.dataset.id=s.segment_id;b.append(node('strong',`${s.segment_id+1}. ${fmt(s.start_time_sec)} — ${fmt(s.end_time_sec)}`),node('small',names[s.view_candidate]||s.view_candidate),node('small',speedNames[s.playback_evidence.status]));b.onclick=()=>showShot(s);$('segments').append(b);}
  $('segmentCount').textContent=`(${current.segments.length})`;showShot(current.segments[0]);}
$('clip').onchange=showClip;
$('playSegment').onclick=()=>{video.currentTime=selected.start_time_sec;stopAt=selected.end_time_sec;video.play().catch(()=>{$('videoError').textContent='播放未开始，请点击视频原生播放按钮。';});};
function step(n){stopAt=null;video.pause();video.currentTime=Math.max(0,Math.min(current.duration_sec-1/current.fps,video.currentTime+n/current.fps));}
$('prevFrame').onclick=()=>step(-1);$('nextFrame').onclick=()=>step(1);
function enforceEnd(t){if(stopAt!==null&&t>=stopAt-1/current.fps){const lastFrame=Math.max(selected.start_time_sec,stopAt-1/current.fps);stopAt=null;video.pause();video.currentTime=lastFrame;}}
video.addEventListener('timeupdate',()=>{$('time').textContent=`${fmt(video.currentTime)} / ${fmt(current.duration_sec)}`;enforceEnd(video.currentTime);});
// timeupdate can lag by ~250 ms and display the following shot. Prefer the
// presented-frame callback and clamp to the selected interval's final frame.
if(video.requestVideoFrameCallback){const watchFrame=(_,meta)=>{enforceEnd(meta.mediaTime);video.requestVideoFrameCallback(watchFrame);};video.requestVideoFrameCallback(watchFrame);}
video.addEventListener('error',()=>{$('videoError').textContent='视频未载入。请通过项目的本地复核服务器打开本页，并确认 review/videos 中已生成 H.264 视频。';});
$('saveReview').onclick=()=>{const reviewer=$('reviewer').value.trim(),start=Number($('reviewStart').value),end=Number($('reviewEnd').value);
  if(!reviewer||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>current.duration_sec+.00001){$('reviewStatus').textContent='请填写复核者，并检查起止时间（需在本视频范围内）。';return;}
  const record={clip_id:current.clip_id,segment_id:selected.segment_id,source_sha256:current.source_sha256,reviewer,
    view:$('reviewView').value,playback:$('reviewSpeed').value,start_time_sec:start,end_time_sec:end,note:$('reviewNote').value,
    recorded_at:new Date().toISOString(),provenance:'explicit_browser_reviewer_entry',independently_validated:false};
  const next={...drafts,[key()]:record};try{localStorage.setItem(storageKey,JSON.stringify(next));drafts=next;$('reviewStatus').textContent='本段草稿已保存在浏览器；模型结果未修改。请导出备份。';}catch(e){$('reviewStatus').textContent='浏览器存储不可用，未保存。';}};
$('exportReview').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'camera_review_drafts_v1',notes:'User-entered drafts, not independently validated ground truth.',records:Object.values(drafts)},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download='camera_review_drafts.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
showClip();
