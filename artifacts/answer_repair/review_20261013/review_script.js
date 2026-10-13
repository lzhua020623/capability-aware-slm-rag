
'use strict';
const packet=JSON.parse(document.getElementById('packet').textContent), records=packet.records;
const $=id=>document.getElementById(id), storageKey='E-human-review:'+packet.packet_sha256;
const labels={evidence_utilisation:'Utilisation',evidence_accessibility:'Accessibility',evidence_insufficiency:'Insufficiency',correct:'Correct',uncertain:'争议',uncertain_utilisation_or_accessibility:'Utilisation / Accessibility 边界',uncertain_reference_or_insufficiency:'参考答案 / 证据缺失疑点'};
const suggestions={include_candidate:'建议纳入 E 核心候选',hold_candidate:'建议优先复核，暂缓纳入',dispute_source_label:'建议核查原标签',retain_control_candidate:'建议保留正确对照',hold_control:'对照样本有争议，建议复核'};
let selected=records[0].review_id, reviews={}, reviewerDefault='', storageOK=true;
function make(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;}
function current(){return records.find(r=>r.review_id===selected);}
function blank(){return {status:'pending',reviewer:'',initial_answer_status:'',evidence_status:'',primary_label:'',use:'',supporting_ranks:[],reason:'',reviewed_at:null};}
function state(id){return reviews[id]||blank();}
function validate(v){
 const sets={status:['pending','draft','completed'],initial_answer_status:['','incorrect','correct','partial','uncertain'],evidence_status:['','sufficient_clear','sufficient_difficult','insufficient','uncertain'],primary_label:['','evidence_utilisation','evidence_accessibility','evidence_insufficiency','correct','uncertain'],use:['','include_e_core','correct_control','boundary_case','hold']};
 if(!v||typeof v!=='object')throw Error('审核记录无效');
 for(const [key,allowed] of Object.entries(sets))if(!allowed.includes(v[key]))throw Error('审核字段无效：'+key);
 if(typeof v.reviewer!=='string'||typeof v.reason!=='string'||!Array.isArray(v.supporting_ranks)||v.supporting_ranks.some(x=>!Number.isInteger(x)||x<1||x>5)||new Set(v.supporting_ranks).size!==v.supporting_ranks.length)throw Error('审核人、理由或证据编号格式无效');
 if(v.status==='completed'){
  if(!v.reviewer.trim()||!v.initial_answer_status||!v.evidence_status||!v.primary_label||!v.use||typeof v.reviewed_at!=='string'||!Number.isFinite(Date.parse(v.reviewed_at)))throw Error('人工确认记录缺少必要字段');
  if(v.use==='include_e_core'&&(v.primary_label!=='evidence_utilisation'||v.initial_answer_status!=='incorrect'||v.evidence_status!=='sufficient_clear'))throw Error('纳入 E 核心需要同时确认：原答案有实质错误、证据充分且清楚、标签为 Utilisation。');
  if(v.use==='correct_control'&&(v.primary_label!=='correct'||v.initial_answer_status!=='correct'))throw Error('正确对照需要同时确认：原答案语义正确、标签为 Correct。');
 }
 return v;
}
function loadSaved(data){if(data.packet_sha256!==packet.packet_sha256||!Array.isArray(data.reviews))throw Error('文件不是同一版本审核包');const out={};for(const row of data.reviews){if(!records.some(r=>r.review_id===row.review_id)||out[row.review_id])throw Error('未知或重复 sample ID');out[row.review_id]=validate(row.human_review);}return out;}
try{const saved=localStorage.getItem(storageKey);if(saved){const data=JSON.parse(saved);reviews=loadSaved(data);reviewerDefault=typeof data.reviewer_default==='string'?data.reviewer_default:'';}}catch(e){storageOK=false;$('saveStatus').textContent='本机草稿未载入：'+e.message+'。请使用导出文件保存。';}
function payload(){return {schema_version:'e_human_review_v1',packet_id:packet.packet_id,packet_sha256:packet.packet_sha256,source_commits:packet.source_commits,exported_at:new Date().toISOString(),reviewer_default:reviewerDefault,agent_assessments_are_not_human_labels:true,reviews:records.map(r=>({review_id:r.review_id,dataset:r.dataset,sample_id:r.sample_id,human_review:state(r.review_id)}))};}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(payload()));storageOK=true;$('saveStatus').textContent='草稿已保存在此浏览器本机。完成一批后请导出 JSON；换浏览器或版本前也请导出。';}catch(e){storageOK=false;$('saveStatus').textContent='浏览器本地保存不可用，请及时导出 JSON。';}}
function priority(r){return ['hold_candidate','dispute_source_label','hold_control'].includes(r.agent_assessment?.e_core_suggestion);}
function filtered(){const f=$('filter').value,q=$('search').value.trim().toLowerCase();return records.filter(r=>{
 const status=state(r.review_id).status;
 const ok=f==='all'||f===r.cohort||(f==='priority'&&priority(r))||(f==='fresh'&&!r.previous_E_dev50)||(f==='pending'&&status!=='completed')||(f==='completed'&&status==='completed');
 return ok&&(!q||[r.review_id,r.query,r.initial_answer,...r.evidence.map(e=>e.text)].join(' ').toLowerCase().includes(q));});}
function refreshStats(){const done=records.filter(r=>state(r.review_id).status==='completed').length;$('stats').textContent=`27 条 utilisation 候选 · 10 条对照 · 人工确认 ${done} / ${records.length}`;$('progress').value=done;}
function renderList(){const list=$('list');list.replaceChildren();for(const r of filtered()){const b=make('button',undefined,'sample'+(r.review_id===selected?' active':''));b.type='button';b.dataset.reviewId=r.review_id;b.append(make('span',r.query),make('span',r.review_id,'id'),make('span',state(r.review_id).status==='completed'?'✓ 已人工确认':state(r.review_id).status==='draft'?'草稿 · 待确认':'待人工审核','state'));b.onclick=()=>{selected=r.review_id;render();};list.append(b);}refreshStats();}
function fieldValues(){const raw=$('ranks').value.trim();let ranks=[];if(raw){if(!/^\s*[1-5](?:\s*[,，]\s*[1-5])*\s*$/.test(raw))throw Error('证据编号请使用 1–5，并以逗号分隔。');ranks=[...new Set(raw.split(/[,，]/).map(x=>Number(x.trim())))];}return {status:'draft',reviewer:$('reviewer').value.trim(),initial_answer_status:$('answerStatus').value,evidence_status:$('evidenceStatus').value,primary_label:$('label').value,use:$('use').value,supporting_ranks:ranks,reason:$('reason').value.trim(),reviewed_at:null};}
function draft(){try{reviews[selected]=fieldValues();reviewerDefault=$('reviewer').value.trim();$('formError').textContent='';$('reviewState').textContent='草稿已修改，尚未确认。';save();renderList();}catch(e){$('formError').textContent=e.message;}}
function render(){let visible=filtered();if(!visible.some(r=>r.review_id===selected))selected=visible[0]?.review_id;
 $('content').classList.toggle('hidden',!selected);$('empty').classList.toggle('hidden',!!selected);renderList();
 const index=visible.findIndex(r=>r.review_id===selected);$('position').textContent=selected?`当前列表 ${index+1} / ${visible.length}`:'当前列表 0 条';$('previous').disabled=index<=0;$('next').disabled=index<0||index===visible.length-1;if(!selected)return;
 const r=current(),v=state(selected);$('identity').textContent=r.review_id+' · 冻结 C1 / 训练划分';$('query').textContent=r.query;$('initial').textContent=r.initial_answer;
 $('tags').replaceChildren(make('span',r.cohort==='utilisation_candidate'?'原标签 utilisation 候选':'原标签 correct 对照','tag'),make('span',r.previous_E_dev50?'已用于上一轮开发':'本轮新纳入开发','tag'));
 $('evidence').replaceChildren();for(const e of r.evidence){const d=make('details',undefined,'evidence');d.open=e.rank===1;d.append(make('summary',`[${e.rank}] ${e.page_id||e.passage_id||e.evidence_id||'原始证据'} · ${e.text.length.toLocaleString()} 字符`),make('p',e.text));$('evidence').append(d);}
 $('reviewer').value=v.reviewer||reviewerDefault;$('answerStatus').value=v.initial_answer_status;$('evidenceStatus').value=v.evidence_status;$('label').value=v.primary_label;$('use').value=v.use;$('ranks').value=v.supporting_ranks.join(', ');$('reason').value=v.reason;$('formError').textContent='';$('reviewState').textContent=v.status==='completed'?`已人工确认 · ${v.reviewer} · ${new Date(v.reviewed_at).toLocaleString()}`:'尚未人工确认';
 $('assist').open=false;$('original').open=false;const a=r.agent_assessment,box=$('assistantContent');box.replaceChildren();if(a){box.append(make('p',suggestions[a.e_core_suggestion]||a.e_core_suggestion),make('div',`建议标签：${labels[a.suggested_label]||a.suggested_label}；辅助判断把握：${a.confidence}`,'subtle'),make('p',a.reason_zh,'reason'));for(const q of a.quoted_evidence)box.append(make('div',`证据 [${q.rank}]：${q.quote}`,'quote'));box.append(make('p','请你确认：'+a.human_question_zh,'question'));}
 $('sourceContent').replaceChildren(make('p','原标签：'+r.source_annotation.primary_label+'；needs_manual_review='+r.source_annotation.needs_manual_review+'（不代表本轮已人工复核）'),make('p','原标签来源：'+r.source_annotation.annotation_source),make('p','离线参考：'+JSON.stringify(r.offline_reference.gold_answer_or_label)),make('p','原冻结指标 correct：'+r.offline_reference.correct+'；语义正确性需另行判断。'),make('p','C1 commit：'+packet.source_commits.c1),make('p','标注与划分 commit：'+packet.source_commits.annotation_and_split));
}
for(const id of ['reviewer','answerStatus','evidenceStatus','label','use','ranks','reason'])$(id).addEventListener(id==='ranks'?'change':'input',draft);
$('reviewForm').onsubmit=e=>{e.preventDefault();try{const v=fieldValues();v.status='completed';v.reviewed_at=new Date().toISOString();validate(v);reviews[selected]=v;reviewerDefault=v.reviewer;save();render();}catch(err){$('formError').textContent=err.message;}};
for(const id of ['search','filter'])$(id).addEventListener('input',render);
function move(delta){const list=filtered(),i=list.findIndex(r=>r.review_id===selected);if(list[i+delta]){selected=list[i+delta].review_id;render();}}$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('expand').onclick=()=>document.querySelectorAll('#evidence details').forEach(d=>d.open=true);$('collapse').onclick=()=>document.querySelectorAll('#evidence details').forEach(d=>d.open=false);
$('export').onclick=()=>{const blob=new Blob([JSON.stringify(payload(),null,2)],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),a=make('a');a.href=url;a.download='E_human_reviews_'+new Date().toISOString().slice(0,10)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('import').onclick=()=>$('file').click();$('file').onchange=async e=>{const f=e.target.files[0];if(!f)return;try{const data=JSON.parse(await f.text()),incoming=loadSaved(data);if(Object.keys(reviews).length&&!confirm('导入将替换当前审核草稿。建议先导出备份。是否继续？'))return;reviews=incoming;reviewerDefault=typeof data.reviewer_default==='string'?data.reviewer_default:'';save();render();}catch(err){alert('导入失败：'+err.message);}finally{e.target.value='';}};
render();if(storageOK)$('saveStatus').textContent='已读取本机草稿。请定期导出 JSON；新样本的人工结论全部留空。';
