const test = require('node:test');
const assert = require('node:assert/strict');
const {blankReview, validateHumanReview, readImportedReviews} = require('../scripts/templates/nq_train_pending_review.js');
const packet = {packet_sha256:'version-1',review_set_id:'set-1',records:[{
  review_id:'set-1:nq:123',review_number:3,dataset:'nq',sample_id:'123',source_record_sha256:'source-1'}]};
function exported(value=blankReview()) {
  return {schema_version:'nq_train_e_human_review_v1',packet_sha256:'version-1',review_set_id:'set-1',
    reviews:packet.records.map(r=>({...r,human_review:value}))};
}
function confirmed() {
  return {...blankReview(),status:'completed',reviewer:'UI test',initial_answer_status:'correct',
    evidence_status:'sufficient_clear',primary_label:'correct',use:'correct_control',reason:'Reviewed evidence',
    reviewed_at:'2026-10-17T12:00:00Z'};
}
test('new records remain blank and adopting a suggestion cannot imply confirmation',()=>{
  const blank=blankReview(); assert.equal(blank.status,'pending'); assert.equal(blank.primary_label,'');
  const draft={...confirmed(),status:'draft',reviewed_at:null,review_mode:'agent_suggestion_as_draft'};
  assert.equal(validateHumanReview(draft).status,'draft');
});
test('completed labels and core inclusion require consistent human fields',()=>{
  assert.equal(validateHumanReview(confirmed()).status,'completed');
  for(const change of [{primary_label:'uncertain'},{reviewer:''},{reason:''},{use:'include_e_core'},
    {primary_label:'evidence_utilisation'},{supporting_ranks:[6]},{supporting_ranks:[1,1]}]){
    assert.throws(()=>validateHumanReview({...confirmed(),...change}));
  }
});
test('deferral is distinct from a confirmed diagnosis',()=>{
  const deferred={...confirmed(),status:'deferred',primary_label:'uncertain',use:'hold'};
  assert.equal(validateHumanReview(deferred).status,'deferred');
  assert.throws(()=>validateHumanReview({...deferred,status:'completed'}));
});
test('human-reviewed omission of key content can be utilisation while remaining partial',()=>{
  const partial={...confirmed(),initial_answer_status:'partial',primary_label:'evidence_utilisation',
    use:'include_e_core',reason:'Location alone omits the requested name origin; R1 states it clearly.'};
  assert.equal(validateHumanReview(partial).initial_answer_status,'partial');
  for(const evidence_status of ['insufficient','sufficient_difficult','uncertain']){
    assert.throws(()=>validateHumanReview({...partial,evidence_status}));
  }
  assert.throws(()=>validateHumanReview({...partial,primary_label:'correct'}));
});
test('imports require the exact packet, record identity and source hash',()=>{
  assert.equal(Object.values(readImportedReviews(exported(),packet))[0].status,'pending');
  for(const key of ['packet_sha256','review_set_id']){
    assert.throws(()=>readImportedReviews({...exported(),[key]:'other'},packet));
  }
  for(const key of ['review_id','review_number','dataset','sample_id','source_record_sha256']){
    const data=exported();data.reviews[0][key]='other';assert.throws(()=>readImportedReviews(data,packet));
  }
  const data=exported();data.reviews=[];assert.throws(()=>readImportedReviews(data,packet));
});
test('imports do not retain unknown fields or execute reason text',()=>{
  const value={...confirmed(),reason:'<script>not code</script>',unknown:'not retained'};
  const decoded=Object.values(readImportedReviews(exported(value),packet))[0];
  assert.equal(decoded.reason,value.reason);assert.equal(decoded.unknown,undefined);
});
