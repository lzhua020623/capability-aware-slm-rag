const assert = require('node:assert/strict');
const {blankReview, validateHumanReview, readImportedReviews} = require('../scripts/templates/nq_label_audit.js');
const empty = blankReview();
assert.equal(empty.answer_supported_by_top5, '');
assert.equal(validateHumanReview(empty).status, 'pending');
const decision = {...empty, status:'completed', reviewer:'test', reason:'Valid answer, unsupported by this evidence.',
  initial_answer_status:'correct', evidence_status:'insufficient', answer_supported_by_top5:false,
  primary_label:'correct', use:'correct_control', reviewed_at:'2026-10-22T00:00:00Z'};
assert.equal(validateHumanReview(decision).answer_supported_by_top5, false);
assert.throws(()=>validateHumanReview({...decision,answer_supported_by_top5:''}));
assert.throws(()=>validateHumanReview({...decision,answer_supported_by_top5:'uncertain'}));
assert.throws(()=>validateHumanReview({...decision,primary_label:'evidence_utilisation'}));
const source={review_id:'audit:nq:1234567890123456789',review_number:1,dataset:'nq',sample_id:'1234567890123456789',source_record_sha256:'source'};
const packet={review_set_id:'audit',packet_sha256:'packet',records:[source]};
const data={schema_version:'nq_label_audit_human_review_v1',review_set_id:'audit',packet_sha256:'packet',reviews:[{...source,human_review:decision}]};
assert.equal(readImportedReviews(data,packet)[source.review_id].answer_supported_by_top5,false);
assert.throws(()=>readImportedReviews({...data,packet_sha256:'different'},packet));
assert.throws(()=>readImportedReviews({...data,reviews:[{...data.reviews[0],sample_id:1234567890123456789}]},packet));
console.log('NQ label audit page validation passed');
