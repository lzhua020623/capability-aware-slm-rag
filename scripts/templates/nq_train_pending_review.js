/* Shared pure validation is exported for Node checks; the browser stays offline. */
(function () {
  'use strict';
  const FIELDS = {
    status: ['pending', 'draft', 'completed', 'deferred'],
    initial_answer_status: ['', 'incorrect', 'correct', 'partial', 'uncertain'],
    evidence_status: ['', 'sufficient_clear', 'sufficient_difficult', 'insufficient', 'uncertain'],
    primary_label: ['', 'correct', 'evidence_utilisation', 'evidence_accessibility', 'evidence_insufficiency', 'uncertain'],
    use: ['', 'include_e_core', 'correct_control', 'boundary_case', 'hold'],
    review_mode: ['manual', 'agent_suggestion_as_draft'],
  };
  function blankReview() {
    return {status: 'pending', reviewer: '', initial_answer_status: '', evidence_status: '',
      primary_label: '', use: '', supporting_ranks: [], reference_issue: null,
      reason: '', reviewed_at: null, review_mode: 'manual'};
  }
  function validateHumanReview(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Invalid review-record format');
    for (const [key, allowed] of Object.entries(FIELDS)) {
      if (!allowed.includes(value[key])) throw Error('Invalid review field: ' + key);
    }
    if (typeof value.reviewer !== 'string' || typeof value.reason !== 'string' ||
        ![true, false, null].includes(value.reference_issue) || !Array.isArray(value.supporting_ranks) ||
        value.supporting_ranks.some(n => !Number.isInteger(n) || n < 1 || n > 5) ||
        new Set(value.supporting_ranks).size !== value.supporting_ranks.length) {
      throw Error('Reviewer, reason, reference flag, or evidence-number format is invalid');
    }
    if (['completed', 'deferred'].includes(value.status)) {
      if (!value.reviewer.trim() || !value.reason.trim() || typeof value.reviewed_at !== 'string' ||
          !Number.isFinite(Date.parse(value.reviewed_at))) throw Error('Please fill in reviewer and reason');
    } else if (value.reviewed_at !== null) throw Error('A draft cannot carry a confirmation timestamp');
    if (value.status === 'deferred' && (value.primary_label !== 'uncertain' || value.use !== 'hold')) {
      throw Error('Deferred status cannot be treated as a confirmed primary label');
    }
    if (value.status === 'completed') {
      if (!value.initial_answer_status || !value.evidence_status || !value.primary_label || !value.use) {
        throw Error('Please complete original answer, evidence, primary label, and development use');
      }
      if (value.primary_label === 'uncertain' || value.use === 'hold' || value.initial_answer_status === 'uncertain' ||
          value.evidence_status === 'uncertain') throw Error('When a judgment is still open, use “Save as deferred”');
      if ((value.primary_label === 'correct') !== (value.initial_answer_status === 'correct')) {
        throw Error('The Correct label must match a semantically correct original answer');
      }
      if (value.use === 'correct_control' && value.primary_label !== 'correct') throw Error('A correct control requires the Correct label');
      if (value.primary_label === 'evidence_utilisation' &&
          (!['incorrect', 'partial'].includes(value.initial_answer_status) || value.evidence_status !== 'sufficient_clear')) {
        throw Error('Utilisation requires a substantive error or omitted key content in the original answer, and evidence that is sufficient and clear');
      }
      if (value.use === 'include_e_core' && value.primary_label !== 'evidence_utilisation') {
        throw Error('E core development requires the Utilisation label');
      }
      if (value.primary_label === 'evidence_accessibility' && value.evidence_status !== 'sufficient_difficult') {
        throw Error('Accessibility requires confirming the information is present but hard to use');
      }
      if (value.primary_label === 'evidence_insufficiency' && value.evidence_status !== 'insufficient') {
        throw Error('Insufficiency requires confirming necessary information is missing');
      }
    }
    // Import only known fields; arbitrary file content never becomes UI instructions.
    return Object.fromEntries(Object.keys(blankReview()).map(key => [key,
      key === 'supporting_ranks' ? [...value[key]] : value[key]]));
  }
  function readImportedReviews(data, packet) {
    if (!data || data.schema_version !== 'nq_train_e_human_review_v1' ||
        data.packet_sha256 !== packet.packet_sha256 || data.review_set_id !== packet.review_set_id ||
        !Array.isArray(data.reviews) || data.reviews.length !== packet.records.length) {
      throw Error('The file is not the complete pending-review packet of the same version');
    }
    const byId = new Map(packet.records.map(row => [row.review_id, row]));
    const output = Object.create(null);
    for (const row of data.reviews) {
      const source = byId.get(row.review_id);
      if (!source || Object.hasOwn(output, row.review_id) || row.dataset !== source.dataset ||
          row.sample_id !== source.sample_id || row.review_number !== source.review_number ||
          row.source_record_sha256 !== source.source_record_sha256) {
        throw Error('Review number, sample ID, source hash, or duplicate records do not match');
      }
      output[row.review_id] = validateHumanReview(row.human_review);
    }
    return output;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {blankReview, validateHumanReview, readImportedReviews};
  }
  if (typeof document === 'undefined') return;

  const $ = id => document.getElementById(id);
  const packet = JSON.parse($('packet').textContent), records = packet.records;
  const storageKey = 'E-NQ-Train-human-review:' + packet.packet_sha256;
  const labels = {correct: 'Correct', evidence_utilisation: 'Utilisation',
    evidence_accessibility: 'Accessibility', evidence_insufficiency: 'Insufficiency'};
  const states = {pending: 'Awaiting human review', draft: 'Draft · pending confirmation', completed: '✓ Confirmed on this page', deferred: 'Deferred · primary label unset'};
  let selected = records[0].review_id, reviews = Object.create(null), reviewerDefault = '', storageOK = true;
  function make(tag, text, cls) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (cls) el.className = cls;
    return el;
  }
  const current = () => records.find(r => r.review_id === selected);
  const state = id => reviews[id] || blankReview();
  function confirmAction(message) {
    return new Promise(resolve => {
      const modal = $('confirmDialog'); $('confirmMessage').textContent = message;
      const finish = answer => {modal.close(); resolve(answer);};
      $('cancelConfirm').onclick = () => finish(false);
      $('acceptConfirm').onclick = () => finish(true);
      modal.oncancel = event => {event.preventDefault(); finish(false);};
      modal.showModal();
    });
  }
  function payload() {
    return {schema_version: 'nq_train_e_human_review_v1', packet_id: packet.packet_id,
      packet_sha256: packet.packet_sha256, review_set_id: packet.review_set_id,
      source_main_commit: packet.source_main_commit, sources_sha256: packet.sources_sha256,
      exported_at: new Date().toISOString(), reviewer_default: reviewerDefault,
      review_basis: 'Offline, reference-visible assisted review; not independent blind annotation',
      agent_assessments_are_not_human_labels: true,
      reviews: records.map(r => ({review_id: r.review_id, review_number: r.review_number,
        dataset: r.dataset, sample_id: r.sample_id, source_record_sha256: r.source_record_sha256,
        human_review: state(r.review_id)}))};
  }
  function save() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(payload())); storageOK = true;
      $('saveStatus').textContent = 'Saved in this browser. After finishing a batch, export JSON; export is what enables backup and hand-back.';
    } catch {
      storageOK = false; $('saveStatus').textContent = 'Browser local save is unavailable; export JSON promptly.';
    }
  }
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      const data = JSON.parse(saved); reviews = readImportedReviews(data, packet);
      reviewerDefault = typeof data.reviewer_default === 'string' ? data.reviewer_default : '';
    }
  } catch (error) {
    storageOK = false; $('saveStatus').textContent = 'Local draft was not loaded: ' + error.message + '. You can restore from an export file.';
  }
  function filtered() {
    const filter = $('filter').value, query = $('search').value.trim().toLowerCase();
    return records.filter(r => {
      const status = state(r.review_id).status, label = r.agent_assessment.suggested_diagnosis_label;
      const match = filter === 'all' || (filter === 'disputed' && label === null) ||
        (filter === 'acceptable' && label === 'correct') || (filter === 'pending' && status !== 'completed') ||
        (['draft', 'completed', 'deferred'].includes(filter) && status === filter);
      return match && (!query || [`N${r.review_number}`, r.sample_id, r.query, r.initial_answer,
        r.offline_reference.gold_answer, ...r.evidence.map(e => e.text)].join(' ').toLowerCase().includes(query));
    });
  }
  function renderList() {
    $('list').replaceChildren();
    for (const r of filtered()) {
      const button = make('button', undefined, 'sample' + (r.review_id === selected ? ' active' : ''));
      button.type = 'button'; button.dataset.reviewId = r.review_id;
      button.append(make('span', `N${r.review_number}`, 'number'), make('span', r.query),
        make('span', r.sample_id, 'id'), make('span', states[state(r.review_id).status], 'state'));
      button.onclick = () => {selected = r.review_id; render();}; $('list').append(button);
    }
    const counts = {completed: 0, deferred: 0, draft: 0};
    for (const r of records) if (Object.hasOwn(counts, state(r.review_id).status)) counts[state(r.review_id).status]++;
    $('stats').textContent = `Confirmed on this page ${counts.completed} / ${records.length} · deferred ${counts.deferred} · drafts ${counts.draft}`;
    $('progress').max = records.length; $('progress').value = counts.completed;
  }
  function fieldValues() {
    const raw = $('ranks').value.trim(); let ranks = [];
    if (raw) {
      if (!/^\s*[1-5](?:\s*[,，]\s*[1-5])*\s*$/.test(raw)) throw Error('Evidence numbers must be 1–5, comma-separated');
      ranks = [...new Set(raw.split(/[,，]/).map(x => Number(x.trim())))];
    }
    return {status: 'draft', reviewer: $('reviewer').value.trim(), initial_answer_status: $('answerStatus').value,
      evidence_status: $('evidenceStatus').value, primary_label: $('label').value, use: $('use').value,
      reference_issue: $('referenceIssue').value === '' ? null : $('referenceIssue').value === 'true',
      supporting_ranks: ranks, reason: $('reason').value.trim(), reviewed_at: null,
      review_mode: state(selected).review_mode};
  }
  function draft() {
    try {
      reviews[selected] = validateHumanReview(fieldValues()); reviewerDefault = $('reviewer').value.trim();
      $('formError').textContent = ''; $('reviewState').textContent = 'Draft changed, not yet confirmed.'; save(); renderList();
    } catch (error) {$('formError').textContent = error.message;}
  }
  function render() {
    const visible = filtered();
    if (!visible.some(r => r.review_id === selected)) selected = visible[0]?.review_id;
    $('content').classList.toggle('hidden', !selected); $('empty').classList.toggle('hidden', !!selected);
    renderList();
    const index = visible.findIndex(r => r.review_id === selected);
    $('position').textContent = selected ? `Current list ${index + 1} / ${visible.length} · original number N${current().review_number}` : 'Current list 0 items';
    $('previous').disabled = index <= 0; $('next').disabled = index < 0 || index === visible.length - 1;
    if (!selected) return;
    const r = current(), value = state(selected), agent = r.agent_assessment;
    $('identity').textContent = `N${r.review_number} · NQ Train / development · ${r.sample_id}`;
    $('query').textContent = r.query; $('initial').textContent = r.initial_answer;
    $('gold').textContent = r.offline_reference.gold_answer;
    $('metrics').textContent = `Frozen EM: ${r.frozen_em}; F1: ${r.frozen_f1.toFixed(4)}. Judge semantic correctness separately.`;
    $('tags').replaceChildren(make('span', agent.suggested_diagnosis_label === null ? 'Disputed sample · pending decision' : 'Suggested semantically acceptable · pending confirmation', 'tag'),
      make('span', 'Original label: null', 'tag'));
    $('evidence').replaceChildren();
    for (const e of r.evidence) {
      const details = make('details', undefined, 'evidence'); details.open = e.rank === 1;
      details.append(make('summary', `[${e.rank}] ${e.passage_id} · ${e.text.length.toLocaleString()} chars`), make('p', e.text));
      $('evidence').append(details);
    }
    $('reviewer').value = value.reviewer || reviewerDefault;
    $('answerStatus').value = value.initial_answer_status; $('evidenceStatus').value = value.evidence_status;
    $('label').value = value.primary_label; $('use').value = value.use;
    $('referenceIssue').value = value.reference_issue === null ? '' : String(value.reference_issue);
    $('ranks').value = value.supporting_ranks.join(', '); $('reason').value = value.reason;
    $('formError').textContent = '';
    $('reviewState').textContent = states[value.status] + (value.reviewed_at ? ` · ${value.reviewer} · ${new Date(value.reviewed_at).toLocaleString()}` : '');
    $('adopt').disabled = agent.suggested_diagnosis_label !== 'correct';
    $('adopt').textContent = agent.suggested_diagnosis_label === 'correct' ? 'Adopt assisted suggestion as draft' : 'For disputed samples, judge yourself or defer';
    $('assist').open = false; $('reference').open = false; $('original').open = false;
    $('assistantContent').replaceChildren(make('p', `Suggestion: ${labels[agent.suggested_diagnosis_label] || 'defer decision'} · confidence: ${agent.confidence}`),
      make('p', agent.reason_zh, 'reason'));
    for (const quote of agent.quoted_evidence) $('assistantContent').append(make('div', `R${quote.rank}：${quote.quote}`, 'quote'));
    $('sourceContent').replaceChildren(make('p', 'Source main: ' + packet.source_main_commit),
      make('p', 'The delivered label is null; screening group and assisted judgment are not source labels.'),
      make('p', 'Source-record SHA-256: ' + r.source_record_sha256), make('p', 'Review-packet SHA-256: ' + packet.packet_sha256));
  }
  $('scope').textContent = `${records.length} pending · ${packet.counts.disputed} disputed first · ${packet.counts.suggested_correct} suggested acceptable · previously confirmed ${packet.already_confirmed_count} excluded`;
  $('guideScope').textContent = `This page contains only the remaining ${records.length} items from the first development review, keeping original N numbers. Previously confirmed ${packet.already_confirmed_count} items are not on this page. Main data and gold follow main; all materials are from the E development split.`;
  $('filter').options[0].textContent = `All ${records.length} items`;
  $('filter').options[1].textContent = `Disputed samples ${packet.counts.disputed} items`;
  $('filter').options[2].textContent = `Suggested semantically acceptable ${packet.counts.suggested_correct} items`;
  for (const id of ['reviewer', 'answerStatus', 'evidenceStatus', 'label', 'use', 'referenceIssue', 'reason']) $(id).addEventListener('input', draft);
  $('ranks').addEventListener('change', draft);
  $('reviewForm').onsubmit = event => {
    event.preventDefault();
    try {
      const value = fieldValues(); value.status = 'completed'; value.reviewed_at = new Date().toISOString();
      reviews[selected] = validateHumanReview(value); reviewerDefault = value.reviewer; save(); render();
    } catch (error) {$('formError').textContent = error.message;}
  };
  $('defer').onclick = () => {
    try {
      const value = fieldValues(); value.status = 'deferred'; value.primary_label = 'uncertain'; value.use = 'hold';
      value.reviewed_at = new Date().toISOString(); reviews[selected] = validateHumanReview(value);
      reviewerDefault = value.reviewer; save(); render();
    } catch (error) {$('formError').textContent = error.message;}
  };
  $('adopt').onclick = () => {
    const agent = current().agent_assessment;
    if (agent.suggested_diagnosis_label !== 'correct') return;
    $('answerStatus').value = 'correct'; $('label').value = 'correct'; $('use').value = 'correct_control';
    $('evidenceStatus').value = agent.evidence_sufficient === true ?
      (agent.evidence_accessible === true ? 'sufficient_clear' : agent.evidence_accessible === false ? 'sufficient_difficult' : '') : '';
    $('referenceIssue').value = agent.reference_issue === null ? '' : String(agent.reference_issue);
    $('ranks').value = [...new Set(agent.quoted_evidence.map(e => e.rank))].join(', ');
    $('reason').value = agent.reason_zh; draft();
    if (reviews[selected]) {reviews[selected].review_mode = 'agent_suggestion_as_draft'; save();}
  };
  for (const id of ['search', 'filter']) $(id).addEventListener('input', render);
  function move(delta) {
    const visible = filtered(), index = visible.findIndex(r => r.review_id === selected);
    if (visible[index + delta]) {selected = visible[index + delta].review_id; render();}
  }
  $('previous').onclick = () => move(-1); $('next').onclick = () => move(1);
  $('jump').onclick = () => {
    const input = $('jumpNumber').value.trim(), match = input.match(/^n?(\d+)$/i);
    const target = match && records.find(r => r.review_number === Number(match[1]));
    if (!target) {$('jumpError').textContent = 'That number is not on this page; the 8 confirmed items were excluded.'; return;}
    $('jumpError').textContent = ''; $('filter').value = 'all'; $('search').value = ''; selected = target.review_id; render();
  };
  $('jumpNumber').onkeydown = event => {if (event.key === 'Enter') $('jump').click();};
  $('expand').onclick = () => document.querySelectorAll('#evidence details').forEach(e => {e.open = true;});
  $('collapse').onclick = () => document.querySelectorAll('#evidence details').forEach(e => {e.open = false;});
  $('export').onclick = () => {
    const blob = new Blob([JSON.stringify(payload(), null, 2)], {type: 'application/json;charset=utf-8'});
    const url = URL.createObjectURL(blob), link = make('a'); link.href = url;
    link.download = 'NQ_Train_E_pending92_reviews_' + new Date().toISOString().slice(0, 10) + '.json';
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  $('import').onclick = () => $('file').click();
  $('file').onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    $('fileError').textContent = '';
    try {
      const data = JSON.parse(await file.text()), incoming = readImportedReviews(data, packet);
      if (Object.keys(reviews).length && !await confirmAction('Import will replace this page’s draft; export a backup first. Continue importing?')) return;
      reviews = incoming; reviewerDefault = typeof data.reviewer_default === 'string' ? data.reviewer_default : ''; save(); render();
    } catch (error) {$('fileError').textContent = 'Import failed: ' + error.message;} finally {event.target.value = '';}
  };
  $('reset').onclick = async () => {
    if (!await confirmAction('This only clears this page’s draft in this browser. Export a backup first. Continue?')) return;
    try {localStorage.removeItem(storageKey);} catch { /* In-memory clearing still works. */ }
    reviews = Object.create(null); reviewerDefault = ''; render();
    $('saveStatus').textContent = 'This page’s draft was cleared; source data and the previous 8 confirmations are unchanged.';
  };
  render();
  if (storageOK) $('saveStatus').textContent = 'Local draft loaded. Human fields start blank; export JSON after finishing a batch.';
})();
