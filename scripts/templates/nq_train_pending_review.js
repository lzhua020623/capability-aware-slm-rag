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
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('审核记录格式无效');
    for (const [key, allowed] of Object.entries(FIELDS)) {
      if (!allowed.includes(value[key])) throw Error('审核字段无效：' + key);
    }
    if (typeof value.reviewer !== 'string' || typeof value.reason !== 'string' ||
        ![true, false, null].includes(value.reference_issue) || !Array.isArray(value.supporting_ranks) ||
        value.supporting_ranks.some(n => !Number.isInteger(n) || n < 1 || n > 5) ||
        new Set(value.supporting_ranks).size !== value.supporting_ranks.length) {
      throw Error('审核人、理由、参考标记或证据编号格式无效');
    }
    if (['completed', 'deferred'].includes(value.status)) {
      if (!value.reviewer.trim() || !value.reason.trim() || typeof value.reviewed_at !== 'string' ||
          !Number.isFinite(Date.parse(value.reviewed_at))) throw Error('请填写审核人和理由');
    } else if (value.reviewed_at !== null) throw Error('草稿不能带已确认时间');
    if (value.status === 'deferred' && (value.primary_label !== 'uncertain' || value.use !== 'hold')) {
      throw Error('暂缓状态不能当作已确认主标签');
    }
    if (value.status === 'completed') {
      if (!value.initial_answer_status || !value.evidence_status || !value.primary_label || !value.use) {
        throw Error('请完整选择原答案、证据、主标签和开发用途');
      }
      if (value.primary_label === 'uncertain' || value.use === 'hold' || value.initial_answer_status === 'uncertain' ||
          value.evidence_status === 'uncertain') throw Error('存在未决判断时，请使用“保存为暂缓”');
      if ((value.primary_label === 'correct') !== (value.initial_answer_status === 'correct')) {
        throw Error('Correct 标签需要与原答案语义正确一致');
      }
      if (value.use === 'correct_control' && value.primary_label !== 'correct') throw Error('正确对照需要 Correct 标签');
      if (value.primary_label === 'evidence_utilisation' &&
          (value.initial_answer_status !== 'incorrect' || value.evidence_status !== 'sufficient_clear')) {
        throw Error('Utilisation 需要原答案有实质错误、证据充分且清楚');
      }
      if (value.use === 'include_e_core' && value.primary_label !== 'evidence_utilisation') {
        throw Error('E 核心开发需要 Utilisation 标签');
      }
      if (value.primary_label === 'evidence_accessibility' && value.evidence_status !== 'sufficient_difficult') {
        throw Error('Accessibility 需要确认信息存在但难以使用');
      }
      if (value.primary_label === 'evidence_insufficiency' && value.evidence_status !== 'insufficient') {
        throw Error('Insufficiency 需要确认缺少必要信息');
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
      throw Error('文件不属于同一版本的完整待审核包');
    }
    const byId = new Map(packet.records.map(row => [row.review_id, row]));
    const output = Object.create(null);
    for (const row of data.reviews) {
      const source = byId.get(row.review_id);
      if (!source || Object.hasOwn(output, row.review_id) || row.dataset !== source.dataset ||
          row.sample_id !== source.sample_id || row.review_number !== source.review_number ||
          row.source_record_sha256 !== source.source_record_sha256) {
        throw Error('审核编号、样本 ID、来源哈希或重复记录不一致');
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
  const states = {pending: '待人工审核', draft: '草稿 · 待确认', completed: '✓ 本页已确认', deferred: '暂缓 · 主标签未定'};
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
      $('saveStatus').textContent = '已保存在此浏览器本机。完成一批后请导出 JSON；导出才便于备份和交回。';
    } catch {
      storageOK = false; $('saveStatus').textContent = '浏览器本地保存不可用，请及时导出 JSON。';
    }
  }
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      const data = JSON.parse(saved); reviews = readImportedReviews(data, packet);
      reviewerDefault = typeof data.reviewer_default === 'string' ? data.reviewer_default : '';
    }
  } catch (error) {
    storageOK = false; $('saveStatus').textContent = '本机草稿未载入：' + error.message + '。可用导出文件恢复。';
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
    $('stats').textContent = `本页已确认 ${counts.completed} / ${records.length} · 暂缓 ${counts.deferred} · 草稿 ${counts.draft}`;
    $('progress').max = records.length; $('progress').value = counts.completed;
  }
  function fieldValues() {
    const raw = $('ranks').value.trim(); let ranks = [];
    if (raw) {
      if (!/^\s*[1-5](?:\s*[,，]\s*[1-5])*\s*$/.test(raw)) throw Error('证据编号须为 1–5，以逗号分隔');
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
      $('formError').textContent = ''; $('reviewState').textContent = '草稿已修改，尚未确认。'; save(); renderList();
    } catch (error) {$('formError').textContent = error.message;}
  }
  function render() {
    const visible = filtered();
    if (!visible.some(r => r.review_id === selected)) selected = visible[0]?.review_id;
    $('content').classList.toggle('hidden', !selected); $('empty').classList.toggle('hidden', !!selected);
    renderList();
    const index = visible.findIndex(r => r.review_id === selected);
    $('position').textContent = selected ? `当前列表 ${index + 1} / ${visible.length} · 原编号 N${current().review_number}` : '当前列表 0 条';
    $('previous').disabled = index <= 0; $('next').disabled = index < 0 || index === visible.length - 1;
    if (!selected) return;
    const r = current(), value = state(selected), agent = r.agent_assessment;
    $('identity').textContent = `N${r.review_number} · NQ Train / development · ${r.sample_id}`;
    $('query').textContent = r.query; $('initial').textContent = r.initial_answer;
    $('gold').textContent = r.offline_reference.gold_answer;
    $('metrics').textContent = `冻结 EM：${r.frozen_em}；F1：${r.frozen_f1.toFixed(4)}。与语义正确性分别判断。`;
    $('tags').replaceChildren(make('span', agent.suggested_diagnosis_label === null ? '争议样本 · 待裁决' : '建议语义可接受 · 待确认', 'tag'),
      make('span', '原始 label：null', 'tag'));
    $('evidence').replaceChildren();
    for (const e of r.evidence) {
      const details = make('details', undefined, 'evidence'); details.open = e.rank === 1;
      details.append(make('summary', `[${e.rank}] ${e.passage_id} · ${e.text.length.toLocaleString()} 字符`), make('p', e.text));
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
    $('adopt').textContent = agent.suggested_diagnosis_label === 'correct' ? '采用辅助建议为草稿' : '争议样本请自行判断或暂缓';
    $('assist').open = false; $('reference').open = false; $('original').open = false;
    $('assistantContent').replaceChildren(make('p', `建议：${labels[agent.suggested_diagnosis_label] || '暂缓裁决'} · 把握：${agent.confidence}`),
      make('p', agent.reason_zh, 'reason'));
    for (const quote of agent.quoted_evidence) $('assistantContent').append(make('div', `R${quote.rank}：${quote.quote}`, 'quote'));
    $('sourceContent').replaceChildren(make('p', '源 main：' + packet.source_main_commit),
      make('p', '原交付 label 为 null，筛查分组及辅助判断不是源标签。'),
      make('p', '源记录 SHA-256：' + r.source_record_sha256), make('p', '审核包 SHA-256：' + packet.packet_sha256));
  }
  $('scope').textContent = `${records.length} 条待审 · ${packet.counts.disputed} 条争议优先 · ${packet.counts.suggested_correct} 条建议可接受 · 此前确认的 ${packet.already_confirmed_count} 条已排除`;
  $('guideScope').textContent = `本页只含首批开发审核中剩余的 ${records.length} 条，保留原 N 编号。此前已确认的 ${packet.already_confirmed_count} 条不在本页。主数据与 gold 以 main 为准，所有材料均来自 E 的 development 部分。`;
  $('filter').options[0].textContent = `全部 ${records.length} 条`;
  $('filter').options[1].textContent = `争议样本 ${packet.counts.disputed} 条`;
  $('filter').options[2].textContent = `建议语义可接受 ${packet.counts.suggested_correct} 条`;
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
    if (!target) {$('jumpError').textContent = '本页没有该编号；已确认的 8 条已排除。'; return;}
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
      if (Object.keys(reviews).length && !await confirmAction('导入会替换本页草稿，建议先导出备份。继续导入？')) return;
      reviews = incoming; reviewerDefault = typeof data.reviewer_default === 'string' ? data.reviewer_default : ''; save(); render();
    } catch (error) {$('fileError').textContent = '导入失败：' + error.message;} finally {event.target.value = '';}
  };
  $('reset').onclick = async () => {
    if (!await confirmAction('只清空本页在此浏览器中的草稿。建议先导出备份。继续？')) return;
    try {localStorage.removeItem(storageKey);} catch { /* In-memory clearing still works. */ }
    reviews = Object.create(null); reviewerDefault = ''; render();
    $('saveStatus').textContent = '本页草稿已清空，源数据和此前 8 条确认未改变。';
  };
  render();
  if (storageOK) $('saveStatus').textContent = '已读取本机草稿。人工字段初始留空；完成一批后请导出 JSON。';
})();
