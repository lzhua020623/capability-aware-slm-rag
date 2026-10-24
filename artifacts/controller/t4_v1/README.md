# T4 开发集完整结果 · 2026-10-25

GPU 推理及冻结自动评分已完成。**Adaptive 在 NQ EM 上持平，在 FEVER 净增4题；E与全增强仍有负净收益。** NQ语义审核、AC6与轨迹人工验收尚未完成，尚未进行T5选型或最终测试。

## 结果是否可用

[Kaggle V2](https://www.kaggle.com/code/yearo111/controller-t3-cuda-engineering-frozen-20261024?scriptVersionId=361785012)运行31,265.1秒（约8.68小时），页面显示failure。实际2,800条预测全部保存后，末尾汇总投影遗漏`total_seconds`，导致`KeyError`。这是汇总工程错误；原始记录中有完整耗时字段，所有固定ID、七组配对、输入/提示哈希、预算、运行环境和工程凭证均通过验收，调用失败及预算阻断为0。没有重新生成答案。

原始[manifest](raw_metadata/manifest.json)的failed状态保留，不改写历史。[验收](release_audit.json)和[从原预测重建的汇总](reconstructed_summary.json)单独保存。当前runner仅补齐汇总所需字段；新增回归测试从400道虚构题、七组、2,800条保存一直执行到成功汇总。774项完整回归通过（85.605秒），测试流关闭清理后该回归再次通过（16.795秒），见[验证](validation/summary_fix_tests.json)。冻结GPU包及原始预测未修改。

原始ZIP为`controller_t4_development_v1_outputs.zip`，14,232,583字节，SHA256：`be3b926b8f808cd39b1c32b64564e5e5619518bebd2585af3fd8f0a6616bc7a5`。其中原始预测127,988,603字节，SHA256：`972d5571b8f2e2263581c17a089a01f5c76498f7057e69e276d5206ddb30a1ad`。Git保留原始ZIP和小型元数据；128MB解压预测仅位于忽略目录`data/processed/controller_results/t4_v1/`，可从ZIP无损恢复。

## 七组比较

每个数据集200条development。NQ使用冻结EM，FEVER使用冻结标签准确率；各列均为正确数/200。NQ的语义等价答案另行审核，不修改本次gold或别名。本轮E为NQ `short_answer_quote`、FEVER `verify_repair`，共享Qwen/NF4；结论对应这两个冻结策略和本批数据。

| 比较组 | NQ EM | FEVER准确率 | NQ 修好 / 改坏 | FEVER 修好 / 改坏 |
|---|---:|---:|---:|---:|
| C1缓存基线 | 78（39.0%） | 170（85.0%） | 0 / 0 | 0 / 0 |
| Always Retrieve | 85（42.5%） | 171（85.5%） | 11 / 4 | 5 / 4 |
| Always Rerank | 81（40.5%） | 172（86.0%） | 5 / 2 | 5 / 3 |
| Always Repair（E） | 68（34.0%） | 166（83.0%） | 4 / 14 | 2 / 6 |
| Full 串行全增强 | 72（36.0%） | 166（83.0%） | 10 / 16 | 5 / 9 |
| Adaptive | 78（39.0%） | 174（87.0%） | 1 / 1 | 4 / 0 |
| Adaptive No Probe | 79（39.5%） | 174（87.0%） | 2 / 1 | 4 / 0 |

NQ token F1：C1为0.553569，Adaptive为0.553021，No Probe为0.557711，Always Retrieve为0.568517；其余完整指标见[evaluation.json](evaluation.json)。送入模块、生成新候选、`answer_replaced`状态均不等于改坏；上表改坏严格指原答指标正确、最终指标错误。

## 损失来自哪里

**NQ路由存在漏修。** 离线观察到的“保留+C1后三个单动作”逐题最优为93/200，Adaptive为78/200。差距由1个可避免误修和14个漏掉的修复机会构成；14题中10题被保留原答、2题送去重排、1题送去E、1题送去补检索。10个漏修可被固定补检索组修好（不同动作的成功集合有重叠）。这是当前动作集合的观察上限，不能当作可部署性能、纯Diagnosis因果损失或模型内在上限；NQ仍受EM口径影响。

**E既有输出契约问题，也有接受候选后的误修。** NQ单独E有111/200次invalid_output（55.5%）：引文不是证据逐字切片50次、RANK无效39次、引文超60词20次，其余行格式/短答案长度各1次；另有3次候选无法锚定。35次实际改答按EM修好4题、改坏14题。FEVER单独E有56次invalid_output，其中46次为引文并非逐字片段、8次为repair没有改变原标签、2次JSON解析失败；实际改答修好2题、改坏6题。格式回退不等于最终答案错误，所有回退保留并照常计入比较，见[错误分类](diagnostics.json)。

Full中逐阶段正确数为：NQ **78→85→84→72**，FEVER **170→171→171→166**，依次对应C1→补检索→重排→E。在已有前序状态上，最后E阶段NQ修4坏16、FEVER修0坏5，不能与E从原C1单独执行的数字混用。

**不能因单独E整体为负就声称它拖累了当前Adaptive。** Adaptive的NQ E子集16题最终没有改答（13次invalid_output、2次keep、1次无法锚定），净值为0；Adaptive的唯一NQ误修来自补检索，唯一修好来自重排。FEVER Adaptive只选择保留或补检索，E与重排触发数为0，FEVER的4题收益来自选择性补检索。FEVER观察最优179/200，Adaptive174/200，仍漏掉5个单动作可修机会。

**Probe在这批没有额外准确率收益。** NQ调用56次Probe，Adaptive比No Probe少正确1题；FEVER调用53次Probe，最终200个标签与No Probe完全一致。NQ额外Qwen调用102对65，FEVER119对67。该比较只支持本批配置的开发观察，不能推广为所有Probe都无效。

## 成本与统计边界

| 数据集 | Adaptive增量Qwen调用 / Full | Adaptive增量tokens估计 / Full | 增量tokens少用 |
|---|---:|---:|---:|
| NQ | 102 / 584 | 111,759 / 488,569 | 77.1% |
| FEVER | 119 / 590 | 171,624 / 946,069 | 81.9% |

这里包含Probe，token口径为输入token加重新分词估计的输出token，**仅统计缓存C1之后的增量**。共有C1的生成/初检索成本不完整，因此不能据此宣布proposal的完整流水线AC5通过。组内累计耗时（排除共用C1及资源加载）：NQ Adaptive868.4秒、Full5003.9秒；FEVER Adaptive1791.5秒、Full8172.8秒。它们是本次交错执行中累计的组内计时，不是分别独立部署的墙钟实验。

这批是development描述性结果。Adaptive对C1的配对精确McNemar：NQ p=1、FEVER p=0.125；四项Holm后分别为1和0.375。Adaptive对Full的保守非劣检验Holm后为NQ0.4262、FEVER0.0840，均未确立预登记非劣。不能用单一bootstrap区间或较高的点估计替代检验结论，也不能把“未确立”写成“证明更差”。统计方法和全部区间保留在[evaluation.json](evaluation.json)。

## 人工审核与后续

- [NQ改答队列](nq_changed_answer_review.jsonl)：104个不同“原答→候选”对，覆盖84道题；按相同答案对合并来源组，保留初始Top5和各组最终证据。前14条来自Adaptive改答。所有人工字段pending，EM分数只作辅助。
- [路由重点队列](routing_priority_review.jsonl)：NQ15个、FEVER5个误修/漏修观察案例。优先确认NQ语义，避免把EM差异误当路由失败。
- [Adaptive警示和完整轨迹队列](adaptive_flagged_trace_review.jsonl)：51条（NQ29、FEVER22），保留完整运行记录。AC6需至少50个人工确认、观察精确率≥90%；至少50条轨迹质性审核的独立性/完成字段另记。两项均未验收，警示本身不是真值。

按预登记T5规则，当前NQ损失中漏修机会14大于可避免误修1，优先检查NQ路由漏修；E契约拒绝和No Probe结果用于解释及候选设计。尚未选定或晋升新候选，本次评分/分析未读取selection/final的gold，也未改阈值、提示、模型、知识库、训练或RSI。语义审核收口后只做一个集中候选，在独立选型集确认，随后冻结最终测试。没有证据支持当前直接把E晋升为已有效的默认策略。

## 重现

从仓库根目录，在现有完整依赖环境运行：

```bash
python artifacts/controller/t4_v1/reproduce.py --output-dir /tmp/controller-t4-offline-replay
```

输出目录必须不存在，防止覆盖人工字段。脚本只读本次ZIP、已封存工程凭证和development gold，验证哈希/配对/预算后重建汇总、冻结评测、损失分析和待审队列；没有模型加载或GPU调用。原始CLI评分与独立ZIP重放的JSON结果已逐对象一致核对，见[重放凭证](reproduction_receipt.json)。原始推理源码是42c7974；后续汇总修补不替换归档中的源码、原manifest或预测。
