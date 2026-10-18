# NQ Train E 开发材料 v1

2026-10-17 接入全部 74,097 条 C1，首批固定 100 条待审核记录（80 条 EM=0 且参考段落命中候选，20 条 EM=1 对照）。全量保存在本地忽略目录，100 是审核批次规模。筛查条件不是 diagnosis 标签，所有人工字段待审。

上句“待审”描述冻结原包。2026-10-19 第四批后的累计状态为 47 条定案（11 个内部 E 目标、36 个正确对照）、22 条待定（含 5 条完整性／表达口径分歧）、31 条未审，见 [审核记录](../../../docs/ANSWER_REPAIR_NQ_TRAIN_REVIEW_20261017.md)。[原 92 条审核网页](review/NQ_train_pending_92.html) 保留同一包范围；可手动导入 [第四批累计对话状态](review/NQ_train_pending_92.chat_confirmations_batch04_20261019.json)，显示本页 39 条已确认、22 条待定、31 条空白。导入会替换页面本机草稿，其他浏览器审核需先导出备份。已导出的 [累计开发清单 v4](../../../configs/splits/answer_repair_nq_train_confirmed_dev_v4.json) 包含 47 条兼容输入，5 条口径分歧未纳入；这不是新实验结果。

本目录没有 GPU 修复结果，不代表 v8 已实现。`preflight.json` 只记录已有 v7 策略读取输入成功，completed=0。

详见 [接入说明与复现步骤](../../../docs/ANSWER_REPAIR_NQ_TRAIN.md)。审核入口为 [TXT 索引](review/review_queue.txt) / [含证据与离线参考的 JSONL](review/review_queue.jsonl)，全部来自 E 内部 development 部分。E 内部预留部分尚未建立团队共同未见性。

`preparation.json` 记录输入与队列哈希；其引用的冻结配置另记录源分片、源审计及所选 ID。辅助判断和用户确认另存到 `data/annotations/`，保留本包的原始字节。旧 v7 输出审核继续使用原队列。
