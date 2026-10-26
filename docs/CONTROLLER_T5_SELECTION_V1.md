# T5.9 一次性选型执行 · 2026-10-26

唯一候选已在读取选型输入之前提交：`e814b6c`，[NQ threshold=0.2](../artifacts/controller/t5_candidate_v1/README.md)。本执行配置只实现已登记的四组与 E unavailable 决定，不新增阈值或语义修改。机器协议：[t5_selection_execution_v1.json](../configs/controller/t5_selection_execution_v1.json)。

NQ、FEVER 各 200 条选型输入，与开发和最终样本隔离。保存四组原始轨迹（Always Retrieve、Always Rerank、Always Repair、原 Adaptive）及缓存 C1，共 2,000 条记录。Always Repair 的 unavailable 策略不调用 E，不生成替换或伪造语义结论；C1 只复用原始预测。原 Adaptive 用于收集冻结 Diagnosis 与已有 Probe 的轨迹，候选只在离线按同输入固定动作回放；它不是本次直接执行的在线 Adaptive。回放成本单列为估计，不冒充实测部署延迟。

运行前六道 >=14k-token 的真实 Qwen 长输入路径检查覆盖两数据集的 Probe、补检索重答、重排重答；这三个路径的提示不改。另检查两数据集的 E unavailable 必须零模型调用、保留答案且记录核验未完成，并实测 CPU FP32 重排器 5／20 对耗时。工程检查不要求语义答案正确，也不宣称 E 验收通过。任一执行检查失败则保留导出，选型样本调用为 0。

复用 T4 的私有检索资源 Dataset，核对全部 14 个文件的大小与 SHA256，载入后复现两条旧查询的有序 Top-5。代码／配置／模型资产还与 r2 冻结核心源码核对；不改变 Diagnosis、Probe、Safe Binary、生成提示、检索、重排、模型权重或设备协议。所有活动生成均保留 native prefill256；重排仍为 CPU FP32 batch4，预算沿用原比较组。

运行包只有这 400 条选型输入、两条已曝光的历史检索检查和合成工程样例，没有 QA gold、开发输入或最终输入；离线 gold 只记录指纹，不打包、不评分。代码先提交再封存。禁止自动重试或见结果后修改输入／候选；所有前缀、错误、环境、执行封存和输出校验都导出保存。

选型之后才离线评分。NQ 冻结 EM 相对同一个 E 模式的原路由净增须 >0；候选相对原路由的所有实际改答需要语义审核，语义净值或待定最坏下界为负则保留原路由，不再切换到 0.3 或 0.4。FEVER 规则不变，并逐条核对回放输出一致。最终冻结与一次性六组评测尚未执行。

2026-10-26 结果登记：首次 Version 1 / 362316572 已完成，400 题／2,000 条完整，活动工程和资源复现检查通过。候选 NQ 冻结 EM 相对原 Adaptive 93→90，未满足正净增门槛，拒绝晋升、保留原路由；没有评估第二候选。FEVER 逐题不变。详见 [原始 ZIP、独立核验与结果](../artifacts/controller/t5_selection_v1/kaggle_v1/README.md)。该新增结果不覆盖运行前的冻结协议、输入或运行包；最终测试未开始。
