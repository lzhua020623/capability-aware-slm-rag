# Controller T3 / T4 当前执行状态

2026-10-25 最新结果：[T4完整归档](../t4_v1/README.md)。2,800条配对预测已全部验收、自动评分并独立重放一致；末尾汇总KeyError导致Kaggle failure，预测完整，已修补并通过774项回归。Adaptive：NQ EM78→78，FEVER170→174；NQ语义及AC6/轨迹人工审核待完成。下面为10月24日工程验收与启动快照，[cloud_status.json](cloud_status.json)已更新为自动评测完成、人工待审。

2026-10-24：T3本地实现与执行前复查完成，773项完整回归通过。最终发布包解压后55项便携测试、400条输入预检与95个内容文件哈希全部通过；91个源码文件逐一匹配已提交的Git blob。[真实CUDA工程验收V1](https://www.kaggle.com/code/yearo111/controller-t3-cuda-engineering-frozen-20261024?scriptVersionId=361774673)已成功结束（960秒），输出已下载并通过本地验收。八条长输入实际最少16005 tokens，八条短流程均无调用失败或预算阻断，两个Full均为3次Qwen/1次检索/2次评分。CPU FP32实测5对8.913秒、20对34.496秒，保持原登记设备/精度/批量。[T4正式V2](https://www.kaggle.com/code/yearo111/controller-t3-cuda-engineering-frozen-20261024?scriptVersionId=361785012)正在配对开发推理；首题七组均已观察到记录、Full完成，日志已进入第2题。完整结果、执行seal尚未下载，尚未评分。[提交凭证](submission_v1.json)为工程首次运行中的历史观察；最新进度以[机器可读状态](cloud_status.json)为准。

GPU原始ZIP、运行环境、逐调用凭证及[本地验收](kaggle_engineering_v1/validation_receipt.json)位于`kaggle_engineering_v1/`。ZIP SHA256为99b34b1e65ee46877b02d2339f8f98b7c54a1b9945dca5e60b2b324f31252b7c。工程运行没有读取gold、没有生成T4问答预测，也不证明增强质量提高。短虚构题的0.421秒调用中位数不可外推到真实开发集。

[T4 Notebook](t4_release_v1/controller_t4_development_v1.ipynb)已编译校验，并在[启动登记](t4_release_v1/launch_receipt.json)绑定代码包、已验收CUDA凭证和原检索资源哈希。三个私有输入已附加，资源解压后为15个文件约8.77GB。[正式提交与运行观察](t4_release_v1/submission_v1.json)记录V2、原始环境、无持久化、GPU T4×2和首批日志。运行前55项便携测试再次通过（18.238秒），资源绑定校验通过；冻结代码只有在环境和两条历史Top5回放通过、保存execution_seal后才输出预测。日志至少9条是延迟观察下界，不当作当前完整进度。首题E的invalid_output按冻结规则保留原答并标记，不触发提示调整或重试；所有状态都要纳入最后评测。冻结策略/输入清单中的启动状态保持封存时原值，实际状态由新execution_seal及当前运行清单记录。

比较范围固定为两个数据集各200条development，七组共2800条配对记录；C1直接读缓存。人工类型组无可靠新标签，明确未执行；Oracle和路由回放离线计算，不新增GPU组。选型/最终题目及gold正文不进入GPU包。

实现与边界见[执行登记](../../../docs/CONTROLLER_T3_EXECUTION_V1.md)、[策略](../../../configs/controller/t3_execution_policy_v1.json)和[T5逐数据集晋升补充](../../../configs/controller/t5_selection_supplement_v1.json)。未修改Diagnosis/Probe/生成器/三个增强模块的原代码、参数与提示。Full真实串行执行且预算不重置；FEVER规范化标签并保留原文本；新增严格配对评分、Oracle、路由回放、保守非劣检验及四项Holm；警示标记不冒充语义真值。

`cpu_validation`、`package_validation`、`final_validation`分别保存先前762/771/772项检查；最新源码验收是[runtime_lock_validation](runtime_lock_validation/receipt.json)的773项回归。`gpu_v1`与`gpu_checked_v1`是本地传输预检快照，均未上传/未运行，保留以便追踪。唯一准备上传的发布包为[gpu_release_v1](gpu_release_v1/bundle_receipt.json)，源码提交42c7974453a60ab06194f958e65a0c18311c22d1，不能混用不同包的CUDA凭证。

上传代码包用[gpu_release_v1/controller_t3_gpu_v1.tar.gz](gpu_release_v1/controller_t3_gpu_v1.tar.gz)，工程Notebook用[controller_t3_engineering_v1.ipynb](gpu_release_v1/controller_t3_engineering_v1.ipynb)。代码包831,145字节，SHA256为ddd51973e9921402f914cf0ccb942e5450a8ca402f2bbcc58d1af2825415cffe；[独立解压验收](gpu_release_v1/validation_receipt.json)已通过。Notebook只启动工程验收，不运行T4问答。完整依赖版本、CUDA、模型量化与CPU线程配置均需与后续T4一致，否则拒绝复用凭证。

完整检索资源包只在本地忽略目录`data/processed/controller_resources/controller_t4_resource_bundle_v1.tar.gz`，5,700,167,663字节，SHA256为d8e84e8b0c4c472ab0c0fe23e3e79d24645456ab3322cb93a24fefe77272b5f4。解压流中14个成员逐一哈希通过，Git只保存资源清单和验收元数据。Linux/Python3.13九个固定直接依赖的wheel均已确认可下载；仍需GPU真实加载验收。

当前工程V1已验收，T4使用同一原始环境及CPU FP32评分配置运行。下一步下载完整输出和执行seal，核对2800条配对记录及预算，再做冻结离线评分、Oracle/路由/Probe/成本分析并生成NQ改答和AC6审核材料。不依据局部输出调整模块或提示；未下载完成结果前不报告准确率或净收益。
