# 当前主线：10 月 24 日 Diagnosis · 2026-10-27

用户已明确要求后续以 10 月 24 日更新为主。来源为 `feature/diagnosis-probe-zheng@138d52aa22e0eee78637e01f981a030d7ad388df`，已通过历史合并进入 `controller`。本次建立当前基线登记、生产加载入口和真实增强适配器，不创建无内容的重复合并。

登记文件：[current_baseline.json](../configs/controller/current_baseline.json)。默认目标为 `diagnosis_20261024_p60_p64_v1`。旧 T4/T5 的源代码、输入、运行包和结果保留，独立复现时继续使用旧入口；不再以 T5 的 0.2 候选作为后续主线。

## 新流程

```text
缓存 C1 / DiagnosisInput
  -> FEVER 标签机械规范化
  -> 原层次 Diagnosis
  -> 原始 78 维在线特征提供器
  -> FrozenOnlineSelectionAdapter（P60 三个 HGB、原阈值）
  -> Controller.decide
  -> EnhancementExecutor
  -> 真实补检索 / 重排 / E 接口
```

新主线按上游设计关闭 Probe。P60 的选择门控、双通道分数和三个模型资产保持原样；本次上游没有更换 Diagnosis 的 27 维权重，P60 也仍保留 Diagnosis 判 correct 的题。

当前入口为 [src/pipeline/current.py](../src/pipeline/current.py) 的 `load_current_pipeline()`。它复用原 P64 host；`MeteredEnhancementAdapter` 注册三个真实动作，通过已有 `IntegratedController.execute()` 复用模块和计量，**不调用旧 `.run()`，不再次 Diagnosis、Probe 或重路由**。

`CheckedFeatures` 要求顺序一致的完整 78 项名称（75 个唯一名称），不去重、补零或用近似值替换。公开结果不暴露内部向量，保存输入/特征名称指纹及独立成本凭据。生产模式缺少当前输入绑定的特征调用凭据会停止，不能把未知成本记成零。

E r2 未通过，因此新主线的 E 仍统一 unavailable：接收路由后保留原答、零 E 调用、核验未完成。Diagnosis 切换不构成 E 重新验收。单动作模块预算登记为最多 1 次 Qwen、1 次检索、1 次评分，回答 64、E 上限 384 输出 tokens，分块预填充 256。特征计算单独计量；模块计量不能冒充全系统成本。

## 接入完成与真实运行就绪分开记录

代码和模型资产已合入，后续默认基线登记已切换；**目前尚不能启动可信的真实 P60 评测**。检查命令：

```bash
python scripts/check_controller_current.py
```

当前退出码 2，明确列出两项未交付依赖，不会回退旧路由或调用虚构 FeatureProvider：

1. 原 P26/P60 在线特征生成代码。仓库只有 `features_for(item, diagnosis)` 协议、模型和名称表，实现只存在于虚构测试；已有 27 维提取器不能直接拼成这 78 项。
2. 三个 P60 模型的训练、调参、校准、错误分析 ID 与来源。不能用已知 Diagnosis 划分代替这些 HGB 的来源证明。

重新 fetch 后 Diagnosis 分支头仍为 `138d52a`。Drive 搜索 P60/P26/diagnosis 均返回 `ACCESS_TOKEN_SCOPE_INSUFFICIENT`，不表示 Drive 中不存在文件。已询问用户交付位置；未向队友发送消息。

## 后续交付格式

`current_baseline.json.feature_provider` 目前为 null。交付原实现后登记 `module`、`factory`、`source_path`、`source_sha256`、`metered_runtime=true`。工厂接收共享 `execution_backend`；实例提供准确的 `feature_names`、`features_for(item, diagnosis)` 和 `runtime_receipt()`。生产 receipt 为 `{"input_sha256": "...", "calls": [...]}`；调用须有种类、是否尝试、状态、非负有限耗时及实际可追溯信息。

`training_provenance` 目前为 null。交付后登记 `path`、`sha256`，原始文件要求：

- `complete=true`：上游声明覆盖全部使用，不能据此假造 ID。
- `model_sha256`：准确绑定登记中的三个 HGB 资产。
- `records`：每项为 `model`、`dataset`、`sample_id`、`role`，role 为 training/tuning/calibration/error_analysis；三个模型均须覆盖。

程序逐 ID 核对完整 NQ 7,564、FEVER 1,886 保留池及当前 selection 均无交集。来源完整性仍需原作者交付和审查，不能仅提供“零重叠”的自填数字。

低层 `wire_latest_pipeline()` 支持显式注入提供器做工程测试；生产入口先检查资产、原实现及数据隔离，再加载 GPU 模型。两者不能混称生产验收。

## 后续顺序

先补原特征及三模型来源，完成真实特征、成本凭据、长输入与执行检查。随后单独登记新基线的比较组、Probe-OFF 变化、数据曝光、预算和指标，再确定独立评测与封存。已看过的 T5 selection 只可作已曝光的工程材料，不能重复作独立确认集。

新主线尚无质量结果；旧 T5 候选失败不能评价 P60。最终数据和完整保留池继续保护；RSI、蒸馏、DPO 不在本次接入中启动。验证见 [接入归档](../artifacts/controller/diagnosis_1024_v1/README.md)。
