# E v2a 接口修订 r1

本包是 [第一次失败](../e_v2a_v1/kaggle_v1/audit.json)之后、另行[登记的唯一接口修正](../../../configs/controller/e_v2a_engineering_r1.json)。原 V1 输出不覆盖。源提交 `fa8bfd6`，模型、NF4、384-token 单次预算、语义 RULES、数据划分不变；明确区分判定依据与替换答案，并用条件 grammar 强制必要 ID 数量。新虚构验收使用 Selen/Veyra，旧 Velora/Nerin 仅保留作结构回归。

运行入口为 [短 Notebook](transport_v1/controller_e_v2a_development_v1_dataset.ipynb)，配套同目录 `.tar.gz.bin`。压缩包 SHA256：`54cbca0d57d5d2733afac260c988f47a6bafef77dcef1bd56d3a188c638d1c50`。嵌入式 Notebook 仅为封存来源，不直接提交到 Kaggle。

本地完整 815 项回归通过；[实际源码的 CPU grammar 检查](../e_v2a_v1/validation/interface_r1_schema_cpu.json)覆盖六类虚构输入，合法结构可接受、已观察空 ID 组合被拒绝。独立[解包检查](validation/extracted_bundle_check.json)通过 90 个文件指纹、800 个输入及 25 项便携测试；800 个开发输入最长 6,522 tokens。以上不等于真实模型核验或 GPU 长输入验收通过。

GPU 仍须先通过六个新虚构行为检查和两条至少 14k-token 的 XGrammar/prefill256 联合检查，之后才允许执行 800 次开发 E。任一门槛失败即停止，不再围绕这些题调整判断提示，不使用选型或 final 数据。
