# E v2a 接口修订 r1

本包是 [第一次失败](../e_v2a_v1/kaggle_v1/audit.json)之后、另行[登记的唯一接口修正](../../../configs/controller/e_v2a_engineering_r1.json)。原 V1 输出不覆盖。源提交 `fa8bfd6`，模型、NF4、384-token 单次预算、语义 RULES、数据划分不变；明确区分判定依据与替换答案，并用条件 grammar 强制必要 ID 数量。新虚构验收使用 Selen/Veyra，旧 Velora/Nerin 仅保留作结构回归。

运行入口为 [短 Notebook](transport_v1/controller_e_v2a_development_v1_dataset.ipynb)，配套同目录 `.tar.gz.bin`。压缩包 SHA256：`54cbca0d57d5d2733afac260c988f47a6bafef77dcef1bd56d3a188c638d1c50`。嵌入式 Notebook 仅为封存来源，不直接提交到 Kaggle。

本地完整 815 项回归通过；[实际源码的 CPU grammar 检查](../e_v2a_v1/validation/interface_r1_schema_cpu.json)覆盖六类虚构输入，合法结构可接受、已观察空 ID 组合被拒绝。独立[解包检查](validation/extracted_bundle_check.json)通过 90 个文件指纹、800 个输入及 25 项便携测试；800 个开发输入最长 6,522 tokens。以上不等于真实模型核验或 GPU 长输入验收通过。

GPU 仍须先通过六个新虚构行为检查和两条至少 14k-token 的 XGrammar/prefill256 联合检查，之后才允许执行 800 次开发 E。任一门槛失败即停止，不再围绕这些题调整判断提示，不使用选型或 final 数据。

修正版已于 2026-10-25 提交至 [Kaggle 私有运行版本 362133401](https://www.kaggle.com/code/yearo111/controller-e-v2a-interface-r1-20261025?scriptVersionId=362133401)，页面确认 GPU T4 x2 / Running；[启动记录](launch_v1.json)保留当时观察。启动不代表工程验收通过，后续结果另外归档，不覆盖启动记录或第一次失败。

该版本于 234.8 秒停止，未进入真实开发集。三项 NQ 虚构输出的结构均有效，前两项“保留正确原答”“修复错误原答”通过；第三项缺失开馆年份，模型解释中正确指出证据没有该信息，却输出 `unsupported`，预登记期望为 `insufficient`。它保持原答 `2019`、没有生成替换，并标记未解决；这是核验类别不符，不是这道题发生误修。FEVER、两条长输入和 800 次开发调用均未执行，因此不能报告开发合规率、准确率或收益。

[原始 ZIP](kaggle_v1/outputs.zip)、全部原字节解包和[独立核验](kaggle_v1/audit.json)已保存；输出 SHA256 为 `d720d4fbfb9d0b43c6d6e0de995759a5a99c16b9a9495beb70aa64b08ffb4ce4`。审计通过表示导出完整、结果与冻结期望一致地被判失败，不表示工程验收通过。[部署决定](deployment_decision.json)按原登记记为 `unavailable`，保留原答并标记核验未完成；不将未执行的 FEVER 称为失败率 100%，也不将 NQ 三个虚构结构输出冒充真实开发合规率。

本轮到此停止，不放宽验收、不再修改这些题的判断提示并重跑。下一步需先澄清“缺少判定事实”与“证据明确证明原答错”的共同定义：当前 FEVER 提示明确写出缺失证据是 insufficient，NQ 的通用段落主要提到相关冲突，说明覆盖不对称是待核查的解释，单题输出尚不能证明因果。若后续另行登记定义修订，应视为新的语义修改，使用全新虚构题，并保留本轮失败。E 的 proposal 打通要求仍未验收。

独立复查命令（不调用模型、不读 gold）：

```bash
python scripts/audit_controller_e_v2a_outputs.py --archive artifacts/controller/e_v2a_interface_r1/kaggle_v1/outputs.zip --release-dir artifacts/controller/e_v2a_interface_r1/gpu_release_v1
```
