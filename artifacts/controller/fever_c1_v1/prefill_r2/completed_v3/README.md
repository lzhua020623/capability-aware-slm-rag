# FEVER C1 900 条采集完成 · 2026-10-24

Kaggle Version #3（361681141）成功完成并已下载验收。统一使用 `prefill_chunk_size=256`；本目录不拼接 Version #2 的 645 条旧输出。

| 用途 | 已完成 / 计划 |
|---|---:|
| development | 200 / 200 |
| selection | 200 / 200 |
| final_candidate | 500 / 500 |

全部 900 条与冻结题目、证据全文、顺序和原始生成记录匹配，无缺行或重复。输入归档的 77 文件、输出索引的 11 文件，以及云端打印的结果 ZIP 哈希全部核对通过；外层下载归档与内层结果 ZIP 中的文件字节一致。`completion_receipt.json` 和 `coverage_audit.json` 保存验收数字；`archive_audit.json` 保存归档校验链。

三条虚构工程输入检查通过，长输入实际为 **13,139 token**、生成耗时 **51.50 秒**。之前 OOM 的真实输入 `81950`（12,911 token）已完成；900 条实测输入 token 数均与原冻结 tokenizer 长度清单一致。实际环境为 Qwen2.5-7B-Instruct 固定 revision、196 个 NF4 模块、FP16 计算、double quant；参数位于 `cuda:0`。完整环境及原始工程 trace 在 `raw_outputs/environment.json`、`raw_outputs/engineering_smoke.json`。

环境中的 `generation_config` 保存模型默认值；实际调用由冻结源码显式覆盖为 `do_sample=False`、64 输出 token、seed 42 和 256-token 预填充。默认配置中的 `do_sample=True` 不表示本轮采用随机采样；有效调用口径另记在 completion receipt。

成功运行总时长 **3,224.4 秒（53 分 44 秒）**。900 次正式 C1 生成合计 3,000.29 秒，平均 3.334 秒/题；输入共 1,198,856 token，输出重分词估计 8,125 token（不含 EOS）。该口径不含 CPU Top5 检索。三次尝试的 Kaggle 运行时长合计 5,086.0 秒，包含两次失败，不能只以最后成功运行代表全部试跑成本。

900 条均可按原 FEVER parser 解析：795 条为纯标签，105 条带额外文字，没有同时出现两种不同标签的输出。额外文字原样保存，不据此修改 prompt、不直接算作 QA 错误。未读取 gold、未计算准确率，也未逐题展示或审核最终候选答案；本轮完成的是 C1 基线输入，不构成 E 或 Controller 的收益结论。

Controller 可使用 `raw_outputs/<role>/runtime.jsonl`；源码和输入指纹由上一级 collection 固定。仓库入口为 `configs/controller/fever_c1_completed_v1.json`，只登记 runtime 及验收哈希，gold 保持在原离线目录。

`kaggle_version3_outputs.zip` 保存完整原始下载，内含云端导出的 `controller_fever_c1_v1_outputs.zip`；解压的实验包代码副本不重复入库。原始运行文件在 `raw_outputs/`，原始云端日志为 `kaggle_version3.log`。

后续正式对照前仍需完成 T3 的各组接口、预算、评测器与执行 seal，并统一生成配置。本目录不把历史 T2 manifest 改写为全部就绪。

完整 CPU 回归 **744 项通过**，原 T2 的 **552 项来源哈希检查通过**，日志和凭证在 `validation/`。另有 7 项完成产物回归，包括删除一条 trace 后必须验收失败。
