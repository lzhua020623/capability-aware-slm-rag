# E v2a r2：统一核验标签 · GPU 运行包

协议见 [CONTROLLER_E_V2A_R2.md](../../../docs/CONTROLLER_E_V2A_R2.md)，验收题封存记录见 [fixture_review](fixture_review/README.md)。

## 运行包

源码提交为 `9d91a40`，压缩包 SHA256 为 `d4e8d3e6ba0901fd6581e8be05e96e86206ff998e38207b81b79efb98f6ad882`。

| 文件 | 用途 |
|---|---|
| [transport_v1/controller_e_v2a_r2_development_dataset.ipynb](transport_v1/controller_e_v2a_r2_development_dataset.ipynb) | 约 4 KB 的短 Notebook，提交到 Kaggle 运行 |
| `transport_v1/controller_e_v2a_r2_development.tar.gz.bin` | 上传为私有 Dataset；保留 `.bin` 后缀，防止被自动解压 |
| `gpu_release_v1/` | 原压缩包、嵌入式 Notebook 和执行清单，仅作封存，不直接提交 |

名称与 r1 的 `controller_e_v2a_development_v1` 不同。即使误挂了 r1 的 Dataset，Notebook 的指纹断言也会失败。

## 运行流程

在 Kaggle 上依次执行：

1. 安装依赖。
2. 运行便携单元测试。
3. 预检。
4. 运行 20 项验收：18 道短题和 2 道长题全部运行，并把失败分为执行、结构、语义三类。

然后按验收结果分支：

| 验收结果 | 后续运行 | 进程退出码 |
|---|---|---|
| 全部通过 | 800 次开发 E | 0 |
| 只有语义失败 | 两个数据集以 verify_only 只跑 Always Repair 400 次 | 0 |
| 有执行或结构失败 | 停止，E 不可用 | 3（Notebook 报错，但输出照常导出） |

无论哪种结果，都会导出 `controller_e_v2a_r2_development_outputs.zip`。

## 本地验证

以下均为 CPU 检查，**不等于** GPU 验收或语义正确：

- 完整回归测试通过。
- [真实 XGrammar 0.2.8 编译](validation/r2_schema_cpu.json)：使用 Qwen 分词器。18 道题的理想输出全部被接受；r1 的旧标签 `unsupported`、非 insufficient 判断配空的核验 ID，都会被拒绝。短题提示最长 1,307 tokens；两道长题分别为 17,514 和 17,539 tokens。
- [解包检查](validation/extracted_bundle_check.json)：92 个文件指纹一致；800 条开发输入中没有 gold 类字段；解包后 29 项便携测试和预检全部通过。

## 独立复核

拿到输出后，不需要调用模型、也不读 gold：

```bash
python scripts/audit_controller_e_v2a_outputs.py --archive <outputs.zip> --release-dir artifacts/controller/e_v2a_interface_r2/gpu_release_v1
```

审计脚本会根据原始记录重新计算每一项的失败类别和验收结果，并检查开发输出是否严格落在对应分支内：语义失败分支不得替换任何答案。

## 尚未发生

- GPU 尚未运行，开发预测数为 0。
- 全局 E 模式要等开发结果和用户审核后再离线决定。
- 误修上限的口径需要在打开开发结果之前由用户确认。
