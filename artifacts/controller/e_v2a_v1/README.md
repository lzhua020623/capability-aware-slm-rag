# E v2a 开发与 T5 路由设计

协议与任务：[执行修订](../../../docs/CONTROLLER_E_V2A_AND_T5_V1.md)。登记、实现、打包各轮均保留开工/收工自检；失败接收后完整回归为 813 项通过。

真实 [Kaggle Version 1](https://www.kaggle.com/code/yearo111/controller-e-v2a-development-frozen-20261025?scriptVersionId=362122804)已停止于第二道虚构行为题，耗时 268.9 秒。两次短输入调用都完成，支持保留题通过；错误原答题的 raw verdict 正确为 unsupported，但原判定证据 ID 为空，候选又复制原错误姓名、候选证据也为空，故契约失败。未运行 FEVER 或 14k 长输入门槛，真实开发预测 **0/800**。详见 [原始输出与校验](kaggle_v1/audit.json)；不能宣称联合长输入或真实合规率通过。后续仅另立接口修订，不覆盖本包或本次失败。

一次接口修正已[另行登记](../../../configs/controller/e_v2a_engineering_r1.json)：判定依据与替换答案改用无歧义的字段名，声明过的必需 ID 数量进入条件解码约束；模型、语义规则和预算不变。旧虚构题只作结构回归，下一次使用全新的 Selen/Veyra 虚构验收。该修订不是已经验证的收益，失败后停止，不围绕样例循环调判断提示。

当前 GPU 入口为 [transport_r2 短 Notebook](transport_r2/controller_e_v2a_development_v1_dataset.ipynb)，配套输入为同目录 `.tar.gz.bin`。它与 [原冻结压缩包](gpu_release_v1/controller_e_v2a_development_v1.tar.gz)字节相同，SHA256 为 `728c8ba512115825c3a4c362f5e4a51a3c860d74ad4d6e977dcaf93ca12adcf3`；源提交 `8560471`。`gpu_release_v1` 的嵌入式 Notebook 超出 Kaggle 1 MB 源码上限，不能直接提交，原文件保留。

私有输入 Dataset：[Controller E v2a frozen development 20261025](https://www.kaggle.com/datasets/yearo111/controller-e-v2a-frozen-development-20261025)。短 Notebook 读取后先验证完整压缩包与全部内部文件，再运行六道虚构行为题、两道至少 14k-token 的 XGrammar/prefill256 联合门槛，通过后才执行 800 次开发 E 调用。无真实 gold、选型或 final 输入，无运行时质量评分或调参。

[15 题审核网页](t5_development_review/routing15_review.html)含问题、原答、各动作输出、原 Top-5 和动作后证据；所有人工字段待确认。已登记阈值 0.2/0.3/0.4 的开发 EM 净增为 3/1/0，尚未冻结或晋升。五条用户指出的语义误修 ID 仍待核对。

[封存解包预检](validation/extracted_bundle_check.json)通过：89 个文件指纹、800 个进入 E 状态、23 项便携检查；不代表 CUDA 联合门槛或回答质量已通过。GPU 实际状态另存启动/验收凭据，不以页面导入或提交动作当作完成。
