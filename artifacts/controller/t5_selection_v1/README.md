# T5.9 一次选型运行包 · 2026-10-26

当前入口为 [gpu_release_v2/controller_t5_selection_v1_dataset.ipynb](gpu_release_v2/controller_t5_selection_v1_dataset.ipynb)，配套同目录 `.tar.gz.bin` 上传为私有 Dataset；另挂载已有私有 [T4 检索资源](https://www.kaggle.com/datasets/yearo111/controller-t4-retrieval-resources-frozen-20261024)，不重新上传 5.7GB 资源。

源提交 `7c281d0`，压缩包 SHA256 `151676a8c07008257086701c5dbced4fdc0b5e0f98f75f6e88caf98dcbae8f6f`，Notebook SHA256 `5e2c5050537702a780a2b628a882ee82cba3faef43b74d65b6c7836d8a9428a5`；唯一指针见 [current_release.json](current_release.json)。

已通过完整 956 项测试，实际解包后 45 项便携测试、105 项文件指纹以及 NQ／FEVER 各 200 条预检；包内无 QA gold。400 条原 Top-5 活动提示长度已检查，Probe 最长为 NQ 2,607、FEVER 3,337 tokens。以上是本地验证，不表示 GPU 检查或候选收益通过。

只运行一个已冻结的 0.2 候选的选型，不调参、不读取 final、不评分。四组原始输出＋缓存 C1 共 2,000 条；E 两数据集全局 unavailable，零 E 模型调用。开始前需通过六个 >=14k-token 的活动路径工程检查和检索资源历史 Top-5 复现。候选由固定动作及原 Adaptive 的已有 Probe 离线回放；候选不是本次直接执行的在线路由。具体协议见 [CONTROLLER_T5_SELECTION_V1.md](../../../docs/CONTROLLER_T5_SELECTION_V1.md)。

原 `gpu_release_v1` 没有上传／运行。实际解包发现便携测试缺少两项依赖，45 项中 5 项报错；原件及失败日志保留。v2 只补测试依赖，新增真实解包回归，候选、提示、预算、分支和输入不变，不是语义重试。

当前 GPU 尚未启动；上传和启动观察后另立回执，不把本地封存当成真实验收。
